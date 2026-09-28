"""Dixon-Coles score-model baseline (Maher 1982; Dixon & Coles 1997).

This is the textbook benchmark for football scorelines and the one used in the
course lectures. Each team gets an attack strength α and a defence weakness β,
estimated by maximum likelihood on *past matches only*. A new fixture's
expected goals are then computed, not learned:

    log λ = base + γ + α_home + β_away      (home team's expected goals)
    log μ = base     + α_away + β_home      (away team's expected goals)

Goals are Poisson(λ) and Poisson(μ), with the Dixon-Coles τ(ρ) correction on
the four low-score cells (0-0, 1-0, 0-1, 1-1) where independent Poisson
misprices draws.

Why it belongs next to the LLM: it uses team identity *legitimately* — only
results that happened before the eval tournament. The fine-tuned LLM on named
prompts also uses team identity, but Llama 3.1's pretraining corpus contains
the 2022 results themselves. If the named LLM beats this model by a wide
margin on exact scores, the gap is recall, not skill.

Design choices (fixed before looking at 2022):
    * Trained on 90-minute scores. Extra-time goals are a different process,
      and O/U / 1X2 markets settle at 90 minutes.
    * L2 (ridge) penalty on α, β. 192 training matches over ~60 teams means
      many teams have three games; without shrinkage the MLE overfits.
      Teams never seen in training get α = β = 0 (tournament average).
    * The ridge strength is chosen by fitting on 2010+2014 and scoring the
      log-likelihood of 2018, then refit on all three tournaments.

Usage:
    python -m football_llm.baselines.dixon_coles
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from football_llm.eval.metrics import Result, result_from_score
from football_llm.paths import RAW_DIR, RESULTS_DIR

logger = logging.getLogger("football_llm.baselines.dixon_coles")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RESULT_LABELS: list[Result] = ["home_win", "draw", "away_win"]
MAX_GOALS = 10  # score matrix is (MAX_GOALS + 1) x (MAX_GOALS + 1)
# Wide on purpose: on World Cup-only history the validation optimum sits at
# heavy shrinkage (team strengths from one tournament barely transfer to the
# next), and a narrow grid would silently pick its own upper edge.
RIDGE_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
RHO_BOUNDS = (-0.3, 0.3)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DixonColesModel:
    teams: tuple[str, ...]
    attack: np.ndarray  # α, one per team
    defence: np.ndarray  # β, one per team (higher = concedes more)
    base: float
    home: float  # γ
    rho: float
    ridge: float

    def _strength(self, team: str) -> tuple[float, float]:
        if team in self.teams:
            i = self.teams.index(team)
            return float(self.attack[i]), float(self.defence[i])
        return 0.0, 0.0  # unseen team → tournament average

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        a_h, d_h = self._strength(home_team)
        a_a, d_a = self._strength(away_team)
        lam = np.exp(self.base + self.home + a_h + d_a)
        mu = np.exp(self.base + a_a + d_h)
        return float(lam), float(mu)

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """P(home = i, away = j) for i, j in 0..MAX_GOALS, renormalised."""
        lam, mu = self.expected_goals(home_team, away_team)
        goals = np.arange(MAX_GOALS + 1)
        m = np.outer(stats.poisson.pmf(goals, lam), stats.poisson.pmf(goals, mu))
        m[0, 0] *= 1 - lam * mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[1, 1] *= 1 - self.rho
        m = np.clip(m, 0.0, None)
        return m / m.sum()


def _tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score adjustment τ(x, y)."""
    tau = np.ones_like(lam)
    tau = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return tau


def fit(matches: pd.DataFrame, ridge: float = 1.0) -> DixonColesModel:
    """Maximum-likelihood fit with an L2 penalty on attack/defence strengths.

    `matches` needs columns home_team, away_team, home_goals, away_goals.
    """
    teams = tuple(sorted(set(matches["home_team"]) | set(matches["away_team"])))
    idx = {t: i for i, t in enumerate(teams)}
    h = matches["home_team"].map(idx).to_numpy()
    a = matches["away_team"].map(idx).to_numpy()
    x = matches["home_goals"].to_numpy(dtype=float)
    y = matches["away_goals"].to_numpy(dtype=float)
    n = len(teams)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        return params[:n], params[n : 2 * n], params[2 * n], params[2 * n + 1], params[2 * n + 2]

    def neg_log_lik(params: np.ndarray) -> float:
        att, dfn, base, home, rho = unpack(params)
        lam = np.exp(base + home + att[h] + dfn[a])
        mu = np.exp(base + att[a] + dfn[h])
        ll = stats.poisson.logpmf(x, lam) + stats.poisson.logpmf(y, mu)
        ll += np.log(np.clip(_tau(x, y, lam, mu, rho), 1e-10, None))
        return float(-ll.sum() + ridge * (att @ att + dfn @ dfn))

    mean_goals = (x.sum() + y.sum()) / (2 * len(matches))
    x0 = np.concatenate([np.zeros(2 * n), [np.log(mean_goals), 0.0, 0.0]])
    bounds = [(None, None)] * (2 * n + 2) + [RHO_BOUNDS]
    res = optimize.minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        logger.warning("Dixon-Coles optimiser did not converge: %s", res.message)
    att, dfn, base, home, rho = unpack(res.x)
    return DixonColesModel(
        teams=teams,
        attack=att,
        defence=dfn,
        base=float(base),
        home=float(home),
        rho=float(rho),
        ridge=ridge,
    )


def mean_log_likelihood(model: DixonColesModel, matches: pd.DataFrame) -> float:
    """Average log-probability the model assigns to the observed scorelines."""
    logps = []
    for row in matches.itertuples(index=False):
        m = model.score_matrix(row.home_team, row.away_team)
        i, j = min(int(row.home_goals), MAX_GOALS), min(int(row.away_goals), MAX_GOALS)
        logps.append(np.log(max(m[i, j], 1e-12)))
    return float(np.mean(logps))


def select_ridge(train: pd.DataFrame, grid: Sequence[float] = RIDGE_GRID) -> float:
    """Pick the ridge strength on the last training tournament, never the eval one."""
    val_year = int(train["world_cup_year"].max())
    inner, val = (
        train[train["world_cup_year"] < val_year],
        train[train["world_cup_year"] == val_year],
    )
    scores = {r: mean_log_likelihood(fit(inner, ridge=r), val) for r in grid}
    for r, s in scores.items():
        logger.info("  ridge=%-5g  mean log-lik on %d = %.4f", r, val_year, s)
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_matches(raw_dir: Path | None = None) -> pd.DataFrame:
    """World Cup matches with both 90-minute and final (incl. extra time) scores."""
    df = pd.read_csv((raw_dir or RAW_DIR) / "world_cup_matches.csv")
    return pd.DataFrame(
        {
            "fixture_id": df["fixture_id"].astype(int),
            "world_cup_year": df["world_cup_year"].astype(int),
            "home_team": df["home_team_name"],
            "away_team": df["away_team_name"],
            # 90-minute score — what the model is trained on.
            "home_goals": df["fulltime_home"].astype(int),
            "away_goals": df["fulltime_away"].astype(int),
            # Final score incl. extra time — the label the LLM eval files use.
            "final_home": df["home_goals"].astype(int),
            "final_away": df["away_goals"].astype(int),
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict(model: DixonColesModel, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Per-fixture probabilities plus the point predictions the metrics code expects.

    pred_home/pred_away is the single most likely scoreline; pred_result is the
    argmax of the 1X2 probabilities (these can disagree — a 1-1 mode with a
    home-win argmax is common). p_over_25 comes from the full score matrix.
    """
    rows = []
    goals = np.arange(MAX_GOALS + 1)
    total = goals[:, None] + goals[None, :]
    for fx in fixtures.itertuples(index=False):
        m = model.score_matrix(fx.home_team, fx.away_team)
        lam, mu = model.expected_goals(fx.home_team, fx.away_team)
        p = {
            "home_win": float(np.tril(m, -1).sum()),
            "draw": float(np.trace(m)),
            "away_win": float(np.triu(m, 1).sum()),
        }
        i, j = np.unravel_index(np.argmax(m), m.shape)
        rows.append(
            {
                "fixture_id": int(fx.fixture_id),
                "anonymized": False,
                "home_team": fx.home_team,
                "away_team": fx.away_team,
                "gt_home": int(fx.final_home),
                "gt_away": int(fx.final_away),
                "gt_result": result_from_score(int(fx.final_home), int(fx.final_away)),
                "pred_result": max(RESULT_LABELS, key=p.get),
                "pred_home": int(i),
                "pred_away": int(j),
                "lambda_home": lam,
                "lambda_away": mu,
                "p_home_win": p["home_win"],
                "p_draw": p["draw"],
                "p_away_win": p["away_win"],
                "p_over_25": float(m[total > 2.5].sum()),
            }
        )
    return pd.DataFrame(rows)


def train_and_evaluate(eval_year: int = 2022) -> tuple[DixonColesModel, pd.DataFrame]:
    matches = load_matches()
    train = matches[matches["world_cup_year"] < eval_year]
    eval_ = matches[matches["world_cup_year"] == eval_year]
    logger.info("Selecting ridge on held-out %d …", int(train["world_cup_year"].max()))
    ridge = select_ridge(train)
    model = fit(train, ridge=ridge)
    logger.info(
        "Fitted on %d matches (%d teams): ridge=%g base=%.3f home=%.3f rho=%.3f",
        len(train),
        len(model.teams),
        ridge,
        model.base,
        model.home,
        model.rho,
    )
    unseen = sorted((set(eval_["home_team"]) | set(eval_["away_team"])) - set(model.teams))
    if unseen:
        logger.info("Teams in %d with no training matches (α=β=0): %s", eval_year, unseen)
    return model, predict(model, eval_)


def save_predictions(df: pd.DataFrame, results_dir: Path | None = None) -> Path:
    out_dir = results_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dixon_coles_predictions_pregame.json"
    df.to_json(path, orient="records", indent=2)
    logger.info("Wrote %d predictions to %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _summary(df: pd.DataFrame) -> None:
    from football_llm.eval.metrics import brier_score, goal_mae, wilson_ci

    n = len(df)
    correct = int((df["pred_result"] == df["gt_result"]).sum())
    exact = int(((df["pred_home"] == df["gt_home"]) & (df["pred_away"] == df["gt_away"])).sum())
    gt_over = (df["gt_home"] + df["gt_away"]) > 2.5
    ou_correct = int(((df["p_over_25"] > 0.5) == gt_over).sum())
    print(f"  Result accuracy:  {wilson_ci(correct, n)}")
    print(f"  Score exact:      {wilson_ci(exact, n)}")
    print(
        f"  Goal MAE (λ, μ):  "
        f"{goal_mae(df['lambda_home'], df['lambda_away'], df['gt_home'], df['gt_away']):.3f}"
    )
    print(f"  O/U 2.5 (p>0.5):  {wilson_ci(ou_correct, n)}")
    print(f"  O/U 2.5 Brier:    {brier_score(df['p_over_25'], gt_over):.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-year", type=int, default=2022)
    parser.add_argument("--no-save", action="store_true", help="Print metrics only")
    args = parser.parse_args()

    print(f"\n=== Dixon-Coles baseline — pregame, eval {args.eval_year} ===")
    _, df = train_and_evaluate(args.eval_year)
    _summary(df)
    if not args.no_save:
        save_predictions(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
