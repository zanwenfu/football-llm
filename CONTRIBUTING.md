# Contributing to Football-LLM

Thanks for considering a contribution. This project is primarily a research artifact supporting the paper, so PRs are evaluated against two bars:

1. **Does it make reviewers trust the numbers more?** (new tests, better CI, tighter statistical protocol, reproducibility fixes)
2. **Does it make the paper's claims easier to extend?** (new tournaments, new baselines, new prompt regimes, new markets)

Cleanup PRs, doc fixes, and typo fixes are also welcome without ceremony.

## Getting set up

```bash
git clone https://github.com/zanwenfu/football-llm.git
cd football-llm
pip install -e ".[baselines,dev]"   # everything needed to run tests + regenerate paper
pre-commit install                  # hooks: ruff, black, detect-secrets
pytest                              # should show 74+ tests passing in ~2s
python scripts/reproduce_paper.py --skip-bootstrap  # smoke test
```

If you want to fine-tune locally or run vLLM:

```bash
pip install -e ".[all]"             # GPU required for train/vllm extras
```

## Ground rules

- **Don't weaken the statistical protocol.** Claims must continue to use Wilson CIs for proportions and paired exact McNemar for within-match comparisons. If a new metric is added, add a test locking in at least one expected value.
- **Add tests for new code.** We have 74+ tests and CI is green — keep it that way. For eval/metrics changes, prefer property-based or fixture-locked tests over hand-rolled asserts.
- **Don't commit secrets.** `.env` is gitignored and `detect-secrets` is a pre-commit hook. If you're adding a new env var, document it in `pyproject.toml` description + README + `api.py` docstring.
- **Keep the package installable.** `pip install -e ".[dev]" && pytest` must pass from a clean checkout. CI verifies this on Python 3.10 and 3.11.
- **Match the paper's sign-off discipline.** If your change affects headline numbers in the paper, either (a) the numbers don't change and existing tests still pass, or (b) the numbers change and you've updated `README.md`, the model card, and the relevant sections of `IDS598_Final_Project_Report.pdf` accordingly.

## Code style

- `ruff` handles linting + import sorting.
- `black` handles formatting (line length 100).
- Type annotations are encouraged but not required for data-prep scripts (legacy code is `# mypy: ignore-errors` for now).
- **Don't use emojis in code** unless explicitly requested by the user.

## What we're specifically looking for

Near-term asks that match the roadmap:

1. **Euro 2024 + Copa 2024 extension.** Adds ~83 matches (83 × 2 = 166 samples). Biggest single upgrade available — would push every borderline paired test below p=0.05 if effect sizes replicate.
2. **Fine-tuning on halftime+events prompts.** Currently pure inference-time template generalization. Training on the richer format is expected to preserve O/U gains and recover the −2.4pp halftime-only → halftime+events direction regression.
3. **Better in-play odds data.** Current backtest uses flat 1.90/1.90. Swapping in real time-stamped odds from Betfair Exchange / Pinnacle would convert the simulation into a more credible projection.
4. **Asian handicap conversion.** The magnitude/direction decomposition says the LLM should transfer to AH. Add the conversion + backtest.
5. **CMP or negative-binomial replacement of Poisson.** Football scorelines are mildly over-dispersed; §7 notes the Poisson approximation is conservative.

## Commit + PR conventions

- Write commit messages that explain **why**, not what (`git diff` already shows the what).
- One logical change per PR. If you're adding a new dataset AND a new baseline, that's two PRs.
- PRs that touch numbers should include a summary of which `reproduce_paper.py` outputs changed.

## License

MIT. By contributing, you agree your code is released under the same license.
