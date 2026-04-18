"""
Football-LLM: Gradio demo UI.

Interactive interface for predicting FIFA World Cup match outcomes.
Calls the FastAPI /predict endpoint running on localhost:8001.

Usage:
    python src/serving/demo_ui.py
"""

import json
import gradio as gr
import requests

API_URL = "http://localhost:8001"

# ---------------------------------------------------------------------------
# Preset examples for quick testing
# ---------------------------------------------------------------------------
EXAMPLES = {
    "Argentina vs France (2022 Final)": {
        "home_name": "Argentina", "home_goals": 450, "home_gp90": 0.35,
        "home_assists": 180, "home_rating": 7.2, "home_top": 200,
        "home_yellows": 120, "home_reds": 2, "home_tackles": 0.6,
        "home_duels": 55, "home_pass": 72, "home_formation": "4-3-3",
        "home_coach": "Lionel Scaloni",
        "away_name": "France", "away_goals": 420, "away_gp90": 0.38,
        "away_assists": 170, "away_rating": 7.3, "away_top": 190,
        "away_yellows": 100, "away_reds": 1, "away_tackles": 0.55,
        "away_duels": 54, "away_pass": 74, "away_formation": "4-2-3-1",
        "away_coach": "Didier Deschamps",
        "tournament": "World Cup 2022", "stage": "Final",
        "venue": "Lusail Stadium",
    },
    "Brazil vs Germany (2014 SF)": {
        "home_name": "Brazil", "home_goals": 350, "home_gp90": 0.30,
        "home_assists": 130, "home_rating": 6.9, "home_top": 80,
        "home_yellows": 140, "home_reds": 3, "home_tackles": 0.5,
        "home_duels": 50, "home_pass": 68, "home_formation": "4-2-3-1",
        "home_coach": "Luiz Felipe Scolari",
        "away_name": "Germany", "away_goals": 410, "away_gp90": 0.36,
        "away_assists": 200, "away_rating": 7.1, "away_top": 130,
        "away_yellows": 110, "away_reds": 1, "away_tackles": 0.58,
        "away_duels": 53, "away_pass": 75, "away_formation": "4-2-3-1",
        "away_coach": "Joachim Löw",
        "tournament": "World Cup 2014", "stage": "Semi Final",
        "venue": "Estádio Mineirão",
    },
    "England vs Iran (2022 Group)": {
        "home_name": "England", "home_goals": 308, "home_gp90": 0.32,
        "home_assists": 140, "home_rating": 7.1, "home_top": 109,
        "home_yellows": 95, "home_reds": 1, "home_tackles": 0.52,
        "home_duels": 52, "home_pass": 72, "home_formation": "4-3-3",
        "home_coach": "Gareth Southgate",
        "away_name": "Iran", "away_goals": 139, "away_gp90": 0.18,
        "away_assists": 60, "away_rating": 6.9, "away_top": 77,
        "away_yellows": 80, "away_reds": 2, "away_tackles": 0.48,
        "away_duels": 49, "away_pass": 64, "away_formation": "4-4-2",
        "away_coach": "Carlos Queiroz",
        "tournament": "World Cup 2022", "stage": "Group Stage",
        "venue": "Khalifa International Stadium",
    },
}


def predict_match(
    # Home team
    home_name, home_goals, home_gp90, home_assists, home_rating,
    home_top, home_yellows, home_reds, home_tackles, home_duels,
    home_pass, home_formation, home_coach,
    # Away team
    away_name, away_goals, away_gp90, away_assists, away_rating,
    away_top, away_yellows, away_reds, away_tackles, away_duels,
    away_pass, away_formation, away_coach,
    # Match context
    tournament, stage, venue,
):
    """Call the FastAPI /predict endpoint and format the result."""
    payload = {
        "home_team": {
            "name": home_name,
            "goals": int(home_goals),
            "goals_per_90": float(home_gp90),
            "assists": int(home_assists),
            "avg_rating": float(home_rating),
            "top_scorer_goals": int(home_top),
            "yellows": int(home_yellows),
            "reds": int(home_reds),
            "tackles_per_90": float(home_tackles),
            "duels_pct": float(home_duels),
            "pass_accuracy": float(home_pass),
            "formation": home_formation,
            "coach": home_coach,
        },
        "away_team": {
            "name": away_name,
            "goals": int(away_goals),
            "goals_per_90": float(away_gp90),
            "assists": int(away_assists),
            "avg_rating": float(away_rating),
            "top_scorer_goals": int(away_top),
            "yellows": int(away_yellows),
            "reds": int(away_reds),
            "tackles_per_90": float(away_tackles),
            "duels_pct": float(away_duels),
            "pass_accuracy": float(away_pass),
            "formation": away_formation,
            "coach": away_coach,
        },
        "match": {
            "tournament": tournament,
            "stage": stage,
            "venue": venue,
        },
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Format result card
        pred_emoji = {"home_win": "🏠", "away_win": "✈️", "draw": "🤝"}.get(
            data["prediction"], "❓"
        )
        pred_label = {
            "home_win": f"{home_name} Win",
            "away_win": f"{away_name} Win",
            "draw": "Draw",
        }.get(data["prediction"], data["prediction"])

        result_md = f"""## {pred_emoji} {pred_label}

### Score: {data['score']}

**{home_name}** {data['score'].split('-')[0].strip()} — {data['score'].split('-')[1].strip()} **{away_name}**

### Reasoning
{data['reasoning']}

---
<details>
<summary>Raw model output</summary>

```
{data['raw_output']}
```
</details>
"""
        return result_md

    except requests.exceptions.ConnectionError:
        return (
            "## ❌ Connection Error\n\n"
            "Could not connect to the Football-LLM API at `localhost:8001`.\n\n"
            "Make sure both servers are running:\n"
            "1. vLLM: `./src/serving/serve_vllm.sh`\n"
            "2. FastAPI: `uvicorn src.serving.api:app --port 8001`"
        )
    except requests.exceptions.HTTPError as e:
        return f"## ❌ API Error\n\n`{e}`\n\n```json\n{e.response.text}\n```"
    except Exception as e:
        return f"## ❌ Error\n\n`{str(e)}`"


def load_example(example_name):
    """Load preset example values into the form."""
    ex = EXAMPLES.get(example_name, {})
    if not ex:
        return [gr.update()] * 29
    return [
        ex["home_name"], ex["home_goals"], ex["home_gp90"],
        ex["home_assists"], ex["home_rating"], ex["home_top"],
        ex["home_yellows"], ex["home_reds"], ex["home_tackles"],
        ex["home_duels"], ex["home_pass"], ex["home_formation"],
        ex["home_coach"],
        ex["away_name"], ex["away_goals"], ex["away_gp90"],
        ex["away_assists"], ex["away_rating"], ex["away_top"],
        ex["away_yellows"], ex["away_reds"], ex["away_tackles"],
        ex["away_duels"], ex["away_pass"], ex["away_formation"],
        ex["away_coach"],
        ex["tournament"], ex["stage"], ex["venue"],
    ]


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

def create_ui():
    with gr.Blocks(
        title="Football-LLM — Match Predictor",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # ⚽ Football-LLM — Match Predictor
            Predict FIFA World Cup match outcomes using a fine-tuned Llama 3.1 8B model.
            Enter team statistics below or load a preset example.
            """
        )

        # Preset examples
        with gr.Row():
            example_dropdown = gr.Dropdown(
                choices=list(EXAMPLES.keys()),
                label="Load Preset Example",
                interactive=True,
            )
            load_btn = gr.Button("Load Example", variant="secondary")

        # --- Match context ---
        gr.Markdown("### 🏟️ Match Context")
        with gr.Row():
            tournament = gr.Textbox(label="Tournament", value="World Cup 2026")
            stage = gr.Textbox(label="Stage", value="Group Stage")
            venue = gr.Textbox(label="Venue", value="MetLife Stadium")

        # --- Team stats ---
        with gr.Row():
            # Home team
            with gr.Column():
                gr.Markdown("### 🏠 Home Team")
                home_name = gr.Textbox(label="Team Name", value="Argentina")
                home_coach = gr.Textbox(label="Coach", value="Lionel Scaloni")
                home_formation = gr.Textbox(label="Formation", value="4-3-3")
                with gr.Row():
                    home_goals = gr.Number(label="Total Goals", value=450)
                    home_gp90 = gr.Number(label="Goals/90", value=0.35)
                with gr.Row():
                    home_assists = gr.Number(label="Assists", value=180)
                    home_rating = gr.Number(label="Avg Rating", value=7.2)
                home_top = gr.Number(label="Top Scorer Goals", value=200)
                with gr.Row():
                    home_yellows = gr.Number(label="Yellows", value=120)
                    home_reds = gr.Number(label="Reds", value=2)
                with gr.Row():
                    home_tackles = gr.Number(label="Tackles/90", value=0.6)
                    home_duels = gr.Number(label="Duel Win %", value=55)
                home_pass = gr.Number(label="Pass Accuracy %", value=72)

            # Away team
            with gr.Column():
                gr.Markdown("### ✈️ Away Team")
                away_name = gr.Textbox(label="Team Name", value="Brazil")
                away_coach = gr.Textbox(label="Coach", value="Dorival Júnior")
                away_formation = gr.Textbox(label="Formation", value="4-2-3-1")
                with gr.Row():
                    away_goals = gr.Number(label="Total Goals", value=380)
                    away_gp90 = gr.Number(label="Goals/90", value=0.32)
                with gr.Row():
                    away_assists = gr.Number(label="Assists", value=150)
                    away_rating = gr.Number(label="Avg Rating", value=7.0)
                away_top = gr.Number(label="Top Scorer Goals", value=160)
                with gr.Row():
                    away_yellows = gr.Number(label="Yellows", value=130)
                    away_reds = gr.Number(label="Reds", value=3)
                with gr.Row():
                    away_tackles = gr.Number(label="Tackles/90", value=0.55)
                    away_duels = gr.Number(label="Duel Win %", value=52)
                away_pass = gr.Number(label="Pass Accuracy %", value=70)

        # --- Predict button ---
        predict_btn = gr.Button("⚽ Predict Match", variant="primary", size="lg")

        # --- Output ---
        output = gr.Markdown(label="Prediction")

        # --- Wire up events ---
        all_inputs = [
            home_name, home_goals, home_gp90, home_assists, home_rating,
            home_top, home_yellows, home_reds, home_tackles, home_duels,
            home_pass, home_formation, home_coach,
            away_name, away_goals, away_gp90, away_assists, away_rating,
            away_top, away_yellows, away_reds, away_tackles, away_duels,
            away_pass, away_formation, away_coach,
            tournament, stage, venue,
        ]

        predict_btn.click(fn=predict_match, inputs=all_inputs, outputs=output)

        load_btn.click(
            fn=load_example,
            inputs=example_dropdown,
            outputs=all_inputs,
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
