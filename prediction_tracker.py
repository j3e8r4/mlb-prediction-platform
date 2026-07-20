import pandas as pd
from datetime import datetime
import os


LOG_FILE = "prediction_log.csv"


def normalize_team_name(name):
    return str(name).lower().replace(".", "").replace(" ", "").strip()


def save_prediction(
    game_date,
    away_team,
    home_team,
    predicted_winner,
    away_win_prob,
    home_win_prob
):
    row = {
        "timestamp": datetime.now(),
        "game_date": game_date,
        "away_team": away_team,
        "home_team": home_team,
        "predicted_winner": predicted_winner,
        "away_win_prob": away_win_prob,
        "home_win_prob": home_win_prob,
    }

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)

        duplicate = (
            (df["game_date"] == game_date) &
            (df["away_team"] == away_team) &
            (df["home_team"] == home_team)
        )

        if duplicate.any():
            return

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(LOG_FILE, index=False)


def score_predictions():
    predictions = pd.read_csv("prediction_log.csv")
    results = pd.read_csv("historical_games.csv")

    predictions["game_date"] = pd.to_datetime(
        predictions["game_date"]
    ).dt.date.astype(str)

    results["game_date"] = pd.to_datetime(
        results["game_date"]
    ).dt.date.astype(str)

    predictions["matchup_key"] = (
        predictions["game_date"]
        + "_"
        + predictions["away_team"].apply(normalize_team_name)
        + "_"
        + predictions["home_team"].apply(normalize_team_name)
    )

    results["matchup_key"] = (
        results["game_date"]
        + "_"
        + results["away_team"].apply(normalize_team_name)
        + "_"
        + results["home_team"].apply(normalize_team_name)
    )

    results["actual_winner_result"] = results.apply(
        lambda row: row["home_team"] if row["home_win"] == 1 else row["away_team"],
        axis=1
    )

    merged = predictions.merge(
        results[["matchup_key", "actual_winner_result"]],
        on="matchup_key",
        how="left"
    )

    merged["correct"] = (
        merged["predicted_winner"].apply(normalize_team_name)
        == merged["actual_winner_result"].apply(normalize_team_name)
    )

    merged.to_csv("prediction_log_scored.csv", index=False)

    print("Total predictions:", len(merged))
    print("Matched results:", merged["actual_winner_result"].notna().sum())

    scored_only = merged[merged["actual_winner_result"].notna()]

    if len(scored_only) > 0:
        print("Accuracy:", scored_only["correct"].mean())
    else:
        print("Accuracy: No completed games matched yet")