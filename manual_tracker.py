import pandas as pd
import os

MANUAL_FILE = "manual_pick_tracker.csv"


def load_manual_tracker():
    if os.path.exists(MANUAL_FILE):
        return pd.read_csv(MANUAL_FILE)

    return pd.DataFrame(columns=[
        "game_date",
        "game",
        "pick",
        "confidence",
        "result"
    ])


def save_manual_tracker(df):
    df.to_csv(MANUAL_FILE, index=False)