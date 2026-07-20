import requests
import pandas as pd
from team_strikeout_profile import get_team_strikeout_profile


def get_pitcher_season_stats(player_id, season=2025):

    if pd.isna(player_id):
        return None

    url = f"https://statsapi.mlb.com/api/v1/people/{int(player_id)}/stats"

    params = {
        "stats": "season",
        "group": "pitching",
        "season": season
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None

    data = response.json()
    stats_list = data.get("stats", [])

    if not stats_list:
        return None

    splits = stats_list[0].get("splits", [])

    if not splits:
        return None

    stat = splits[0].get("stat", {})

    strikeouts = float(stat.get("strikeOuts", 0))
    games_started = float(stat.get("gamesStarted", 0))

    if games_started <= 0:
        avg_k_per_start = 0
    else:
        avg_k_per_start = strikeouts / games_started

    return {
        "strikeouts": strikeouts,
        "games_started": games_started,
        "avg_k_per_start": avg_k_per_start,
        "era": stat.get("era"),
        "whip": stat.get("whip")
    }


def predict_pitcher_strikeouts(
    player_id,
    opponent_team=None,
    season=2025
):

    stats = get_pitcher_season_stats(player_id, season)

    if stats is None:
        return None

    base_k = stats["avg_k_per_start"]

    opponent_k_factor = 1.0
    opponent_k_per_game = None

    if opponent_team:
        profile = get_team_strikeout_profile(season)

        if not profile.empty:
            match = profile[profile["team"] == opponent_team]

            if not match.empty:
                opponent_k_factor = match.iloc[0]["opponent_k_factor"]
                opponent_k_per_game = match.iloc[0]["team_batter_k_per_game"]

    expected_k = base_k * opponent_k_factor

    expected_k = max(1.0, min(expected_k, 12.0))

    stats["expected_strikeouts"] = expected_k
    stats["opponent_k_factor"] = opponent_k_factor
    stats["opponent_k_per_game"] = opponent_k_per_game

    return stats