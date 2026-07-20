import requests
import pandas as pd
from datetime import date
from pybaseball import statcast_batter_exitvelo_barrels


def get_games(game_date=None):
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    url = "https://statsapi.mlb.com/api/v1/schedule"

    params = {
        "sportId": 1,
        "startDate": game_date,
        "endDate": game_date,
        "hydrate": "probablePitcher"
    }
    
    response = requests.get(url, params=params)

    if response.status_code != 200:
        return pd.DataFrame()

    data = response.json()
    games = []

    if "dates" not in data or len(data["dates"]) == 0:
        return pd.DataFrame()

    for d in data["dates"]:
        for game in d["games"]:
            away = game["teams"]["away"]
            home = game["teams"]["home"]

            away_team = away["team"]["name"]
            home_team = home["team"]["name"]

            away_pitcher_obj = away.get("probablePitcher", {})
            home_pitcher_obj = home.get("probablePitcher", {})

            games.append({
                "game_date": game_date,
                "away_team": away_team,
                "home_team": home_team,
                "away_probable_pitcher": away_pitcher_obj.get("fullName", "TBD"),
                "home_probable_pitcher": home_pitcher_obj.get("fullName", "TBD"),
                "away_probable_pitcher_id": away_pitcher_obj.get("id"),
                "home_probable_pitcher_id": home_pitcher_obj.get("id")
            })

    return pd.DataFrame(games)


def get_hr_hitters(year=2025):
    hitters = statcast_batter_exitvelo_barrels(year, minBBE=25)

    teams = pd.read_csv("player_teams.csv")

    possible_cols = [
        "last_name, first_name",
        "player_id",
        "avg_hit_speed",
        "max_hit_speed",
        "brl_percent",
        "hard_hit_percent",
        "anglesweetspotpercent"
    ]

    cols = [c for c in possible_cols if c in hitters.columns]
    hitters = hitters[cols].copy()

    hitters["hr_score"] = (
        hitters.get("brl_percent", 0) * 3
        + hitters.get("hard_hit_percent", 0) * 2
        + hitters.get("avg_hit_speed", 0)
        + hitters.get("max_hit_speed", 0) * 0.25
    )

    min_score = hitters["hr_score"].min()
    max_score = hitters["hr_score"].max()

    if max_score > min_score:
        hitters["hr_confidence"] = (
            (hitters["hr_score"] - min_score)
            / (max_score - min_score)
            * 100
        )
    else:
        hitters["hr_confidence"] = 0

    hitters = hitters.merge(
        teams,
        on="player_id",
        how="left"
    )

    hitters = hitters.sort_values("hr_score", ascending=False)

    return hitters