import requests
import pandas as pd
from datetime import date


def get_mlb_games(start_date="2024-03-28", end_date=None):
    if end_date is None:
     end_date = date.today().strftime("%Y-%m-%d")
    url = "https://statsapi.mlb.com/api/v1/schedule"

    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "probablePitcher,team"
    }

    response = requests.get(url, params=params)
    data = response.json()

    games = []

    for d in data.get("dates", []):
        for game in d.get("games", []):
            status = game.get("status", {}).get("detailedState", "")

            if status != "Final":
                continue

            away = game["teams"]["away"]
            home = game["teams"]["home"]

            away_team = away["team"]["name"]
            home_team = home["team"]["name"]

            away_score = away.get("score", 0)
            home_score = home.get("score", 0)

            home_win = 1 if home_score > away_score else 0

            games.append({
               "game_date": game.get("gameDate"),
               "away_team": away_team,
               "home_team": home_team,
               "away_probable_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD"),
               "home_probable_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD"),
               "away_score": away_score,
               "home_score": home_score,
               "home_win": home_win,
               "game_pk": game.get("gamePk")
})
    return pd.DataFrame(games)


if __name__ == "__main__":
    df = get_mlb_games()

    print(df.head())
    print("Rows:", len(df))

    df.to_csv("historical_games.csv", index=False)

    print("Saved: historical_games.csv")