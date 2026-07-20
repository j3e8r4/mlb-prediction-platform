import requests
import pandas as pd


def get_team_strikeout_profile(season=2025):
    url = "https://statsapi.mlb.com/api/v1/teams/stats"

    params = {
        "sportIds": 1,
        "stats": "season",
        "group": "hitting",
        "season": season
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return pd.DataFrame()

    data = response.json()

    rows = []

    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            team = split.get("team", {}).get("name")
            stat = split.get("stat", {})

            strikeouts = float(stat.get("strikeOuts", 0))
            games = float(stat.get("gamesPlayed", 1))

            k_per_game = strikeouts / games if games else 0

            rows.append({
                "team": team,
                "team_batter_k_per_game": k_per_game
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        league_avg = df["team_batter_k_per_game"].mean()

        df["opponent_k_factor"] = (
            df["team_batter_k_per_game"] / league_avg
        )

    return df


if __name__ == "__main__":
    df = get_team_strikeout_profile()
    print(df.sort_values("team_batter_k_per_game", ascending=False))
    df.to_csv("team_strikeout_profile.csv", index=False)