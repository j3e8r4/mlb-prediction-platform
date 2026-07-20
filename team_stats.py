import requests
import pandas as pd


def get_team_stats(season=2025):
    url = "https://statsapi.mlb.com/api/v1/teams/stats"

    rows = []

    for group in ["hitting", "pitching"]:
        params = {
            "sportIds": 1,
            "stats": "season",
            "group": group,
            "season": season
        }

        data = requests.get(url, params=params).json()

        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                team = split.get("team", {}).get("name")
                stat = split.get("stat", {})

                row = {"team": team}

                if group == "hitting":
                    row.update({
                        "runs_per_game": float(stat.get("runs", 0)) / max(float(stat.get("gamesPlayed", 1)), 1),
                        "team_avg": float(stat.get("avg", 0)),
                        "team_obp": float(stat.get("obp", 0)),
                        "team_slg": float(stat.get("slg", 0)),
                        "team_ops": float(stat.get("ops", 0)),
                        "team_strikeouts": float(stat.get("strikeOuts", 0)),
                    })

                if group == "pitching":
                    row.update({
                        "team_era": float(stat.get("era", 0)),
                        "team_whip": float(stat.get("whip", 0)),
                        "team_pitching_k9": float(stat.get("strikeoutsPer9Inn", 0)),
                        "team_pitching_hr9": float(stat.get("homeRunsPer9", 0)),
                    })

                rows.append(row)

    df = pd.DataFrame(rows)

    hitting = df[df.columns[df.columns.str.contains("team|runs|avg|obp|slg|ops|strikeouts")]].dropna(subset=["team"])
    pitching = df[df.columns[df.columns.str.contains("team|era|whip|pitching")]].dropna(subset=["team"])

    final = hitting.merge(pitching, on="team", how="outer")
    final.to_csv("team_stats.csv", index=False)

    print("Saved: team_stats.csv")
    print(final.head())


if __name__ == "__main__":
    get_team_stats()