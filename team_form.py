import pandas as pd


def build_team_form():
    df = pd.read_csv("historical_games.csv")

    rows = []

    for team in pd.concat([df["away_team"], df["home_team"]]).unique():
        team_games = df[
            (df["away_team"] == team) |
            (df["home_team"] == team)
        ].copy()

        team_games["game_date"] = pd.to_datetime(team_games["game_date"])
        team_games = team_games.sort_values("game_date")

        wins = []

        for _, row in team_games.iterrows():
            is_home = row["home_team"] == team

            if is_home:
                won = row["home_win"] == 1
            else:
                won = row["home_win"] == 0

            rows.append({
                "team": team,
                "game_pk": row["game_pk"],
                "last_10_win_pct": sum(wins[-10:]) / len(wins[-10:]) if wins else 0.5
            })

            wins.append(1 if won else 0)

    form = pd.DataFrame(rows)
    form.to_csv("team_form.csv", index=False)

    print("Saved: team_form.csv")


if __name__ == "__main__":
    build_team_form()