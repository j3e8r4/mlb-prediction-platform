import pandas as pd


def get_current_team_form():
    df = pd.read_csv("historical_games.csv")

    rows = []

    teams = pd.concat([df["away_team"], df["home_team"]]).unique()

    for team in teams:
        team_games = df[
            (df["away_team"] == team) |
            (df["home_team"] == team)
        ].copy()

        team_games["game_date"] = pd.to_datetime(team_games["game_date"])
        team_games = team_games.sort_values("game_date")

        last_10 = team_games.tail(10)

        wins = []

        for _, row in last_10.iterrows():
            if row["home_team"] == team:
                won = row["home_win"] == 1
            else:
                won = row["home_win"] == 0

            wins.append(1 if won else 0)

        win_pct = sum(wins) / len(wins) if wins else 0.5

        rows.append({
            "team": team,
            "current_last_10_win_pct": win_pct
        })

    return pd.DataFrame(rows)