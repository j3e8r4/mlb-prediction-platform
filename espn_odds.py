import requests
import pandas as pd


def get_espn_odds():
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/"
        "baseball/mlb/scoreboard"
    )

    response = requests.get(url)
    data = response.json()

    rows = []

    for event in data.get("events", []):
        competition = event["competitions"][0]
        competitors = competition["competitors"]

        home_team = next(c["team"]["displayName"] for c in competitors if c["homeAway"] == "home")
        away_team = next(c["team"]["displayName"] for c in competitors if c["homeAway"] == "away")

        odds = competition.get("odds", [])

        if odds:
            odds_data = odds[0]

            rows.append({
                "away_team": away_team,
                "home_team": home_team,
                "details": odds_data.get("details"),
                "over_under": odds_data.get("overUnder")
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    odds = get_espn_odds()
    print(odds)
    odds.to_csv("espn_odds.csv", index=False)
    print("Saved: espn_odds.csv")
    