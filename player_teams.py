import requests
import pandas as pd

def get_player_teams():
    url = "https://statsapi.mlb.com/api/v1/sports/1/players"
    params = {
        "season": 2025,
        "hydrate": "currentTeam"
    }

    data = requests.get(url, params=params).json()

    players = []

    for person in data.get("people", []):
        players.append({
            "player_id": person.get("id"),
            "player_name": person.get("fullName"),
            "team": person.get("currentTeam", {}).get("name")
        })

    df = pd.DataFrame(players)
    df.to_csv("player_teams.csv", index=False)

    print("Saved: player_teams.csv")
    print(df.head(20))

if __name__ == "__main__":
    get_player_teams()