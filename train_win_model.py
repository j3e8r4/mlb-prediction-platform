import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("historical_games.csv")
form = pd.read_csv("team_form.csv")
team_stats = pd.read_csv("team_stats.csv")

df["home_field_advantage"] = 0.4

away_form = form.rename(columns={
    "team": "away_team",
    "last_10_win_pct": "away_last_10_win_pct"
})

home_form = form.rename(columns={
    "team": "home_team",
    "last_10_win_pct": "home_last_10_win_pct"
})

df = df.merge(
    away_form[["away_team", "game_pk", "away_last_10_win_pct"]],
    on=["away_team", "game_pk"],
    how="left"
)

df = df.merge(
    home_form[["home_team", "game_pk", "home_last_10_win_pct"]],
    on=["home_team", "game_pk"],
    how="left"
)

df["away_last_10_win_pct"] = df["away_last_10_win_pct"].fillna(0.5)
df["home_last_10_win_pct"] = df["home_last_10_win_pct"].fillna(0.5)

away_team_stats = team_stats.add_prefix("away_")
home_team_stats = team_stats.add_prefix("home_")

df = df.merge(
    away_team_stats,
    left_on="away_team",
    right_on="away_team",
    how="left"
)

df = df.merge(
    home_team_stats,
    left_on="home_team",
    right_on="home_team",
    how="left"
)

numeric_cols = df.select_dtypes(include=["number"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

df_encoded = pd.get_dummies(
    df,
    columns=[
        "away_team",
        "home_team",
        "away_probable_pitcher",
        "home_probable_pitcher"
    ]
)

X = df_encoded.drop(
    columns=[
        "game_date",
        "away_score",
        "home_score",
        "home_win",
        "game_pk"
    ],
    errors="ignore"
)

y = df_encoded["home_win"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Win Model Accuracy: {accuracy:.2%}")

joblib.dump(model, "win_model.pkl")
joblib.dump(X.columns.tolist(), "win_model_columns.pkl")

print("Saved: win_model.pkl")
print("Saved: win_model_columns.pkl")