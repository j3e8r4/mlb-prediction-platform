import pandas as pd
from pybaseball import playerid_reverse_lookup

df = pd.read_csv("batter_features.csv")

player_ids = df["batter"].dropna().astype(int).unique().tolist()

names = playerid_reverse_lookup(player_ids, key_type="mlbam")

names["player_name"] = names["name_first"] + " " + names["name_last"]

df = df.merge(
    names[["key_mlbam", "player_name"]],
    left_on="batter",
    right_on="key_mlbam",
    how="left"
)

df = df.drop(columns=["key_mlbam"])

df.to_csv("batter_features_named.csv", index=False)

print("Saved: batter_features_named.csv")
print(df[["batter", "player_name", "avg_exit_velocity"]].head(20))