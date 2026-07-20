import pandas as pd

df = pd.read_csv("statcast_sample.csv")

df = df.dropna(subset=["launch_speed"])

pitcher_features = df.groupby("pitcher").agg(
    avg_exit_velocity_allowed=("launch_speed", "mean"),
    max_exit_velocity_allowed=("launch_speed", "max"),
    avg_launch_angle_allowed=("launch_angle", "mean"),
    batted_balls_allowed=("launch_speed", "count")
).reset_index()

pitcher_features = pitcher_features.sort_values(
    "avg_exit_velocity_allowed",
    ascending=False
)

print(pitcher_features.head(20))

pitcher_features.to_csv("pitcher_features.csv", index=False)

print("Saved: pitcher_features.csv")