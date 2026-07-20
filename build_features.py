import pandas as pd

df = pd.read_csv("statcast_sample.csv")

# Keep only batted-ball rows with exit velocity
df = df.dropna(subset=["launch_speed"])

features = df.groupby("batter").agg(
    avg_exit_velocity=("launch_speed", "mean"),
    max_exit_velocity=("launch_speed", "max"),
    avg_launch_angle=("launch_angle", "mean"),
    batted_balls=("launch_speed", "count")
).reset_index()

features = features.sort_values("avg_exit_velocity", ascending=False)

print(features.head(20))

features.to_csv("batter_features.csv", index=False)

print("Saved: batter_features.csv")