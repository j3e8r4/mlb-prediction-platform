import pandas as pd

df = pd.read_csv("statcast_sample.csv")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nImportant columns:")
print(df[["game_date", "batter", "pitcher", "launch_speed", "launch_angle", "events"]].head(20))

print("\nAverage exit velocity:")
print(df["launch_speed"].mean())