from pybaseball import statcast
import pandas as pd

print("Pulling MLB Statcast data...")

data = statcast(start_dt="2024-04-01", end_dt="2024-04-07")

print(data[["game_date", "batter", "pitcher", "launch_speed", "launch_angle", "events"]].head())

data.to_csv("statcast_sample.csv", index=False)

print("Saved file: statcast_sample.csv")