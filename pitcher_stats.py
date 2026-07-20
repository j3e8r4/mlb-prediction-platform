import pandas as pd
from pybaseball import playerid_reverse_lookup


def build_pitcher_stats():
    df = pd.read_csv("statcast_sample.csv")

    df = df.dropna(subset=["pitcher", "launch_speed"])

    pitcher_stats = df.groupby("pitcher").agg(
        pitcher_avg_exit_velocity_allowed=("launch_speed", "mean"),
        pitcher_max_exit_velocity_allowed=("launch_speed", "max"),
        pitcher_batted_balls_allowed=("launch_speed", "count")
    ).reset_index()

    pitcher_stats = pitcher_stats.rename(
        columns={"pitcher": "pitcher_id"}
    )

    pitcher_ids = pitcher_stats["pitcher_id"].astype(int).tolist()

    names = playerid_reverse_lookup(
        pitcher_ids,
        key_type="mlbam"
    )

    names["pitcher_name"] = (
        names["name_first"] + " " + names["name_last"]
    )

    pitcher_stats = pitcher_stats.merge(
        names[["key_mlbam", "pitcher_name"]],
        left_on="pitcher_id",
        right_on="key_mlbam",
        how="left"
    )

    pitcher_stats = pitcher_stats.drop(
        columns=["key_mlbam"]
    )

    pitcher_stats.to_csv("pitcher_stats.csv", index=False)

    print("Saved: pitcher_stats.csv")
    print(pitcher_stats.head())


if __name__ == "__main__":
    build_pitcher_stats()