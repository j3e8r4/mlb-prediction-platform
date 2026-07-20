import subprocess

steps = [
    "python build_game_dataset.py",
    "python team_form.py",
    "python pitcher_stats.py",
    "python team_stats.py",
    "python train_win_model.py",
]

for step in steps:
    print(f"\nRunning: {step}")
    subprocess.run(step, shell=True, check=True)

print("\nDaily model update complete.")

# To run this script, use the command: python daily_update.py