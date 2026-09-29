import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# IPL MATCH ANALYZER - FEATURE ENGINEERING
# ============================================================

matches = pd.read_csv("data/clean/matches.csv")
deliveries = pd.read_csv("data/clean/deliveries.csv")

print("=" * 60)
print("IPL MATCH ANALYZER - FEATURE ENGINEERING")
print("=" * 60)

# ------------------------------------------------------------
# 1. Prepare match data
# ------------------------------------------------------------

matches["date"] = pd.to_datetime(matches["date"])

# Only matches with a valid winner can be used for
# win/loss prediction
model_data = matches[matches["winner"].notna()].copy()

print("\nValid matches for ML:", len(model_data))

# ------------------------------------------------------------
# 2. Target variable
# ------------------------------------------------------------

model_data["target"] = (
    model_data["winner"] == model_data["team1"]
).astype(int)

# 1 = Team 1 wins
# 0 = Team 2 wins

# ------------------------------------------------------------
# 3. Team 1 / Team 2 historical win statistics
# ------------------------------------------------------------

team_matches = {}
team_wins = {}

for team in set(model_data["team1"]) | set(model_data["team2"]):
    team_matches[team] = 0
    team_wins[team] = 0

features = []

for _, row in model_data.sort_values("date").iterrows():

    team1 = row["team1"]
    team2 = row["team2"]

    # Historical win rate BEFORE this match
    team1_win_rate = (
        team_wins[team1] / team_matches[team1]
        if team_matches[team1] > 0 else 0.5
    )

    team2_win_rate = (
        team_wins[team2] / team_matches[team2]
        if team_matches[team2] > 0 else 0.5
    )

    # Head-to-head historical record
    h2h_total = 0
    h2h_team1_wins = 0

    previous_matches = model_data[
        (
            ((model_data["team1"] == team1) &
             (model_data["team2"] == team2)) |
            ((model_data["team1"] == team2) &
             (model_data["team2"] == team1))
        )
        &
        (model_data["date"] < row["date"])
    ]

    h2h_total = len(previous_matches)

    if h2h_total > 0:
        h2h_team1_wins = (
            previous_matches["winner"] == team1
        ).sum()

    h2h_win_rate = (
        h2h_team1_wins / h2h_total
        if h2h_total > 0 else 0.5
    )

    # Toss feature
    toss_team1 = int(row["toss_winner"] == team1)

    # Toss decision
    toss_bat = int(row["toss_decision"] == "bat")

    features.append({
        "team1": team1,
        "team2": team2,
        "venue": row["venue"],
        "season": row["season"],
        "toss_team1": toss_team1,
        "toss_bat": toss_bat,
        "team1_win_rate": team1_win_rate,
        "team2_win_rate": team2_win_rate,
        "h2h_win_rate": h2h_win_rate,
        "target": row["target"]
    })

    # Update historical statistics AFTER the match
    team_matches[team1] += 1
    team_matches[team2] += 1

    if row["winner"] == team1:
        team_wins[team1] += 1
    elif row["winner"] == team2:
        team_wins[team2] += 1

features_df = pd.DataFrame(features)

# ------------------------------------------------------------
# 4. One-hot encode categorical features
# ------------------------------------------------------------

features_df = pd.get_dummies(
    features_df,
    columns=["team1", "team2", "venue"],
    dtype=int
)

# ------------------------------------------------------------
# 5. Save ML-ready dataset
# ------------------------------------------------------------

Path("data/features").mkdir(parents=True, exist_ok=True)

output_path = "data/features/match_features.csv"

features_df.to_csv(output_path, index=False)

# ------------------------------------------------------------
# 6. Final information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE ENGINEERING RESULTS")
print("=" * 60)

print("Original matches:", len(matches))
print("ML matches:", len(model_data))

print("\nFeature dataset shape:")
print(features_df.shape)

print("\nTarget distribution:")
print(features_df["target"].value_counts())

print("\nMissing values:")
print(features_df.isnull().sum().sum())

print("\nSaved file:")
print(output_path)

print("\nFirst 5 rows:")
print(features_df.head())

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)