import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# IPL MATCH ANALYZER - IMPROVED FEATURE ENGINEERING
# ============================================================

matches = pd.read_csv("data/clean/matches.csv")
deliveries = pd.read_csv("data/clean/deliveries.csv")

matches["date"] = pd.to_datetime(matches["date"])

print("=" * 60)
print("IPL MATCH ANALYZER - IMPROVED FEATURE ENGINEERING")
print("=" * 60)

# ------------------------------------------------------------
# 1. Valid matches only
# ------------------------------------------------------------

matches = matches[matches["winner"].notna()].copy()
matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

# ------------------------------------------------------------
# 2. Calculate team batting and bowling strength
# ------------------------------------------------------------

team_batting = (
    deliveries.groupby("batting_team")["runs_total"]
    .sum()
    .to_dict()
)

team_balls = (
    deliveries.groupby("batting_team")
    .size()
    .to_dict()
)

team_wickets = (
    deliveries[deliveries["wicket"] == 1]
    .groupby("bowler")
    .size()
    .to_dict()
)

# Average runs per delivery
team_run_rate = {}

for team in team_batting:
    balls = team_balls.get(team, 1)
    team_run_rate[team] = team_batting[team] / balls

# ------------------------------------------------------------
# 3. Historical tracking
# ------------------------------------------------------------

teams = set(matches["team1"]) | set(matches["team2"])

team_matches = {team: 0 for team in teams}
team_wins = {team: 0 for team in teams}

team_recent_results = {team: [] for team in teams}

venue_matches = {}
venue_wins = {}

h2h_matches = {}
h2h_wins = {}

features = []

# ------------------------------------------------------------
# 4. Process matches chronologically
# ------------------------------------------------------------

for _, row in matches.iterrows():

    team1 = row["team1"]
    team2 = row["team2"]
    venue = row["venue"]

    # ========================================================
    # Overall win rate
    # ========================================================

    team1_win_rate = (
        team_wins[team1] / team_matches[team1]
        if team_matches[team1] > 0
        else 0.5
    )

    team2_win_rate = (
        team_wins[team2] / team_matches[team2]
        if team_matches[team2] > 0
        else 0.5
    )

    # ========================================================
    # Recent 5-match form
    # ========================================================

    team1_recent = team_recent_results[team1][-5:]
    team2_recent = team_recent_results[team2][-5:]

    team1_recent_form = (
        sum(team1_recent) / len(team1_recent)
        if team1_recent
        else 0.5
    )

    team2_recent_form = (
        sum(team2_recent) / len(team2_recent)
        if team2_recent
        else 0.5
    )

    # ========================================================
    # Venue performance
    # ========================================================

    venue_key_1 = (team1, venue)
    venue_key_2 = (team2, venue)

    venue_team1_matches = venue_matches.get(venue_key_1, 0)
    venue_team1_wins = venue_wins.get(venue_key_1, 0)

    venue_team2_matches = venue_matches.get(venue_key_2, 0)
    venue_team2_wins = venue_wins.get(venue_key_2, 0)

    team1_venue_rate = (
        venue_team1_wins / venue_team1_matches
        if venue_team1_matches > 0
        else 0.5
    )

    team2_venue_rate = (
        venue_team2_wins / venue_team2_matches
        if venue_team2_matches > 0
        else 0.5
    )

    # ========================================================
    # Head-to-head
    # ========================================================

    h2h_key = tuple(sorted([team1, team2]))

    total_h2h = h2h_matches.get(h2h_key, 0)
    team1_h2h_wins = h2h_wins.get((h2h_key, team1), 0)

    h2h_team1_rate = (
        team1_h2h_wins / total_h2h
        if total_h2h > 0
        else 0.5
    )

    # ========================================================
    # Batting strength
    # ========================================================

    team1_batting = team_run_rate.get(team1, 0)
    team2_batting = team_run_rate.get(team2, 0)

    # ========================================================
    # Bowling strength
    # ========================================================

    team1_bowling = team_wickets.get(team1, 0)
    team2_bowling = team_wickets.get(team2, 0)

    # Normalize bowling strength
    max_wickets = max(team_wickets.values()) if team_wickets else 1

    team1_bowling_strength = team1_bowling / max_wickets
    team2_bowling_strength = team2_bowling / max_wickets

    # ========================================================
    # Toss
    # ========================================================

    toss_team1 = int(row["toss_winner"] == team1)

    toss_bat = int(row["toss_decision"] == "bat")

    # ========================================================
    # Target
    # ========================================================

    target = int(row["winner"] == team1)

    # ========================================================
    # Store features
    # ========================================================

    features.append({

        "team1": team1,
        "team2": team2,
        "venue": venue,
        "season": row["season"],

        "toss_team1": toss_team1,
        "toss_bat": toss_bat,

        "team1_win_rate": team1_win_rate,
        "team2_win_rate": team2_win_rate,

        "team1_recent_form": team1_recent_form,
        "team2_recent_form": team2_recent_form,

        "team1_venue_rate": team1_venue_rate,
        "team2_venue_rate": team2_venue_rate,

        "h2h_team1_rate": h2h_team1_rate,

        "team1_batting_strength": team1_batting,
        "team2_batting_strength": team2_batting,

        "team1_bowling_strength": team1_bowling_strength,
        "team2_bowling_strength": team2_bowling_strength,

        "target": target
    })

    # ========================================================
    # Update historical information AFTER the match
    # ========================================================

    team_matches[team1] += 1
    team_matches[team2] += 1

    if target == 1:
        team_wins[team1] += 1
        team_recent_results[team1].append(1)
        team_recent_results[team2].append(0)

        h2h_wins[(h2h_key, team1)] = (
            h2h_wins.get((h2h_key, team1), 0) + 1
        )

    else:
        team_wins[team2] += 1
        team_recent_results[team1].append(0)
        team_recent_results[team2].append(1)

        h2h_wins[(h2h_key, team2)] = (
            h2h_wins.get((h2h_key, team2), 0) + 1
        )

    h2h_matches[h2h_key] = total_h2h + 1

    # Venue history
    venue_matches[venue_key_1] = venue_team1_matches + 1
    venue_matches[venue_key_2] = venue_team2_matches + 1

    if target == 1:
        venue_wins[venue_key_1] = venue_team1_wins + 1
        venue_wins[venue_key_2] = venue_team2_wins
    else:
        venue_wins[venue_key_1] = venue_team1_wins
        venue_wins[venue_key_2] = venue_team2_wins + 1

# ------------------------------------------------------------
# 5. Create DataFrame
# ------------------------------------------------------------

features_df = pd.DataFrame(features)

# ------------------------------------------------------------
# 6. One-hot encoding
# ------------------------------------------------------------

features_df = pd.get_dummies(
    features_df,
    columns=["team1", "team2", "venue"],
    dtype=int
)

# ------------------------------------------------------------
# 7. Clean numeric values
# ------------------------------------------------------------

features_df = features_df.replace(
    [np.inf, -np.inf],
    np.nan
)

features_df = features_df.fillna(0)

# ------------------------------------------------------------
# 8. Save
# ------------------------------------------------------------

Path("data/features").mkdir(
    parents=True,
    exist_ok=True
)

output_path = "data/features/match_features_v2.csv"

features_df.to_csv(
    output_path,
    index=False
)

# ------------------------------------------------------------
# 9. Display results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("IMPROVED FEATURE ENGINEERING RESULTS")
print("=" * 60)

print("Valid matches:", len(matches))

print("\nFeature dataset shape:")
print(features_df.shape)

print("\nTarget distribution:")
print(features_df["target"].value_counts())

print("\nMissing values:")
print(features_df.isnull().sum().sum())

print("\nSaved file:")
print(output_path)

print("\nNew numerical features:")

new_features = [
    "team1_win_rate",
    "team2_win_rate",
    "team1_recent_form",
    "team2_recent_form",
    "team1_venue_rate",
    "team2_venue_rate",
    "h2h_team1_rate",
    "team1_batting_strength",
    "team2_batting_strength",
    "team1_bowling_strength",
    "team2_bowling_strength"
]

print(new_features)

print("\n" + "=" * 60)
print("IMPROVED FEATURE ENGINEERING COMPLETED")
print("=" * 60)