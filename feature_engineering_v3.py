import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# IPL MATCH ANALYZER - ADVANCED LEAKAGE-SAFE FEATURES
# ============================================================

print("=" * 60)
print("IPL MATCH ANALYZER - ADVANCED FEATURE ENGINEERING")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

matches = pd.read_csv("data/clean/matches.csv")
deliveries = pd.read_csv("data/clean/deliveries.csv")

matches["date"] = pd.to_datetime(matches["date"])

# Only completed matches with a known winner
matches = matches[matches["winner"].notna()].copy()

matches = matches.sort_values(
    ["date", "match_id"]
).reset_index(drop=True)

print("\nValid matches:", len(matches))

# ------------------------------------------------------------
# 2. Calculate match-level batting/bowling statistics
# ------------------------------------------------------------

team_match_stats = {}

for match_id, group in deliveries.groupby("match_id"):

    # Batting runs for each team in this match
    batting_runs = (
        group.groupby("batting_team")["runs_total"]
        .sum()
        .to_dict()
    )

    # Balls faced by each batting team
    balls_faced = (
        group.groupby("batting_team")
        .size()
        .to_dict()
    )

    # Wickets taken by each bowling team
    bowling_wickets = {}

    for _, row in group[group["wicket"] == 1].iterrows():

        bowler = row["bowler"]

        batting_team = row["batting_team"]

        # Find the bowling team from the match teams later.
        # Store bowler wickets first.
        bowling_wickets[bowler] = (
            bowling_wickets.get(bowler, 0) + 1
        )

    team_match_stats[match_id] = {
        "batting_runs": batting_runs,
        "balls_faced": balls_faced,
        "bowler_wickets": bowling_wickets
    }

# ------------------------------------------------------------
# 3. Team list
# ------------------------------------------------------------

teams = sorted(
    set(matches["team1"]) |
    set(matches["team2"])
)

# ------------------------------------------------------------
# 4. Historical tracking
# ------------------------------------------------------------

team_matches = {
    team: 0 for team in teams
}

team_wins = {
    team: 0 for team in teams
}

team_runs = {
    team: 0 for team in teams
}

team_balls = {
    team: 0 for team in teams
}

team_wickets = {
    team: 0 for team in teams
}

# Recent match results
recent_results = {
    team: [] for team in teams
}

# Recent runs
recent_runs = {
    team: [] for team in teams
}

# Recent wickets
recent_wickets = {
    team: [] for team in teams
}

# Venue history
venue_matches = {}
venue_wins = {}

# Head-to-head
h2h_matches = {}
h2h_wins = {}

# Elo ratings
elo = {
    team: 1500 for team in teams
}

# ------------------------------------------------------------
# 5. Helper function
# ------------------------------------------------------------

def recent_average(values, n, default=0.5):

    values = values[-n:]

    if len(values) == 0:
        return default

    return sum(values) / len(values)


# ------------------------------------------------------------
# 6. Build features chronologically
# ------------------------------------------------------------

features = []

for _, match in matches.iterrows():

    match_id = match["match_id"]

    team1 = match["team1"]
    team2 = match["team2"]

    venue = match["venue"]

    # ========================================================
    # Overall historical win rate
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
    # Recent form
    # ========================================================

    team1_form_5 = recent_average(
        recent_results[team1],
        5
    )

    team2_form_5 = recent_average(
        recent_results[team2],
        5
    )

    team1_form_10 = recent_average(
        recent_results[team1],
        10
    )

    team2_form_10 = recent_average(
        recent_results[team2],
        10
    )

    # ========================================================
    # Recent batting performance
    # ========================================================

    team1_recent_runs = recent_average(
        recent_runs[team1],
        5,
        default=0
    )

    team2_recent_runs = recent_average(
        recent_runs[team2],
        5,
        default=0
    )

    # ========================================================
    # Recent bowling performance
    # ========================================================

    team1_recent_wickets = recent_average(
        recent_wickets[team1],
        5,
        default=0
    )

    team2_recent_wickets = recent_average(
        recent_wickets[team2],
        5,
        default=0
    )

    # ========================================================
    # Historical batting strength
    # ========================================================

    team1_run_rate = (
        team_runs[team1] / team_balls[team1]
        if team_balls[team1] > 0
        else 0
    )

    team2_run_rate = (
        team_runs[team2] / team_balls[team2]
        if team_balls[team2] > 0
        else 0
    )

    # ========================================================
    # Historical wickets per match
    # ========================================================

    team1_wickets_per_match = (
        team_wickets[team1] / team_matches[team1]
        if team_matches[team1] > 0
        else 0
    )

    team2_wickets_per_match = (
        team_wickets[team2] / team_matches[team2]
        if team_matches[team2] > 0
        else 0
    )

    # ========================================================
    # Venue performance
    # ========================================================

    venue_key_1 = (team1, venue)
    venue_key_2 = (team2, venue)

    v1_matches = venue_matches.get(
        venue_key_1,
        0
    )

    v2_matches = venue_matches.get(
        venue_key_2,
        0
    )

    v1_wins = venue_wins.get(
        venue_key_1,
        0
    )

    v2_wins = venue_wins.get(
        venue_key_2,
        0
    )

    team1_venue_rate = (
        v1_wins / v1_matches
        if v1_matches > 0
        else 0.5
    )

    team2_venue_rate = (
        v2_wins / v2_matches
        if v2_matches > 0
        else 0.5
    )

    # ========================================================
    # Head-to-head
    # ========================================================

    h2h_key = tuple(
        sorted([team1, team2])
    )

    total_h2h = h2h_matches.get(
        h2h_key,
        0
    )

    team1_h2h_wins = h2h_wins.get(
        (h2h_key, team1),
        0
    )

    h2h_team1_rate = (
        team1_h2h_wins / total_h2h
        if total_h2h > 0
        else 0.5
    )

    # ========================================================
    # Elo rating
    # ========================================================

    team1_elo = elo[team1]
    team2_elo = elo[team2]

    elo_difference = team1_elo - team2_elo

    # ========================================================
    # Toss
    # ========================================================

    toss_team1 = int(
        match["toss_winner"] == team1
    )

    toss_bat = int(
        match["toss_decision"] == "bat"
    )

    # ========================================================
    # Team strength differences
    # ========================================================

    win_rate_difference = (
        team1_win_rate -
        team2_win_rate
    )

    form_difference = (
        team1_form_5 -
        team2_form_5
    )

    form_10_difference = (
        team1_form_10 -
        team2_form_10
    )

    batting_difference = (
        team1_run_rate -
        team2_run_rate
    )

    bowling_difference = (
        team1_wickets_per_match -
        team2_wickets_per_match
    )

    venue_difference = (
        team1_venue_rate -
        team2_venue_rate
    )

    recent_runs_difference = (
        team1_recent_runs -
        team2_recent_runs
    )

    recent_wickets_difference = (
        team1_recent_wickets -
        team2_recent_wickets
    )

    # ========================================================
    # Target
    # ========================================================

    target = int(
        match["winner"] == team1
    )

    # ========================================================
    # Store features
    # ========================================================

    features.append({

        "team1": team1,
        "team2": team2,
        "venue": venue,
        "season": match["season"],

        "toss_team1": toss_team1,
        "toss_bat": toss_bat,

        "team1_win_rate": team1_win_rate,
        "team2_win_rate": team2_win_rate,

        "team1_form_5": team1_form_5,
        "team2_form_5": team2_form_5,

        "team1_form_10": team1_form_10,
        "team2_form_10": team2_form_10,

        "team1_recent_runs": team1_recent_runs,
        "team2_recent_runs": team2_recent_runs,

        "team1_recent_wickets": team1_recent_wickets,
        "team2_recent_wickets": team2_recent_wickets,

        "team1_run_rate": team1_run_rate,
        "team2_run_rate": team2_run_rate,

        "team1_wickets_per_match": team1_wickets_per_match,
        "team2_wickets_per_match": team2_wickets_per_match,

        "team1_venue_rate": team1_venue_rate,
        "team2_venue_rate": team2_venue_rate,

        "h2h_team1_rate": h2h_team1_rate,

        "team1_elo": team1_elo,
        "team2_elo": team2_elo,
        "elo_difference": elo_difference,

        "win_rate_difference": win_rate_difference,
        "form_difference": form_difference,
        "form_10_difference": form_10_difference,

        "batting_difference": batting_difference,
        "bowling_difference": bowling_difference,

        "venue_difference": venue_difference,

        "recent_runs_difference": recent_runs_difference,
        "recent_wickets_difference": recent_wickets_difference,

        "target": target
    })

    # ========================================================
    # Get current match statistics
    # ========================================================

    stats = team_match_stats.get(
        match_id,
        {}
    )

    batting_runs = stats.get(
        "batting_runs",
        {}
    )

    balls_faced = stats.get(
        "balls_faced",
        {}
    )

    bowler_wickets = stats.get(
        "bowler_wickets",
        {}
    )

    # ========================================================
    # Match batting runs
    # ========================================================

    team1_match_runs = batting_runs.get(
        team1,
        0
    )

    team2_match_runs = batting_runs.get(
        team2,
        0
    )

    # ========================================================
    # Match wickets
    #
    # Count wickets credited to bowlers belonging to the
    # opposing batting team.
    # ========================================================

    team1_match_wickets = 0
    team2_match_wickets = 0

    # Build bowler -> wickets
    # and determine bowling team from match batting teams.

    for bowler, wickets in bowler_wickets.items():

        # If bowler is not in batting team names directly,
        # determine their team using the match deliveries.
        bowler_rows = deliveries[
            (deliveries["match_id"] == match_id) &
            (deliveries["bowler"] == bowler)
        ]

        if len(bowler_rows) == 0:
            continue

        bowling_against = bowler_rows.iloc[0]["batting_team"]

        if bowling_against == team1:
            team2_match_wickets += wickets

        elif bowling_against == team2:
            team1_match_wickets += wickets

    # ========================================================
    # Update team historical statistics
    # AFTER creating the features
    # ========================================================

    team_matches[team1] += 1
    team_matches[team2] += 1

    team_runs[team1] += team1_match_runs
    team_runs[team2] += team2_match_runs

    team_balls[team1] += balls_faced.get(
        team1,
        0
    )

    team_balls[team2] += balls_faced.get(
        team2,
        0
    )

    team_wickets[team1] += team1_match_wickets
    team_wickets[team2] += team2_match_wickets

    # Recent history
    recent_results[team1].append(target)
    recent_results[team2].append(1 - target)

    recent_runs[team1].append(
        team1_match_runs
    )

    recent_runs[team2].append(
        team2_match_runs
    )

    recent_wickets[team1].append(
        team1_match_wickets
    )

    recent_wickets[team2].append(
        team2_match_wickets
    )

    # Overall wins
    if target == 1:

        team_wins[team1] += 1

    else:

        team_wins[team2] += 1

    # Venue history
    venue_matches[venue_key_1] = (
        v1_matches + 1
    )

    venue_matches[venue_key_2] = (
        v2_matches + 1
    )

    if target == 1:

        venue_wins[venue_key_1] = (
            v1_wins + 1
        )

        venue_wins[venue_key_2] = (
            v2_wins
        )

    else:

        venue_wins[venue_key_1] = (
            v1_wins
        )

        venue_wins[venue_key_2] = (
            v2_wins + 1
        )

    # H2H history
    h2h_matches[h2h_key] = (
        total_h2h + 1
    )

    if target == 1:

        h2h_wins[
            (h2h_key, team1)
        ] = team1_h2h_wins + 1

    else:

        h2h_wins[
            (h2h_key, team2)
        ] = h2h_wins.get(
            (h2h_key, team2),
            0
        ) + 1

    # ========================================================
    # Update Elo AFTER match
    # ========================================================

    expected_team1 = 1 / (
        1 + 10 ** (
            (team2_elo - team1_elo) / 400
        )
    )

    actual_team1 = target

    K = 20

    elo[team1] = (
        team1_elo +
        K * (
            actual_team1 -
            expected_team1
        )
    )

    elo[team2] = (
        team2_elo +
        K * (
            (1 - actual_team1) -
            (1 - expected_team1)
        )
    )

# ------------------------------------------------------------
# 7. Create DataFrame
# ------------------------------------------------------------

features_df = pd.DataFrame(
    features
)

# ------------------------------------------------------------
# 8. One-hot encoding
# ------------------------------------------------------------

features_df = pd.get_dummies(
    features_df,
    columns=[
        "team1",
        "team2",
        "venue"
    ],
    dtype=int
)

# ------------------------------------------------------------
# 9. Handle invalid values
# ------------------------------------------------------------

features_df = features_df.replace(
    [np.inf, -np.inf],
    np.nan
)

features_df = features_df.fillna(0)

# ------------------------------------------------------------
# 10. Save
# ------------------------------------------------------------

Path("data/features").mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    "data/features/"
    "match_features_v3.csv"
)

features_df.to_csv(
    output_path,
    index=False
)

# ------------------------------------------------------------
# 11. Results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ADVANCED FEATURE ENGINEERING RESULTS")
print("=" * 60)

print(
    "Feature dataset shape:",
    features_df.shape
)

print(
    "\nTarget distribution:"
)

print(
    features_df["target"].value_counts()
)

print(
    "\nMissing values:",
    features_df.isnull().sum().sum()
)

print(
    "\nTotal features:",
    features_df.shape[1] - 1
)

print(
    "\nSaved file:"
)

print(output_path)

print(
    "\nAdvanced numerical features:"
)

advanced_features = [
    "team1_elo",
    "team2_elo",
    "elo_difference",
    "team1_form_5",
    "team2_form_5",
    "team1_form_10",
    "team2_form_10",
    "team1_recent_runs",
    "team2_recent_runs",
    "team1_recent_wickets",
    "team2_recent_wickets",
    "team1_run_rate",
    "team2_run_rate",
    "team1_wickets_per_match",
    "team2_wickets_per_match",
    "team1_venue_rate",
    "team2_venue_rate",
    "h2h_team1_rate",
    "win_rate_difference",
    "form_difference",
    "form_10_difference",
    "batting_difference",
    "bowling_difference",
    "venue_difference",
    "recent_runs_difference",
    "recent_wickets_difference"
]

for feature in advanced_features:
    print("-", feature)

print("\n" + "=" * 60)
print("ADVANCED FEATURE ENGINEERING COMPLETED")
print("=" * 60)