# ============================================================
# IPL MATCH ANALYZER
# FIRST INNINGS SCORE PREDICTION MODEL
# ============================================================

from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
# ============================================================

MATCHES_PATH = "data/clean/matches.csv"
DELIVERIES_PATH = "data/clean/deliveries.csv"

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "ipl_score_model.pkl"
FEATURE_PATH = MODEL_DIR / "score_feature_columns.pkl"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("IPL FIRST INNINGS SCORE MODEL")
print("=" * 60)

matches = pd.read_csv(MATCHES_PATH)
deliveries = pd.read_csv(DELIVERIES_PATH)

matches["date"] = pd.to_datetime(
    matches["date"],
    errors="coerce"
)

matches = matches.sort_values(
    ["date", "match_id"]
).reset_index(drop=True)

print("\nMatches:", matches.shape)
print("Deliveries:", deliveries.shape)


# ============================================================
# 3. PREPARE DELIVERY DATA
# ============================================================

deliveries["runs_total"] = pd.to_numeric(
    deliveries["runs_total"],
    errors="coerce"
).fillna(0)

deliveries["wicket"] = pd.to_numeric(
    deliveries["wicket"],
    errors="coerce"
).fillna(0)

deliveries["innings"] = pd.to_numeric(
    deliveries["innings"],
    errors="coerce"
)


# ============================================================
# 4. CALCULATE MATCH-LEVEL INNINGS DATA
# ============================================================

print("\nPreparing innings information...")

innings_data = {}

for match_id, group in deliveries.groupby("match_id"):

    innings_list = sorted(
        group["innings"].dropna().unique()
    )

    if len(innings_list) == 0:
        continue

    first_innings = innings_list[0]

    first = group[
        group["innings"] == first_innings
    ].copy()

    if first.empty:
        continue

    batting_team = first["batting_team"].iloc[0]

    first_score = float(
        first["runs_total"].sum()
    )

    first_wickets = int(
        first["wicket"].sum()
    )

    innings_data[match_id] = {
        "batting_team": batting_team,
        "score": first_score,
        "wickets": first_wickets
    }


# ============================================================
# 5. MATCH-LEVEL TEAM PERFORMANCE
# ============================================================

print("Preparing team performance history...")

match_team_stats = {}

for match_id, group in deliveries.groupby("match_id"):

    innings_groups = []

    for innings_number, innings_group in group.groupby(
        "innings"
    ):

        if pd.isna(innings_number):
            continue

        batting_team = innings_group[
            "batting_team"
        ].iloc[0]

        score = float(
            innings_group["runs_total"].sum()
        )

        wickets_lost = int(
            innings_group["wicket"].sum()
        )

        innings_groups.append({
            "team": batting_team,
            "score": score,
            "wickets_lost": wickets_lost
        })

    if len(innings_groups) < 1:
        continue

    match_team_stats[match_id] = innings_groups


# ============================================================
# 6. HISTORY STRUCTURES
# ============================================================

# Team batting scores
team_scores = defaultdict(
    lambda: deque(maxlen=10)
)

# Team wickets taken
team_wickets = defaultdict(
    lambda: deque(maxlen=10)
)

# Team matches played
team_matches = defaultdict(int)

# Venue first-innings scores
venue_scores = defaultdict(
    lambda: deque(maxlen=20)
)

# Head-to-head batting scores
h2h_scores = defaultdict(
    lambda: deque(maxlen=10)
)


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def average(values):
    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def recent_average(history, n=5):
    values = list(history)[-n:]

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def recent_sum(history, n=5):
    values = list(history)[-n:]

    if not values:
        return 0.0

    return float(
        np.sum(values)
    )


def wickets_average(history):
    if not history:
        return 0.0

    return float(
        np.mean(history)
    )


# ============================================================
# 8. BUILD PRE-MATCH FEATURES
# ============================================================

print("\nBuilding leakage-safe score features...")

feature_rows = []

for _, match in matches.iterrows():

    match_id = match["match_id"]

    if match_id not in innings_data:
        continue

    first_innings_info = innings_data[
        match_id
    ]

    batting_team = first_innings_info[
        "batting_team"
    ]

    first_innings_score = first_innings_info[
        "score"
    ]

    team1 = match["team1"]
    team2 = match["team2"]

    venue = match["venue"]

    # --------------------------------------------------------
    # Determine bowling team
    # --------------------------------------------------------

    if batting_team == team1:
        bowling_team = team2
    else:
        bowling_team = team1

    # --------------------------------------------------------
    # Batting team historical statistics
    # --------------------------------------------------------

    batting_history = team_scores[
        batting_team
    ]

    batting_wicket_history = team_wickets[
        batting_team
    ]

    # --------------------------------------------------------
    # Bowling team historical statistics
    # --------------------------------------------------------

    bowling_wicket_history = team_wickets[
        bowling_team
    ]

    bowling_score_history = team_scores[
        bowling_team
    ]

    # --------------------------------------------------------
    # Venue statistics
    # --------------------------------------------------------

    venue_history = venue_scores[
        venue
    ]

    # --------------------------------------------------------
    # Head-to-head
    # --------------------------------------------------------

    h2h_key = (
        batting_team,
        bowling_team
    )

    h2h_history = h2h_scores[
        h2h_key
    ]

    # --------------------------------------------------------
    # Toss information
    # --------------------------------------------------------

    toss_winner = match.get(
        "toss_winner",
        None
    )

    toss_decision = match.get(
        "toss_decision",
        None
    )

    toss_winner_batting = int(
        toss_winner == batting_team
    )

    toss_bat_decision = int(
        toss_decision == "bat"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    row = {

        # Date / season
        "season": float(
            match["season"]
        ),

        # Batting team experience
        "batting_team_matches": float(
            team_matches[batting_team]
        ),

        # Bowling team experience
        "bowling_team_matches": float(
            team_matches[bowling_team]
        ),

        # Batting team's recent scoring
        "batting_recent_score_5": recent_average(
            batting_history,
            5
        ),

        "batting_recent_score_10": recent_average(
            batting_history,
            10
        ),

        "batting_recent_runs_5": recent_sum(
            batting_history,
            5
        ),

        # Batting team's historical average
        "batting_average_score": average(
            batting_history
        ),

        # Batting team's recent wickets taken
        "batting_team_wickets_5": recent_average(
            batting_wicket_history,
            5
        ),

        # Bowling team recent scoring
        "bowling_recent_score_5": recent_average(
            bowling_score_history,
            5
        ),

        # Bowling team's wicket-taking ability
        "bowling_recent_wickets_5": recent_average(
            bowling_wicket_history,
            5
        ),

        "bowling_recent_wickets_10": recent_average(
            bowling_wicket_history,
            10
        ),

        "bowling_average_wickets": average(
            bowling_wicket_history
        ),

        # Venue
        "venue_average_score": average(
            venue_history
        ),

        "venue_recent_score": recent_average(
            venue_history,
            10
        ),

        "venue_matches": float(
            len(venue_history)
        ),

        # Head-to-head
        "h2h_average_score": average(
            h2h_history
        ),

        "h2h_recent_score": recent_average(
            h2h_history,
            5
        ),

        "h2h_matches": float(
            len(h2h_history)
        ),

        # Toss
        "toss_winner_batting": toss_winner_batting,

        "toss_bat_decision": toss_bat_decision,

        # Target
        "target_score": first_innings_score
    }

    feature_rows.append(row)

    # ========================================================
    # UPDATE HISTORY AFTER CREATING FEATURES
    # ========================================================
    # IMPORTANT:
    # The current match score is added only AFTER the
    # pre-match features have been created.
    # This prevents target leakage.
    # ========================================================

    # Add batting score
    team_scores[
        batting_team
    ].append(first_innings_score)

    # Add wickets taken by bowling team
    team_wickets[
        bowling_team
    ].append(
        first_innings_info["wickets"]
    )

    # Venue history
    venue_scores[
        venue
    ].append(first_innings_score)

    # H2H batting score
    h2h_scores[
        h2h_key
    ].append(first_innings_score)

    # --------------------------------------------------------
    # Process second innings for team history
    # --------------------------------------------------------

    if match_id in match_team_stats:

        innings_stats = match_team_stats[
            match_id
        ]

        for innings_info in innings_stats:

            team = innings_info["team"]

            score = innings_info["score"]

            wickets_lost = innings_info[
                "wickets_lost"
            ]

            # Avoid adding first innings twice
            if team == batting_team:
                continue

            team_scores[
                team
            ].append(score)

            # Wickets taken by this team
            # = opponent wickets lost
            team_wickets[
                team
            ].append(wickets_lost)

    # Every team in the match gets a match count
    team_matches[team1] += 1
    team_matches[team2] += 1


# ============================================================
# 9. CREATE DATAFRAME
# ============================================================

score_df = pd.DataFrame(
    feature_rows
)

score_df = score_df.replace(
    [np.inf, -np.inf],
    np.nan
)

score_df = score_df.fillna(0)


print("\nScore dataset shape:")
print(
    score_df.shape
)

print(
    "\nTarget score statistics:"
)

print(
    score_df["target_score"].describe()
)


# ============================================================
# 10. REMOVE EXTREME / INVALID TARGETS
# ============================================================

score_df = score_df[
    (score_df["target_score"] > 0)
    &
    (score_df["target_score"] < 350)
].reset_index(drop=True)


print(
    "\nFinal dataset shape:",
    score_df.shape
)


# ============================================================
# 11. FEATURES / TARGET
# ============================================================

FEATURE_COLUMNS = [
    column
    for column in score_df.columns
    if column != "target_score"
]

X = score_df[
    FEATURE_COLUMNS
]

y = score_df[
    "target_score"
]


# ============================================================
# 12. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(score_df) * 0.80
)

X_train = X.iloc[
    :split_index
].copy()

X_test = X.iloc[
    split_index:
].copy()

y_train = y.iloc[
    :split_index
].copy()

y_test = y.iloc[
    split_index:
].copy()


print("\n" + "=" * 60)
print("CHRONOLOGICAL SPLIT")
print("=" * 60)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# ============================================================
# 13. RANDOM FOREST REGRESSOR
# ============================================================

print("\nTraining Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)


rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


# ============================================================
# 14. GRADIENT BOOSTING REGRESSOR
# ============================================================

print("Training Gradient Boosting...")

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

gb_model.fit(
    X_train,
    y_train
)

gb_predictions = gb_model.predict(
    X_test
)


gb_mae = mean_absolute_error(
    y_test,
    gb_predictions
)

gb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gb_predictions
    )
)

gb_r2 = r2_score(
    y_test,
    gb_predictions
)


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

results = pd.DataFrame([
    {
        "Model": "Random Forest",
        "MAE": rf_mae,
        "RMSE": rf_rmse,
        "R2": rf_r2
    },
    {
        "Model": "Gradient Boosting",
        "MAE": gb_mae,
        "RMSE": gb_rmse,
        "R2": gb_r2
    }
])


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 16. SELECT BEST MODEL
# ============================================================

if rf_mae <= gb_mae:

    selected_model = rf_model
    selected_name = "Random Forest"
    selected_mae = rf_mae
    selected_rmse = rf_rmse
    selected_r2 = rf_r2

else:

    selected_model = gb_model
    selected_name = "Gradient Boosting"
    selected_mae = gb_mae
    selected_rmse = gb_rmse
    selected_r2 = gb_r2


print("\n" + "=" * 60)
print("SELECTED SCORE MODEL")
print("=" * 60)

print(
    "Model:",
    selected_name
)

print(
    f"Future-like Test MAE: {selected_mae:.2f} runs"
)

print(
    f"Future-like Test RMSE: {selected_rmse:.2f} runs"
)

print(
    f"Future-like Test R2: {selected_r2:.4f}"
)


# ============================================================
# 17. RETRAIN ON ALL HISTORICAL DATA
# ============================================================

print("\nRetraining selected model on all historical data...")

selected_model.fit(
    X,
    y
)


# ============================================================
# 18. SAVE MODEL
# ============================================================

joblib.dump(
    selected_model,
    MODEL_PATH
)

joblib.dump(
    FEATURE_COLUMNS,
    FEATURE_PATH
)


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

if hasattr(
    selected_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame({

        "Feature": FEATURE_COLUMNS,

        "Importance":
            selected_model.feature_importances_

    }).sort_values(
        "Importance",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("TOP SCORE PREDICTION FEATURES")
    print("=" * 60)

    print(
        importance_df.head(10).to_string(
            index=False
        )
    )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("SCORE MODEL TRAINING COMPLETE")
print("=" * 60)

print(
    "\nModel saved:"
)

print(
    MODEL_PATH
)

print(
    "\nFeature columns saved:"
)

print(
    FEATURE_PATH
)

print(
    "\nThe model is ready for future IPL score prediction."
)