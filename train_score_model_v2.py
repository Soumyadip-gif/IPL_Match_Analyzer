# ============================================================
# IPL MATCH ANALYZER
# FIRST INNINGS SCORE PREDICTION MODEL - V2
# ============================================================

from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
# ============================================================

MATCHES_PATH = "data/clean/matches.csv"
DELIVERIES_PATH = "data/clean/deliveries.csv"

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "ipl_score_model_v2.pkl"
FEATURE_PATH = MODEL_DIR / "score_feature_columns_v2.pkl"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("IPL FIRST INNINGS SCORE MODEL V2")
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
# 4. CALCULATE FIRST INNINGS INFORMATION
# ============================================================

print("\nPreparing innings information...")

innings_data = {}

for match_id, group in deliveries.groupby("match_id"):

    innings_numbers = sorted(
        group["innings"].dropna().unique()
    )

    if not innings_numbers:
        continue

    first_innings = innings_numbers[0]

    first = group[
        group["innings"] == first_innings
    ].copy()

    if first.empty:
        continue

    batting_team = first["batting_team"].iloc[0]

    score = float(
        first["runs_total"].sum()
    )

    wickets = int(
        first["wicket"].sum()
    )

    innings_data[match_id] = {
        "batting_team": batting_team,
        "score": score,
        "wickets": wickets
    }


# ============================================================
# 5. TEAM MATCH STATISTICS
# ============================================================

print("Preparing team performance history...")

match_team_stats = {}

for match_id, group in deliveries.groupby("match_id"):

    innings_stats = []

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

        innings_stats.append({
            "team": batting_team,
            "score": score,
            "wickets_lost": wickets_lost
        })

    if innings_stats:
        match_team_stats[match_id] = innings_stats


# ============================================================
# 6. HISTORY STRUCTURES
# ============================================================

team_scores = defaultdict(
    lambda: deque(maxlen=10)
)

team_wickets_taken = defaultdict(
    lambda: deque(maxlen=10)
)

team_matches = defaultdict(int)

venue_scores = defaultdict(
    lambda: deque(maxlen=20)
)

h2h_scores = defaultdict(
    lambda: deque(maxlen=10)
)


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def mean_value(values):

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def recent_mean(history, n):

    values = list(history)[-n:]

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def recent_sum(history, n):

    values = list(history)[-n:]

    if not values:
        return 0.0

    return float(
        np.sum(values)
    )


# ============================================================
# 8. BUILD PRE-MATCH DATA
# ============================================================

print("\nBuilding leakage-safe score features...")

rows = []

for _, match in matches.iterrows():

    match_id = match["match_id"]

    if match_id not in innings_data:
        continue

    innings_info = innings_data[
        match_id
    ]

    batting_team = innings_info[
        "batting_team"
    ]

    target_score = innings_info[
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
    # Historical data
    # --------------------------------------------------------

    batting_history = team_scores[
        batting_team
    ]

    bowling_history = team_scores[
        bowling_team
    ]

    bowling_wicket_history = team_wickets_taken[
        bowling_team
    ]

    venue_history = venue_scores[
        venue
    ]

    h2h_key = (
        batting_team,
        bowling_team
    )

    h2h_history = h2h_scores[
        h2h_key
    ]

    # --------------------------------------------------------
    # Toss
    # --------------------------------------------------------

    toss_winner = match.get(
        "toss_winner",
        ""
    )

    toss_decision = match.get(
        "toss_decision",
        ""
    )

    toss_batting = int(
        toss_winner == batting_team
    )

    chose_bat = int(
        toss_decision == "bat"
    )

    # --------------------------------------------------------
    # PRE-MATCH FEATURES
    # --------------------------------------------------------

    row = {

        # Identity
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,

        # Team experience
        "batting_team_matches":
            float(team_matches[batting_team]),

        "bowling_team_matches":
            float(team_matches[bowling_team]),

        # Batting form
        "batting_recent_score_5":
            recent_mean(
                batting_history,
                5
            ),

        "batting_recent_score_10":
            recent_mean(
                batting_history,
                10
            ),

        "batting_recent_runs_5":
            recent_sum(
                batting_history,
                5
            ),

        "batting_average_score":
            mean_value(
                batting_history
            ),

        # Bowling strength
        "bowling_recent_score_5":
            recent_mean(
                bowling_history,
                5
            ),

        "bowling_recent_wickets_5":
            recent_mean(
                bowling_wicket_history,
                5
            ),

        "bowling_recent_wickets_10":
            recent_mean(
                bowling_wicket_history,
                10
            ),

        "bowling_average_wickets":
            mean_value(
                bowling_wicket_history
            ),

        # Venue
        "venue_average_score":
            mean_value(
                venue_history
            ),

        "venue_recent_score":
            recent_mean(
                venue_history,
                10
            ),

        "venue_matches":
            float(len(venue_history)),

        # H2H
        "h2h_average_score":
            mean_value(
                h2h_history
            ),

        "h2h_recent_score":
            recent_mean(
                h2h_history,
                5
            ),

        "h2h_matches":
            float(len(h2h_history)),

        # Toss
        "toss_winner_batting":
            toss_batting,

        "toss_bat_decision":
            chose_bat,

        # Target
        "target_score":
            target_score
    }

    rows.append(row)

    # ========================================================
    # UPDATE HISTORY AFTER THE CURRENT MATCH
    # ========================================================

    team_scores[
        batting_team
    ].append(target_score)

    team_wickets_taken[
        bowling_team
    ].append(
        innings_info["wickets"]
    )

    venue_scores[
        venue
    ].append(target_score)

    h2h_scores[
        h2h_key
    ].append(target_score)

    # --------------------------------------------------------
    # Process both innings for team history
    # --------------------------------------------------------

    if match_id in match_team_stats:

        for innings_info_item in match_team_stats[
            match_id
        ]:

            team = innings_info_item[
                "team"
            ]

            score = innings_info_item[
                "score"
            ]

            wickets_lost = innings_info_item[
                "wickets_lost"
            ]

            # Current first innings already added
            if team == batting_team:
                continue

            team_scores[
                team
            ].append(score)

            team_wickets_taken[
                team
            ].append(wickets_lost)

    # Match experience
    team_matches[team1] += 1
    team_matches[team2] += 1


# ============================================================
# 9. DATAFRAME
# ============================================================

score_df = pd.DataFrame(rows)

score_df = score_df.replace(
    [np.inf, -np.inf],
    np.nan
)

score_df = score_df.fillna(0)

score_df = score_df[
    (score_df["target_score"] > 0)
    &
    (score_df["target_score"] < 350)
].reset_index(drop=True)


print("\nScore dataset shape:")
print(score_df.shape)

print("\nTarget statistics:")
print(
    score_df[
        "target_score"
    ].describe()
)


# ============================================================
# 10. CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(
    len(score_df) * 0.80
)

train_df = score_df.iloc[
    :split_index
].copy()

test_df = score_df.iloc[
    split_index:
].copy()


print("\n" + "=" * 60)
print("CHRONOLOGICAL SPLIT")
print("=" * 60)

print(
    "Training samples:",
    len(train_df)
)

print(
    "Testing samples :",
    len(test_df)
)


# ============================================================
# 11. FEATURES
# ============================================================

categorical_features = [
    "batting_team",
    "bowling_team",
    "venue"
]

numeric_features = [
    column
    for column in score_df.columns
    if column not in (
        categorical_features
        + ["target_score"]
    )
]


X_train = train_df[
    categorical_features + numeric_features
]

X_test = test_df[
    categorical_features + numeric_features
]

y_train = train_df[
    "target_score"
]

y_test = test_df[
    "target_score"
]


# ============================================================
# 12. ONE-HOT ENCODER
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numeric",

            "passthrough",

            numeric_features
        )
    ]
)


# ============================================================
# 13. RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf_pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            RandomForestRegressor(
                n_estimators=400,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

rf_pipeline.fit(
    X_train,
    y_train
)

rf_pred = rf_pipeline.predict(
    X_test
)


rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_pred
    )
)

rf_r2 = r2_score(
    y_test,
    rf_pred
)


# ============================================================
# 14. GRADIENT BOOSTING
# ============================================================

print("Training Gradient Boosting...")

gb_pipeline = Pipeline(
    steps=[

        (
            "preprocessor",

            ColumnTransformer(
                transformers=[

                    (
                        "categorical",

                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),

                        categorical_features
                    ),

                    (
                        "numeric",

                        "passthrough",

                        numeric_features
                    )
                ]
            )
        ),

        (
            "model",

            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42
            )
        )
    ]
)

gb_pipeline.fit(
    X_train,
    y_train
)

gb_pred = gb_pipeline.predict(
    X_test
)


gb_mae = mean_absolute_error(
    y_test,
    gb_pred
)

gb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gb_pred
    )
)

gb_r2 = r2_score(
    y_test,
    gb_pred
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

    selected_model = rf_pipeline
    selected_name = "Random Forest"
    selected_mae = rf_mae
    selected_rmse = rf_rmse
    selected_r2 = rf_r2

else:

    selected_model = gb_pipeline
    selected_name = "Gradient Boosting"
    selected_mae = gb_mae
    selected_rmse = gb_rmse
    selected_r2 = gb_r2


print("\n" + "=" * 60)
print("SELECTED SCORE MODEL V2")
print("=" * 60)

print(
    "Model:",
    selected_name
)

print(
    f"Future-like Test MAE: "
    f"{selected_mae:.2f} runs"
)

print(
    f"Future-like Test RMSE: "
    f"{selected_rmse:.2f} runs"
)

print(
    f"Future-like Test R2: "
    f"{selected_r2:.4f}"
)


# ============================================================
# 17. RETRAIN ON ALL HISTORICAL DATA
# ============================================================

print(
    "\nRetraining selected model "
    "on all historical data..."
)

X_all = score_df[
    categorical_features + numeric_features
]

y_all = score_df[
    "target_score"
]

selected_model.fit(
    X_all,
    y_all
)


# ============================================================
# 18. SAVE MODEL
# ============================================================

joblib.dump(
    selected_model,
    MODEL_PATH
)

joblib.dump(
    {
        "categorical_features":
            categorical_features,

        "numeric_features":
            numeric_features,

        "all_features":
            categorical_features
            + numeric_features
    },
    FEATURE_PATH
)


# ============================================================
# 19. SAVE METADATA
# ============================================================

metadata = {

    "model_name":
        selected_name,

    "mae":
        float(selected_mae),

    "rmse":
        float(selected_rmse),

    "r2":
        float(selected_r2),

    "training_samples":
        int(len(X_all)),

    "test_samples":
        int(len(X_test))
}

joblib.dump(
    metadata,
    MODEL_DIR / "score_model_metadata_v2.pkl"
)


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("SCORE MODEL V2 TRAINING COMPLETE")
print("=" * 60)

print(
    "\nModel saved:"
)

print(
    MODEL_PATH
)

print(
    "\nFeature configuration saved:"
)

print(
    FEATURE_PATH
)

print(
    "\nMetadata saved:"
)

print(
    MODEL_DIR /
    "score_model_metadata_v2.pkl"
)

print(
    "\nThe improved score model is ready "
    "for evaluation and Flask integration."
)