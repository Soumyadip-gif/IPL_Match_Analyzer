import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MATCHES_PATH = "data/clean/matches.csv"
DELIVERIES_PATH = "data/clean/deliveries.csv"

V1_MODEL_PATH = "model/ipl_score_model.pkl"
V1_FEATURE_PATH = "model/score_feature_columns.pkl"


# ============================================================
# LOAD
# ============================================================

matches = pd.read_csv(MATCHES_PATH)
deliveries = pd.read_csv(DELIVERIES_PATH)

matches["date"] = pd.to_datetime(
    matches["date"],
    errors="coerce"
)

matches = matches.sort_values(
    ["date", "match_id"]
).reset_index(drop=True)

deliveries = deliveries.sort_values(
    ["match_id", "innings", "over", "ball"]
)


# ============================================================
# FIRST INNINGS
# ============================================================

innings = (
    deliveries[
        deliveries["innings"] == 1
    ]
    .groupby("match_id")
    .agg(
        first_innings_score=(
            "runs_total",
            "sum"
        ),
        first_innings_wickets=(
            "wicket",
            "sum"
        ),
        batting_team=(
            "batting_team",
            "first"
        )
    )
    .reset_index()
)


data = matches.merge(
    innings,
    on="match_id",
    how="inner"
)


# ============================================================
# HISTORY
# ============================================================

from collections import defaultdict, deque


team_scores = defaultdict(
    lambda: deque(maxlen=10)
)

team_wickets = defaultdict(
    lambda: deque(maxlen=10)
)

team_matches = defaultdict(int)

venue_scores = defaultdict(
    lambda: deque(maxlen=20)
)

h2h_scores = defaultdict(
    lambda: deque(maxlen=10)
)


features = []
targets = []


# ============================================================
# CREATE EXACT V1 FEATURES
# ============================================================

for _, match in data.iterrows():

    team1 = match["team1"]
    team2 = match["team2"]
    venue = match["venue"]

    batting_team = match["batting_team"]

    bowling_team = (
        team2
        if batting_team == team1
        else team1
    )


    batting_history = list(
        team_scores[batting_team]
    )

    bowling_history = list(
        team_scores[bowling_team]
    )

    batting_wickets = list(
        team_wickets[batting_team]
    )

    bowling_wickets = list(
        team_wickets[bowling_team]
    )

    venue_history = list(
        venue_scores[venue]
    )

    h2h_key = tuple(
        sorted(
            [batting_team, bowling_team]
        )
    )

    h2h_history = list(
        h2h_scores[h2h_key]
    )


    row = {

        "season":
            match["season"],

        "batting_team_matches":
            team_matches[batting_team],

        "bowling_team_matches":
            team_matches[bowling_team],

        "batting_recent_score_5":
            np.mean(
                batting_history[-5:]
            )
            if batting_history else 0,

        "batting_recent_score_10":
            np.mean(
                batting_history[-10:]
            )
            if batting_history else 0,

        "batting_recent_runs_5":
            np.mean(
                batting_history[-5:]
            )
            if batting_history else 0,

        "batting_average_score":
            np.mean(
                batting_history
            )
            if batting_history else 0,

        "batting_team_wickets_5":
            np.mean(
                batting_wickets[-5:]
            )
            if batting_wickets else 0,

        "bowling_recent_score_5":
            np.mean(
                bowling_history[-5:]
            )
            if bowling_history else 0,

        "bowling_recent_wickets_5":
            np.mean(
                bowling_wickets[-5:]
            )
            if bowling_wickets else 0,

        "bowling_recent_wickets_10":
            np.mean(
                bowling_wickets[-10:]
            )
            if bowling_wickets else 0,

        "bowling_average_wickets":
            np.mean(
                bowling_wickets
            )
            if bowling_wickets else 0,

        "venue_average_score":
            np.mean(
                venue_history
            )
            if venue_history else 0,

        "venue_recent_score":
            np.mean(
                venue_history[-5:]
            )
            if venue_history else 0,

        "venue_matches":
            len(venue_history),

        "h2h_average_score":
            np.mean(
                h2h_history
            )
            if h2h_history else 0,

        "h2h_recent_score":
            np.mean(
                h2h_history[-5:]
            )
            if h2h_history else 0,

        "h2h_matches":
            len(h2h_history),

        "toss_winner_batting":
            int(
                match["toss_winner"]
                == batting_team
            ),

        "toss_bat_decision":
            int(
                match["toss_decision"]
                == "bat"
            )
    }


    features.append(row)

    targets.append(
        match["first_innings_score"]
    )


    # Update AFTER features
    score = match[
        "first_innings_score"
    ]

    wickets = int(
        match["first_innings_wickets"]
    )


    team_scores[
        batting_team
    ].append(score)

    team_wickets[
        bowling_team
    ].append(wickets)

    venue_scores[
        venue
    ].append(score)

    h2h_scores[
        h2h_key
    ].append(score)

    team_matches[team1] += 1
    team_matches[team2] += 1


# ============================================================
# DATASET
# ============================================================

X = pd.DataFrame(features)

y = pd.Series(
    targets,
    name="score"
)


# ============================================================
# LOAD V1
# ============================================================

v1_model = joblib.load(
    V1_MODEL_PATH
)

v1_columns = joblib.load(
    V1_FEATURE_PATH
)

X = X.reindex(
    columns=v1_columns,
    fill_value=0
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(
    len(X) * 0.8
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


# ============================================================
# V1 PREDICTIONS
# ============================================================

v1_train_pred = v1_model.predict(
    X_train
)

v1_test_pred = v1_model.predict(
    X_test
)


# ============================================================
# RESIDUALS
# ============================================================

train_residual = (
    y_train.values
    - v1_train_pred
)


test_residual = (
    y_test.values
    - v1_test_pred
)


# ============================================================
# RESIDUAL MODEL
# ============================================================

residual_model = RandomForestRegressor(

    n_estimators=400,

    max_depth=6,

    min_samples_split=8,

    min_samples_leaf=4,

    max_features="sqrt",

    random_state=42,

    n_jobs=-1
)


residual_model.fit(
    X_train,
    train_residual
)


# ============================================================
# CORRECTION
# ============================================================

correction = residual_model.predict(
    X_test
)


v7_prediction = (
    v1_test_pred
    + correction
)


# ============================================================
# CLAMP
# ============================================================

v7_prediction = np.clip(
    v7_prediction,
    80,
    280
)


# ============================================================
# METRICS
# ============================================================

v1_mae = mean_absolute_error(
    y_test,
    v1_test_pred
)

v1_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        v1_test_pred
    )
)

v1_r2 = r2_score(
    y_test,
    v1_test_pred
)


v7_mae = mean_absolute_error(
    y_test,
    v7_prediction
)

v7_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        v7_prediction
    )
)

v7_r2 = r2_score(
    y_test,
    v7_prediction
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 60)
print("IPL SCORE MODEL V7")
print("RESIDUAL-BASED SCORE CORRECTION")
print("=" * 60)


print(
    "\nV1 BASELINE"
)

print(
    f"MAE  : {v1_mae:.2f}"
)

print(
    f"RMSE : {v1_rmse:.2f}"
)

print(
    f"R²   : {v1_r2:.4f}"
)


print(
    "\nV7 RESIDUAL CORRECTION"
)

print(
    f"MAE  : {v7_mae:.2f}"
)

print(
    f"RMSE : {v7_rmse:.2f}"
)

print(
    f"R²   : {v7_r2:.4f}"
)


print(
    "\nImprovement:"
)

print(
    f"MAE improvement  : "
    f"{v1_mae - v7_mae:.2f}"
)

print(
    f"RMSE improvement : "
    f"{v1_rmse - v7_rmse:.2f}"
)


# ============================================================
# ERROR BEHAVIOR
# ============================================================

comparison = pd.DataFrame({

    "actual":
        y_test.values,

    "v1":
        v1_test_pred,

    "v7":
        v7_prediction
})


comparison["v1_error"] = (
    comparison["v1"]
    - comparison["actual"]
)

comparison["v7_error"] = (
    comparison["v7"]
    - comparison["actual"]
)


comparison["v1_abs_error"] = (
    comparison["v1_error"].abs()
)

comparison["v7_abs_error"] = (
    comparison["v7_error"].abs()
)


print(
    "\n" + "=" * 60
)

print(
    "ERROR BEHAVIOR"
)

print(
    "=" * 60
)

print(
    f"\nV1 within ±15 runs: "
    f"{(
        comparison['v1_abs_error'] <= 15
    ).mean() * 100:.2f}%"
)

print(
    f"V7 within ±15 runs: "
    f"{(
        comparison['v7_abs_error'] <= 15
    ).mean() * 100:.2f}%"
)


# ============================================================
# BIGGEST V7 ERRORS
# ============================================================

comparison["date"] = (
    data.iloc[
        split_index:
    ]["date"].values
)

comparison["venue"] = (
    data.iloc[
        split_index:
    ]["venue"].values
)

comparison["batting_team"] = (
    data.iloc[
        split_index:
    ]["batting_team"].values
)


print(
    "\n" + "=" * 60
)

print(
    "BIGGEST V7 ERRORS"
)

print(
    "=" * 60
)

print(
    comparison
    .sort_values(
        "v7_abs_error",
        ascending=False
    )
    .head(10)
    [
        [
            "date",
            "venue",
            "batting_team",
            "actual",
            "v1",
            "v7",
            "v7_abs_error"
        ]
    ]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# SAVE EXPERIMENTAL MODEL
# ============================================================

joblib.dump(
    residual_model,
    "model/score_residual_model_v7.pkl"
)


print(
    "\n" + "=" * 60
)

print(
    "V7 EXPERIMENT COMPLETED"
)

print(
    "=" * 60
)

print(
    "\nV1 production model remains untouched."
)

print(
    "\nV7 residual model saved to:"
)

print(
    "model/score_residual_model_v7.pkl"
)