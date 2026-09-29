import pandas as pd
import numpy as np
import joblib

from collections import defaultdict, deque

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

MATCHES_PATH = "data/clean/matches.csv"
DELIVERIES_PATH = "data/clean/deliveries.csv"

MODEL_PATH = "model/ipl_score_model_v4.pkl"
FEATURE_PATH = "model/score_feature_columns_v4.pkl"


# ============================================================
# LOAD DATA
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
# FIRST INNINGS DATA
# ============================================================

innings_stats = (
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
    innings_stats,
    on="match_id",
    how="inner"
)


# ============================================================
# HISTORY
# ============================================================

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

team_venue_scores = defaultdict(
    lambda: deque(maxlen=10)
)

h2h_scores = defaultdict(
    lambda: deque(maxlen=10)
)


features = []
targets = []


# ============================================================
# FEATURE ENGINEERING
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


    # --------------------------------------------------------
    # HISTORIES
    # --------------------------------------------------------

    batting_scores = list(
        team_scores[batting_team]
    )

    bowling_scores = list(
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

    team_venue_key = (
        batting_team,
        venue
    )

    team_venue_history = list(
        team_venue_scores[
            team_venue_key
        ]
    )

    h2h_key = tuple(
        sorted(
            [batting_team, bowling_team]
        )
    )

    h2h_history = list(
        h2h_scores[h2h_key]
    )


    # ========================================================
    # BATTING
    # ========================================================

    recent_batting = batting_scores[-5:]

    batting_recent_avg = (
        np.mean(recent_batting)
        if recent_batting
        else 0
    )

    batting_long_avg = (
        np.mean(batting_scores)
        if batting_scores
        else 0
    )


    # Batting trend

    if len(batting_scores) >= 5:

        old_avg = np.mean(
            batting_scores[-5:-2]
        )

        new_avg = np.mean(
            batting_scores[-2:]
        )

        batting_trend = (
            new_avg - old_avg
        )

    else:

        batting_trend = 0


    # ========================================================
    # BOWLING
    # ========================================================

    recent_bowling_scores = (
        bowling_scores[-5:]
    )

    bowling_recent_avg = (
        np.mean(
            recent_bowling_scores
        )
        if recent_bowling_scores
        else 0
    )

    recent_bowling_wickets = (
        bowling_wickets[-5:]
    )

    bowling_wicket_avg = (
        np.mean(
            recent_bowling_wickets
        )
        if recent_bowling_wickets
        else 0
    )


    # ========================================================
    # VENUE
    # ========================================================

    venue_avg = (
        np.mean(venue_history)
        if venue_history
        else 0
    )

    venue_recent_avg = (
        np.mean(venue_history[-5:])
        if venue_history
        else 0
    )


    # ========================================================
    # TEAM + VENUE
    # ========================================================

    team_venue_avg = (
        np.mean(team_venue_history)
        if team_venue_history
        else 0
    )


    # ========================================================
    # H2H
    # ========================================================

    h2h_avg = (
        np.mean(h2h_history)
        if h2h_history
        else 0
    )


    # ========================================================
    # TOSS
    # ========================================================

    toss_winner_batting = int(
        match["toss_winner"]
        == batting_team
    )

    toss_bat_decision = int(
        match["toss_decision"]
        == "bat"
    )


    # ========================================================
    # FEATURE ROW
    # ========================================================

    row = {

        "season":
            match["season"],

        "batting_team_matches":
            team_matches[batting_team],

        "bowling_team_matches":
            team_matches[bowling_team],

        "batting_recent_avg":
            batting_recent_avg,

        "batting_long_avg":
            batting_long_avg,

        "batting_trend":
            batting_trend,

        "bowling_recent_avg":
            bowling_recent_avg,

        "bowling_wicket_avg":
            bowling_wicket_avg,

        "venue_avg":
            venue_avg,

        "venue_recent_avg":
            venue_recent_avg,

        "team_venue_avg":
            team_venue_avg,

        "h2h_avg":
            h2h_avg,

        "h2h_matches":
            len(h2h_history),

        "toss_winner_batting":
            toss_winner_batting,

        "toss_bat_decision":
            toss_bat_decision
    }


    features.append(row)

    targets.append(
        match["first_innings_score"]
    )


    # ========================================================
    # UPDATE HISTORY AFTER CURRENT MATCH
    # ========================================================

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

    team_venue_scores[
        team_venue_key
    ].append(score)

    h2h_scores[
        h2h_key
    ].append(score)


    team_matches[
        team1
    ] += 1

    team_matches[
        team2
    ] += 1


# ============================================================
# DATASET
# ============================================================

X = pd.DataFrame(features)

y = pd.Series(
    targets,
    name="first_innings_score"
)


print("=" * 60)
print("IPL SCORE MODEL V4")
print("=" * 60)

print(
    f"\nDataset shape: {X.shape}"
)

print(
    f"Number of features: {X.shape[1]}"
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


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestRegressor(

    n_estimators=500,

    max_depth=7,

    min_samples_split=8,

    min_samples_leaf=4,

    max_features=0.8,

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


predictions = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print(
    "\n" + "=" * 60
)

print(
    "V4 RESULTS"
)

print(
    "=" * 60
)

print(
    f"\nMAE  : {mae:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        X.columns,

    "importance":
        model.feature_importances_

}).sort_values(
    "importance",
    ascending=False
)


print(
    "\n" + "=" * 60
)

print(
    "FEATURE IMPORTANCE"
)

print(
    "=" * 60
)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# RETRAIN ON ALL DATA
# ============================================================

print(
    "\nRetraining V4 on full dataset..."
)


final_model = RandomForestRegressor(

    n_estimators=500,

    max_depth=7,

    min_samples_split=8,

    min_samples_leaf=4,

    max_features=0.8,

    random_state=42,

    n_jobs=-1
)


final_model.fit(
    X,
    y
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    final_model,
    MODEL_PATH
)

joblib.dump(
    list(X.columns),
    FEATURE_PATH
)


print(
    "\n" + "=" * 60
)

print(
    "V4 MODEL SAVED"
)

print(
    "=" * 60
)

print(
    f"\nModel: {MODEL_PATH}"
)

print(
    f"Features: {FEATURE_PATH}"
)

print(
    "\nV1 model remains untouched."
)