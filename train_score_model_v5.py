import pandas as pd
import numpy as np
import joblib

from collections import defaultdict, deque

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

MATCHES_PATH = "data/clean/matches.csv"
DELIVERIES_PATH = "data/clean/deliveries.csv"

REGIME_MODEL_PATH = "model/score_regime_model_v5.pkl"
SCORE_MODELS_PATH = "model/score_regression_models_v5.pkl"
FEATURE_PATH = "model/score_feature_columns_v5.pkl"


# ============================================================
# LOAD DATA
# ============================================================

matches = pd.read_csv(MATCHES_PATH)

deliveries = pd.read_csv(
    DELIVERIES_PATH
)

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

h2h_scores = defaultdict(
    lambda: deque(maxlen=10)
)


features = []
targets = []


# ============================================================
# BUILD FEATURES
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


    # ========================================================
    # BATTING
    # ========================================================

    batting_recent_5 = (
        np.mean(
            batting_history[-5:]
        )
        if batting_history
        else 0
    )

    batting_recent_10 = (
        np.mean(
            batting_history[-10:]
        )
        if batting_history
        else 0
    )

    batting_average = (
        np.mean(
            batting_history
        )
        if batting_history
        else 0
    )


    # ========================================================
    # BOWLING
    # ========================================================

    bowling_recent_5 = (
        np.mean(
            bowling_history[-5:]
        )
        if bowling_history
        else 0
    )

    bowling_recent_10 = (
        np.mean(
            bowling_history[-10:]
        )
        if bowling_history
        else 0
    )

    bowling_wickets_5 = (
        np.mean(
            bowling_wickets[-5:]
        )
        if bowling_wickets
        else 0
    )

    bowling_wickets_10 = (
        np.mean(
            bowling_wickets[-10:]
        )
        if bowling_wickets
        else 0
    )


    # ========================================================
    # VENUE
    # ========================================================

    venue_average = (
        np.mean(
            venue_history
        )
        if venue_history
        else 0
    )

    venue_recent = (
        np.mean(
            venue_history[-5:]
        )
        if venue_history
        else 0
    )


    # ========================================================
    # H2H
    # ========================================================

    h2h_average = (
        np.mean(
            h2h_history
        )
        if h2h_history
        else 0
    )

    h2h_recent = (
        np.mean(
            h2h_history[-5:]
        )
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
    # FEATURES
    # ========================================================

    row = {

        "season":
            match["season"],

        "batting_team_matches":
            team_matches[batting_team],

        "bowling_team_matches":
            team_matches[bowling_team],

        "batting_recent_5":
            batting_recent_5,

        "batting_recent_10":
            batting_recent_10,

        "batting_average":
            batting_average,

        "batting_wickets_5":
            (
                np.mean(
                    batting_wickets[-5:]
                )
                if batting_wickets
                else 0
            ),

        "bowling_recent_5":
            bowling_recent_5,

        "bowling_recent_10":
            bowling_recent_10,

        "bowling_wickets_5":
            bowling_wickets_5,

        "bowling_wickets_10":
            bowling_wickets_10,

        "venue_average":
            venue_average,

        "venue_recent":
            venue_recent,

        "venue_matches":
            len(venue_history),

        "h2h_average":
            h2h_average,

        "h2h_recent":
            h2h_recent,

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
    # UPDATE HISTORY AFTER FEATURES
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

X = pd.DataFrame(
    features
)

y = pd.Series(
    targets,
    name="score"
)


# ============================================================
# SCORING REGIME
# ============================================================

def get_regime(score):

    if score < 160:
        return "LOW"

    elif score < 220:
        return "NORMAL"

    return "HIGH"


regimes = y.apply(
    get_regime
)


print("=" * 60)
print("IPL SCORE MODEL V5")
print("TWO-STAGE SCORE PREDICTION")
print("=" * 60)

print(
    f"\nDataset shape: {X.shape}"
)

print(
    "\nRegime distribution:"
)

print(
    regimes.value_counts()
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

regime_train = regimes.iloc[
    :split_index
]

regime_test = regimes.iloc[
    split_index:
]


# ============================================================
# STAGE 1 — REGIME CLASSIFIER
# ============================================================

regime_model = RandomForestClassifier(

    n_estimators=400,

    max_depth=8,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


regime_model.fit(
    X_train,
    regime_train
)


regime_predictions = (
    regime_model.predict(
        X_test
    )
)


regime_accuracy = accuracy_score(
    regime_test,
    regime_predictions
)


print(
    "\n" + "=" * 60
)

print(
    "STAGE 1 — REGIME CLASSIFIER"
)

print(
    "=" * 60
)

print(
    f"\nAccuracy: {regime_accuracy:.4f}"
)

print(
    f"\n{classification_report(regime_test, regime_predictions, zero_division=0)}"
)


# ============================================================
# STAGE 2 — REGRESSION MODELS
# ============================================================

regression_models = {}

regression_predictions = {}


for regime in [
    "LOW",
    "NORMAL",
    "HIGH"
]:

    mask = (
        regime_train == regime
    )

    X_regime = X_train[
        mask
    ]

    y_regime = y_train[
        mask
    ]


    print(
        f"\nTraining {regime} model..."
    )

    model = RandomForestRegressor(

        n_estimators=400,

        max_depth=8,

        min_samples_split=5,

        min_samples_leaf=2,

        max_features="sqrt",

        random_state=42,

        n_jobs=-1
    )


    model.fit(
        X_regime,
        y_regime
    )

    regression_models[
        regime
    ] = model


# ============================================================
# STAGE 2 PREDICTION
# ============================================================

final_predictions = []


for i in range(
    len(X_test)
):

    regime = regime_predictions[i]

    model = regression_models[
        regime
    ]

    prediction = model.predict(
        X_test.iloc[
            [i]
        ]
    )[0]

    final_predictions.append(
        prediction
    )


final_predictions = np.array(
    final_predictions
)


# ============================================================
# FINAL METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    final_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_predictions
    )
)

r2 = r2_score(
    y_test,
    final_predictions
)


print(
    "\n" + "=" * 60
)

print(
    "STAGE 2 — FINAL SCORE RESULTS"
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
# SCORE RANGE PERFORMANCE
# ============================================================

evaluation = pd.DataFrame({

    "actual":
        y_test.values,

    "predicted":
        final_predictions,

    "regime":
        regime_predictions

})


evaluation["error"] = (
    evaluation["predicted"]
    - evaluation["actual"]
)

evaluation["absolute_error"] = (
    evaluation["error"].abs()
)


def score_range(score):

    if score < 130:
        return "Below 130"

    elif score < 160:
        return "130-159"

    elif score < 190:
        return "160-189"

    elif score < 220:
        return "190-219"

    return "220+"


evaluation["score_range"] = (
    evaluation["actual"]
    .apply(score_range)
)


range_results = (
    evaluation
    .groupby("score_range")
    .agg(
        matches=("actual", "count"),
        actual_average=("actual", "mean"),
        predicted_average=("predicted", "mean"),
        MAE=("absolute_error", "mean")
    )
)


print(
    "\n" + "=" * 60
)

print(
    "ERROR BY SCORE RANGE"
)

print(
    "=" * 60
)

print(
    range_results.round(2).to_string()
)


# ============================================================
# SAVE V5
# ============================================================

joblib.dump(
    regime_model,
    REGIME_MODEL_PATH
)

joblib.dump(
    regression_models,
    SCORE_MODELS_PATH
)

joblib.dump(
    list(X.columns),
    FEATURE_PATH
)


print(
    "\n" + "=" * 60
)

print(
    "V5 MODELS SAVED"
)

print(
    "=" * 60
)

print(
    f"\nRegime model: {REGIME_MODEL_PATH}"
)

print(
    f"Score models: {SCORE_MODELS_PATH}"
)

print(
    f"Features: {FEATURE_PATH}"
)

print(
    "\nV1 remains untouched."
)