import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

MATCHES_PATH = "data/clean/matches.csv"
DELIVERIES_PATH = "data/clean/deliveries.csv"

MODEL_PATH = "model/ipl_score_model.pkl"
FEATURE_PATH = "model/score_feature_columns.pkl"


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


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# ============================================================
# PREPARE FIRST-INNINGS SCORE
# ============================================================

first_innings = (
    deliveries[
        deliveries["innings"] == 1
    ]
    .groupby("match_id")
    .agg(
        first_innings_score=(
            "runs_total",
            "sum"
        )
    )
    .reset_index()
)


data = matches.merge(
    first_innings,
    on="match_id",
    how="inner"
)


# ============================================================
# BUILD SCORE FEATURES
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


feature_rows = []
targets = []


for _, match in data.iterrows():

    match_id = match["match_id"]

    team1 = match["team1"]
    team2 = match["team2"]

    venue = match["venue"]

    toss_winner = match["toss_winner"]
    toss_decision = match["toss_decision"]


    # --------------------------------------------------------
    # Determine first innings batting team
    # --------------------------------------------------------

    match_deliveries = deliveries[
        deliveries["match_id"] == match_id
    ]

    innings_1 = match_deliveries[
        match_deliveries["innings"] == 1
    ]

    if innings_1.empty:
        continue


    batting_team = innings_1[
        "batting_team"
    ].iloc[0]


    if batting_team == team1:
        bowling_team = team2
    else:
        bowling_team = team1


    # --------------------------------------------------------
    # Historical features BEFORE current match
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

    h2h_key = tuple(
        sorted(
            [batting_team, bowling_team]
        )
    )

    h2h_history = list(
        h2h_scores[h2h_key]
    )


    # --------------------------------------------------------
    # Feature values
    # --------------------------------------------------------

    batting_recent_score_5 = (
        np.mean(
            batting_scores[-5:]
        )
        if batting_scores
        else 0
    )

    batting_recent_score_10 = (
        np.mean(
            batting_scores[-10:]
        )
        if batting_scores
        else 0
    )

    batting_recent_runs_5 = (
        np.sum(
            batting_scores[-5:]
        )
        if batting_scores
        else 0
    )

    batting_average_score = (
        np.mean(batting_scores)
        if batting_scores
        else 0
    )

    batting_team_wickets_5 = (
        np.mean(
            batting_wickets[-5:]
        )
        if batting_wickets
        else 0
    )


    bowling_recent_score_5 = (
        np.mean(
            bowling_scores[-5:]
        )
        if bowling_scores
        else 0
    )

    bowling_recent_wickets_5 = (
        np.mean(
            bowling_wickets[-5:]
        )
        if bowling_wickets
        else 0
    )

    bowling_recent_wickets_10 = (
        np.mean(
            bowling_wickets[-10:]
        )
        if bowling_wickets
        else 0
    )

    bowling_average_wickets = (
        np.mean(bowling_wickets)
        if bowling_wickets
        else 0
    )


    venue_average_score = (
        np.mean(venue_history)
        if venue_history
        else 0
    )

    venue_recent_score = (
        np.mean(
            venue_history[-5:]
        )
        if venue_history
        else 0
    )

    venue_matches = len(
        venue_history
    )


    h2h_average_score = (
        np.mean(h2h_history)
        if h2h_history
        else 0
    )

    h2h_recent_score = (
        np.mean(
            h2h_history[-5:]
        )
        if h2h_history
        else 0
    )

    h2h_matches = len(
        h2h_history
    )


    # --------------------------------------------------------
    # Toss features
    # --------------------------------------------------------

    toss_winner_batting = int(
        toss_winner == batting_team
    )


    toss_bat_decision = int(
        toss_decision == "bat"
    )


    # --------------------------------------------------------
    # Build feature row
    # --------------------------------------------------------

    row = {

        "season":
            match["season"],

        "batting_team_matches":
            team_matches[batting_team],

        "bowling_team_matches":
            team_matches[bowling_team],

        "batting_recent_score_5":
            batting_recent_score_5,

        "batting_recent_score_10":
            batting_recent_score_10,

        "batting_recent_runs_5":
            batting_recent_runs_5,

        "batting_average_score":
            batting_average_score,

        "batting_team_wickets_5":
            batting_team_wickets_5,

        "bowling_recent_score_5":
            bowling_recent_score_5,

        "bowling_recent_wickets_5":
            bowling_recent_wickets_5,

        "bowling_recent_wickets_10":
            bowling_recent_wickets_10,

        "bowling_average_wickets":
            bowling_average_wickets,

        "venue_average_score":
            venue_average_score,

        "venue_recent_score":
            venue_recent_score,

        "venue_matches":
            venue_matches,

        "h2h_average_score":
            h2h_average_score,

        "h2h_recent_score":
            h2h_recent_score,

        "h2h_matches":
            h2h_matches,

        "toss_winner_batting":
            toss_winner_batting,

        "toss_bat_decision":
            toss_bat_decision

    }


    feature_rows.append(row)

    targets.append(
        match["first_innings_score"]
    )


    # --------------------------------------------------------
    # Update history AFTER feature creation
    # --------------------------------------------------------

    current_score = match[
        "first_innings_score"
    ]


    innings_wickets = int(
        innings_1["wicket"]
        .sum()
    )


    team_scores[batting_team].append(
        current_score
    )

    team_wickets[bowling_team].append(
        innings_wickets
    )

    venue_scores[venue].append(
        current_score
    )

    h2h_scores[h2h_key].append(
        current_score
    )

    team_matches[team1] += 1
    team_matches[team2] += 1


# ============================================================
# DATAFRAME
# ============================================================

X = pd.DataFrame(
    feature_rows
)

y = pd.Series(
    targets,
    name="actual_score"
)


X = X.reindex(
    columns=feature_columns,
    fill_value=0
)


# ============================================================
# CHRONOLOGICAL TEST SPLIT
# ============================================================

split_index = int(
    len(X) * 0.8
)

X_test = X.iloc[
    split_index:
]

y_test = y.iloc[
    split_index:
]


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)


predictions = np.round(
    predictions
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


print("\n" + "=" * 60)
print("IPL SCORE MODEL — ERROR ANALYSIS")
print("=" * 60)

print(
    f"\nDataset size       : {len(X)}"
)

print(
    f"Test samples       : {len(X_test)}"
)

print(
    f"Actual mean score  : {y_test.mean():.2f}"
)

print(
    f"Predicted mean     : {predictions.mean():.2f}"
)

print(
    f"\nMAE                : {mae:.2f} runs"
)

print(
    f"RMSE               : {rmse:.2f} runs"
)

print(
    f"R²                 : {r2:.4f}"
)


# ============================================================
# ERROR ANALYSIS
# ============================================================

errors = (
    predictions - y_test.values
)

absolute_errors = np.abs(
    errors
)


print(
    f"\nMean error         : {errors.mean():.2f} runs"
)

print(
    f"Median error       : "
    f"{np.median(absolute_errors):.2f} runs"
)

print(
    f"Max error          : "
    f"{absolute_errors.max():.2f} runs"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if hasattr(
    model,
    "feature_importances_"
):

    importance = pd.DataFrame({

        "feature":
            feature_columns,

        "importance":
            model.feature_importances_

    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )


    print(
        "\n" + "=" * 60
    )

    print(
        "TOP SCORE MODEL FEATURES"
    )

    print(
        "=" * 60
    )

    print(
        importance.head(10).to_string(
            index=False
        )
    )


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    y_test,
    predictions,
    alpha=0.6
)

min_value = min(
    y_test.min(),
    predictions.min()
)

max_value = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual First Innings Score"
)

plt.ylabel(
    "Predicted First Innings Score"
)

plt.title(
    "IPL Score Model — Actual vs Predicted"
)

plt.tight_layout()

plt.savefig(
    "score_actual_vs_predicted.png",
    dpi=150
)

plt.show()


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    errors,
    bins=25
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Prediction Error (Predicted - Actual)"
)

plt.ylabel(
    "Number of Matches"
)

plt.title(
    "IPL Score Model — Error Distribution"
)

plt.tight_layout()

plt.savefig(
    "score_error_distribution.png",
    dpi=150
)

plt.show()


# ============================================================
# FEATURE IMPORTANCE CHART
# ============================================================

if hasattr(
    model,
    "feature_importances_"
):

    top_features = importance.head(10)

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 10 Score Prediction Features"
    )

    plt.tight_layout()

    plt.savefig(
        "score_feature_importance.png",
        dpi=150
    )

    plt.show()


print(
    "\nAnalysis completed successfully."
)

print(
    "Generated files:"
)

print(
    "1. score_actual_vs_predicted.png"
)

print(
    "2. score_error_distribution.png"
)

print(
    "3. score_feature_importance.png"
)