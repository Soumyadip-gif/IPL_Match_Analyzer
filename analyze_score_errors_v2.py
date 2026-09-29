import pandas as pd
import numpy as np
import joblib

from collections import defaultdict, deque

from sklearn.metrics import mean_absolute_error, mean_squared_error


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

deliveries = deliveries.sort_values(
    ["match_id", "innings", "over", "ball"]
)


# ============================================================
# FIRST INNINGS TARGET
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


feature_rows = []
targets = []

metadata = []


# ============================================================
# REBUILD V1 FEATURES
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
    # V1 FEATURES
    # ========================================================

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
            np.sum(
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


    feature_rows.append(row)

    targets.append(
        match["first_innings_score"]
    )


    metadata.append({

        "match_id":
            match["match_id"],

        "date":
            match["date"],

        "season":
            match["season"],

        "venue":
            venue,

        "batting_team":
            batting_team,

        "bowling_team":
            bowling_team,

        "actual_score":
            match["first_innings_score"]
    })


    # ========================================================
    # UPDATE HISTORY
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
    feature_rows
)

y = pd.Series(
    targets
)

metadata_df = pd.DataFrame(
    metadata
)


# ============================================================
# CHRONOLOGICAL TEST SET
# ============================================================

split_index = int(
    len(X) * 0.8
)

X_test = X.iloc[
    split_index:
].copy()

y_test = y.iloc[
    split_index:
].copy()

meta_test = metadata_df.iloc[
    split_index:
].copy()


# ============================================================
# LOAD V1 MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)

feature_columns = joblib.load(
    FEATURE_PATH
)


X_test = X_test.reindex(
    columns=feature_columns,
    fill_value=0
)


predictions = model.predict(
    X_test
)


# ============================================================
# ERROR DATAFRAME
# ============================================================

results = meta_test.copy()

results["predicted_score"] = predictions

results["error"] = (
    results["predicted_score"]
    - results["actual_score"]
)

results["absolute_error"] = (
    results["error"].abs()
)


results["error_percent"] = (
    results["absolute_error"]
    /
    results["actual_score"].clip(
        lower=1
    )
    * 100
)


# ============================================================
# OVERALL
# ============================================================

print("=" * 60)
print("IPL SCORE MODEL — DEEP ERROR ANALYSIS")
print("=" * 60)

print(
    f"\nTest samples: {len(results)}"
)

print(
    f"MAE: {results['absolute_error'].mean():.2f}"
)

print(
    f"RMSE: {np.sqrt(np.mean(results['error'] ** 2)):.2f}"
)

print(
    f"Mean Error: {results['error'].mean():.2f}"
)


# ============================================================
# SCORE RANGE ANALYSIS
# ============================================================

def score_range(score):

    if score < 130:
        return "Below 130"

    elif score < 160:
        return "130-159"

    elif score < 190:
        return "160-189"

    elif score < 220:
        return "190-219"

    else:
        return "220+"


results["score_range"] = (
    results["actual_score"]
    .apply(score_range)
)


range_analysis = (
    results
    .groupby("score_range")
    .agg(
        matches=("actual_score", "count"),
        actual_average=("actual_score", "mean"),
        predicted_average=("predicted_score", "mean"),
        MAE=("absolute_error", "mean"),
        mean_error=("error", "mean")
    )
    .sort_index()
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
    range_analysis.round(2).to_string()
)


# ============================================================
# VENUE ANALYSIS
# ============================================================

venue_analysis = (
    results
    .groupby("venue")
    .agg(
        matches=("actual_score", "count"),
        actual_average=("actual_score", "mean"),
        predicted_average=("predicted_score", "mean"),
        MAE=("absolute_error", "mean"),
        mean_error=("error", "mean")
    )
    .sort_values(
        "MAE",
        ascending=False
    )
)


print(
    "\n" + "=" * 60
)

print(
    "WORST VENUES BY MAE"
)

print(
    "=" * 60
)

print(
    venue_analysis
    .head(10)
    .round(2)
    .to_string()
)


# ============================================================
# SEASON ANALYSIS
# ============================================================

season_analysis = (
    results
    .groupby("season")
    .agg(
        matches=("actual_score", "count"),
        actual_average=("actual_score", "mean"),
        predicted_average=("predicted_score", "mean"),
        MAE=("absolute_error", "mean"),
        mean_error=("error", "mean")
    )
    .sort_index()
)


print(
    "\n" + "=" * 60
)

print(
    "ERROR BY SEASON"
)

print(
    "=" * 60
)

print(
    season_analysis.round(2).to_string()
)


# ============================================================
# TEAM ANALYSIS
# ============================================================

team_analysis = (
    results
    .groupby("batting_team")
    .agg(
        matches=("actual_score", "count"),
        actual_average=("actual_score", "mean"),
        predicted_average=("predicted_score", "mean"),
        MAE=("absolute_error", "mean"),
        mean_error=("error", "mean")
    )
    .sort_values(
        "MAE",
        ascending=False
    )
)


print(
    "\n" + "=" * 60
)

print(
    "TEAM ERROR ANALYSIS"
)

print(
    "=" * 60
)

print(
    team_analysis.round(2).to_string()
)


# ============================================================
# BIGGEST ERRORS
# ============================================================

biggest_errors = (
    results
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(15)
)


print(
    "\n" + "=" * 60
)

print(
    "15 BIGGEST PREDICTION ERRORS"
)

print(
    "=" * 60
)

print(
    biggest_errors[
        [
            "match_id",
            "date",
            "venue",
            "batting_team",
            "actual_score",
            "predicted_score",
            "error",
            "absolute_error"
        ]
    ]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# UNDER / OVER PREDICTION
# ============================================================

under = (
    results["error"] < -15
).sum()

over = (
    results["error"] > 15
).sum()

close = (
    results["absolute_error"] <= 15
).sum()


print(
    "\n" + "=" * 60
)

print(
    "PREDICTION BEHAVIOR"
)

print(
    "=" * 60
)

print(
    f"\nUnder-predicted by >15 runs : {under}"
)

print(
    f"Over-predicted by >15 runs  : {over}"
)

print(
    f"Within ±15 runs             : {close}"
)

print(
    f"Within ±15 percentage       : "
    f"{close / len(results) * 100:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    "score_error_analysis.csv",
    index=False
)

range_analysis.to_csv(
    "score_range_analysis.csv"
)

venue_analysis.to_csv(
    "score_venue_analysis.csv"
)

season_analysis.to_csv(
    "score_season_analysis.csv"
)

team_analysis.to_csv(
    "score_team_analysis.csv"
)


print(
    "\n" + "=" * 60
)

print(
    "ANALYSIS FILES CREATED"
)

print(
    "=" * 60
)

print(
    "\n1. score_error_analysis.csv"
)

print(
    "2. score_range_analysis.csv"
)

print(
    "3. score_venue_analysis.csv"
)

print(
    "4. score_season_analysis.csv"
)

print(
    "5. score_team_analysis.csv"
)