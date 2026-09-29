import pandas as pd
import numpy as np
import joblib

from collections import defaultdict, deque
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
# FIRST INNINGS
# ============================================================

innings = (
    deliveries[
        deliveries["innings"] == 1
    ]
    .groupby("match_id")
    .agg(
        first_innings_score=("runs_total", "sum"),
        first_innings_wickets=("wicket", "sum"),
        batting_team=("batting_team", "first")
    )
    .reset_index()
)

data = matches.merge(
    innings,
    on="match_id",
    how="inner"
)


# ============================================================
# HISTORIES
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
# BUILD EXACT V1 FEATURES
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


    # Update only after feature creation
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
# PREPARE DATA
# ============================================================

X = pd.DataFrame(features)

y = pd.Series(
    targets,
    name="actual_score"
)

model = joblib.load(
    MODEL_PATH
)

feature_columns = joblib.load(
    FEATURE_PATH
)

X = X.reindex(
    columns=feature_columns,
    fill_value=0
)


# ============================================================
# CHRONOLOGICAL TEST SET
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

test_data = data.iloc[
    split_index:
].copy()

predictions = model.predict(
    X_test
)


# ============================================================
# EVALUATION DATAFRAME
# ============================================================

results = pd.DataFrame({

    "date":
        test_data["date"].values,

    "season":
        test_data["season"].values,

    "venue":
        test_data["venue"].values,

    "batting_team":
        test_data["batting_team"].values,

    "actual":
        y_test.values,

    "predicted":
        predictions
})

results["error"] = (
    results["predicted"]
    - results["actual"]
)

results["absolute_error"] = (
    results["error"].abs()
)


# ============================================================
# OVERALL
# ============================================================

mae = mean_absolute_error(
    results["actual"],
    results["predicted"]
)

rmse = np.sqrt(
    mean_squared_error(
        results["actual"],
        results["predicted"]
    )
)

r2 = r2_score(
    results["actual"],
    results["predicted"]
)

mean_error = results["error"].mean()


print("=" * 65)
print("IPL SCORE MODEL — FINAL VALIDATION")
print("=" * 65)

print(
    f"\nTest samples : {len(results)}"
)

print(
    f"MAE          : {mae:.2f}"
)

print(
    f"RMSE         : {rmse:.2f}"
)

print(
    f"R²           : {r2:.4f}"
)

print(
    f"Mean error   : {mean_error:.2f}"
)


# ============================================================
# SCORE RANGE
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

    return "220+"


results["score_range"] = (
    results["actual"]
    .apply(score_range)
)


range_analysis = (
    results
    .groupby("score_range")
    .agg(
        matches=("actual", "count"),
        actual_average=("actual", "mean"),
        predicted_average=("predicted", "mean"),
        MAE=("absolute_error", "mean"),
        mean_error=("error", "mean")
    )
)

print(
    "\n" + "=" * 65
)

print(
    "ERROR BY SCORE RANGE"
)

print(
    "=" * 65
)

print(
    range_analysis.round(2).to_string()
)


# ============================================================
# VENUE
# ============================================================

venue_analysis = (
    results
    .groupby("venue")
    .agg(
        matches=("actual", "count"),
        actual_average=("actual", "mean"),
        predicted_average=("predicted", "mean"),
        MAE=("absolute_error", "mean"),
        mean_error=("error", "mean")
    )
    .sort_values(
        "MAE",
        ascending=False
    )
)

print(
    "\n" + "=" * 65
)

print(
    "VENUE ANALYSIS — TOP 10 BY MAE"
)

print(
    "=" * 65
)

print(
    venue_analysis
    .head(10)
    .round(2)
    .to_string()
)


# ============================================================
# SEASON
# ============================================================

season_analysis = (
    results
    .groupby("season")
    .agg(
        matches=("actual", "count"),
        actual_average=("actual", "mean"),
        predicted_average=("predicted", "mean"),
        MAE=("absolute_error", "mean"),
        mean_error=("error", "mean")
    )
)

print(
    "\n" + "=" * 65
)

print(
    "SEASON ANALYSIS"
)

print(
    "=" * 65
)

print(
    season_analysis
    .round(2)
    .to_string()
)


# ============================================================
# TEAM
# ============================================================

team_analysis = (
    results
    .groupby("batting_team")
    .agg(
        matches=("actual", "count"),
        actual_average=("actual", "mean"),
        predicted_average=("predicted", "mean"),
        MAE=("absolute_error", "mean"),
        mean_error=("error", "mean")
    )
    .sort_values(
        "MAE",
        ascending=False
    )
)

print(
    "\n" + "=" * 65
)

print(
    "TEAM ANALYSIS"
)

print(
    "=" * 65
)

print(
    team_analysis
    .round(2)
    .to_string()
)


# ============================================================
# BIAS
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "PREDICTION BEHAVIOR"
)

print(
    "=" * 65
)

under = (
    results["error"] < -15
).sum()

over = (
    results["error"] > 15
).sum()

within = (
    results["absolute_error"] <= 15
).sum()

print(
    f"\nUnder-predicted by >15 runs : {under}"
)

print(
    f"Over-predicted by >15 runs  : {over}"
)

print(
    f"Within ±15 runs             : {within}"
)

print(
    f"Within ±15 percentage       : "
    f"{within / len(results) * 100:.2f}%"
)


# ============================================================
# EXTREME SCORES
# ============================================================

extreme = results[
    (results["actual"] < 130)
    |
    (results["actual"] >= 220)
]

print(
    "\n" + "=" * 65
)

print(
    "EXTREME SCORE PERFORMANCE"
)

print(
    "=" * 65
)

if len(extreme) > 0:

    extreme_mae = mean_absolute_error(
        extreme["actual"],
        extreme["predicted"]
    )

    print(
        f"\nExtreme-score matches : "
        f"{len(extreme)}"
    )

    print(
        f"Extreme-score MAE     : "
        f"{extreme_mae:.2f}"
    )

    print(
        f"Actual average        : "
        f"{extreme['actual'].mean():.2f}"
    )

    print(
        f"Predicted average     : "
        f"{extreme['predicted'].mean():.2f}"
    )


# ============================================================
# FILES
# ============================================================

results.to_csv(
    "score_final_validation.csv",
    index=False
)

range_analysis.to_csv(
    "score_final_range_analysis.csv"
)

venue_analysis.to_csv(
    "score_final_venue_analysis.csv"
)

season_analysis.to_csv(
    "score_final_season_analysis.csv"
)

team_analysis.to_csv(
    "score_final_team_analysis.csv"
)


print(
    "\n" + "=" * 65
)

print(
    "FINAL VALIDATION FILES CREATED"
)

print(
    "=" * 65
)

print(
    "\n1. score_final_validation.csv"
)

print(
    "2. score_final_range_analysis.csv"
)

print(
    "3. score_final_venue_analysis.csv"
)

print(
    "4. score_final_season_analysis.csv"
)

print(
    "5. score_final_team_analysis.csv"
)