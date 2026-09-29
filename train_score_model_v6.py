import pandas as pd
import numpy as np
import joblib

from collections import defaultdict, deque

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MATCHES_PATH = "data/clean/matches.csv"
DELIVERIES_PATH = "data/clean/deliveries.csv"

matches = pd.read_csv(MATCHES_PATH)
deliveries = pd.read_csv(DELIVERIES_PATH)

matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

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
    deliveries[deliveries["innings"] == 1]
    .groupby("match_id")
    .agg(
        first_innings_score=("runs_total", "sum"),
        first_innings_wickets=("wicket", "sum"),
        batting_team=("batting_team", "first")
    )
    .reset_index()
)

data = matches.merge(
    innings_stats,
    on="match_id",
    how="inner"
)


# ============================================================
# HISTORIES
# ============================================================

team_scores = defaultdict(lambda: deque(maxlen=10))
team_wickets = defaultdict(lambda: deque(maxlen=10))
team_matches = defaultdict(int)

venue_scores = defaultdict(lambda: deque(maxlen=20))

h2h_scores = defaultdict(lambda: deque(maxlen=10))

team_venue_scores = defaultdict(
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

    batting_history = list(
        team_scores[batting_team]
    )

    bowling_history = list(
        team_scores[bowling_team]
    )

    batting_wicket_history = list(
        team_wickets[batting_team]
    )

    bowling_wicket_history = list(
        team_wickets[bowling_team]
    )

    venue_history = list(
        venue_scores[venue]
    )

    h2h_key = tuple(
        sorted([batting_team, bowling_team])
    )

    h2h_history = list(
        h2h_scores[h2h_key]
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


    # ========================================================
    # RECENT BATTING
    # ========================================================

    batting_recent_3 = (
        np.mean(batting_history[-3:])
        if batting_history
        else 0
    )

    batting_recent_5 = (
        np.mean(batting_history[-5:])
        if batting_history
        else 0
    )

    batting_recent_10 = (
        np.mean(batting_history[-10:])
        if batting_history
        else 0
    )


    batting_std_5 = (
        np.std(batting_history[-5:])
        if len(batting_history) >= 2
        else 0
    )


    # ========================================================
    # BATTING TREND
    # ========================================================

    if len(batting_history) >= 5:

        recent_avg = np.mean(
            batting_history[-3:]
        )

        older_avg = np.mean(
            batting_history[-5:-2]
        )

        batting_trend = (
            recent_avg - older_avg
        )

    else:

        batting_trend = 0


    # ========================================================
    # BOWLING
    # ========================================================

    bowling_recent_3 = (
        np.mean(bowling_history[-3:])
        if bowling_history
        else 0
    )

    bowling_recent_5 = (
        np.mean(bowling_history[-5:])
        if bowling_history
        else 0
    )

    bowling_recent_10 = (
        np.mean(bowling_history[-10:])
        if bowling_history
        else 0
    )


    bowling_wickets_5 = (
        np.mean(
            bowling_wicket_history[-5:]
        )
        if bowling_wicket_history
        else 0
    )


    bowling_wickets_10 = (
        np.mean(
            bowling_wicket_history[-10:]
        )
        if bowling_wicket_history
        else 0
    )


    # ========================================================
    # VENUE
    # ========================================================

    venue_average = (
        np.mean(venue_history)
        if venue_history
        else 0
    )

    venue_recent_5 = (
        np.mean(venue_history[-5:])
        if venue_history
        else 0
    )

    venue_std = (
        np.std(venue_history)
        if len(venue_history) >= 2
        else 0
    )


    # ========================================================
    # TEAM + VENUE
    # ========================================================

    team_venue_average = (
        np.mean(team_venue_history)
        if team_venue_history
        else venue_average
    )


    # ========================================================
    # H2H
    # ========================================================

    h2h_average = (
        np.mean(h2h_history)
        if h2h_history
        else 0
    )

    h2h_recent = (
        np.mean(h2h_history[-5:])
        if h2h_history
        else 0
    )

    h2h_std = (
        np.std(h2h_history)
        if len(h2h_history) >= 2
        else 0
    )


    # ========================================================
    # TOSS
    # ========================================================

    toss_winner_batting = int(
        match["toss_winner"] == batting_team
    )

    toss_bat_decision = int(
        match["toss_decision"] == "bat"
    )


    # ========================================================
    # SCORING STRENGTH
    # ========================================================

    if batting_history:

        batting_strength = (
            0.5 * batting_recent_3
            + 0.3 * batting_recent_5
            + 0.2 * batting_recent_10
        )

    else:

        batting_strength = 0


    if bowling_history:

        bowling_strength = (
            0.5 * bowling_recent_3
            + 0.3 * bowling_recent_5
            + 0.2 * bowling_recent_10
        )

    else:

        bowling_strength = 0


    # Difference between batting and opponent scoring strength
    team_strength_difference = (
        batting_strength
        - bowling_strength
    )


    # ========================================================
    # FEATURES
    # ========================================================

    row = {

        "batting_team_matches":
            team_matches[batting_team],

        "bowling_team_matches":
            team_matches[bowling_team],

        "batting_recent_3":
            batting_recent_3,

        "batting_recent_5":
            batting_recent_5,

        "batting_recent_10":
            batting_recent_10,

        "batting_std_5":
            batting_std_5,

        "batting_trend":
            batting_trend,

        "bowling_recent_3":
            bowling_recent_3,

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

        "venue_recent_5":
            venue_recent_5,

        "venue_std":
            venue_std,

        "team_venue_average":
            team_venue_average,

        "h2h_average":
            h2h_average,

        "h2h_recent":
            h2h_recent,

        "h2h_std":
            h2h_std,

        "batting_strength":
            batting_strength,

        "bowling_strength":
            bowling_strength,

        "team_strength_difference":
            team_strength_difference,

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
    # UPDATE HISTORY
    # ========================================================

    score = match["first_innings_score"]

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

    team_venue_scores[
        team_venue_key
    ].append(score)

    team_matches[team1] += 1
    team_matches[team2] += 1


# ============================================================
# DATASET
# ============================================================

X = pd.DataFrame(features)
y = pd.Series(targets, name="score")

print("=" * 60)
print("IPL SCORE MODEL V6")
print("TARGETED V1 IMPROVEMENT")
print("=" * 60)

print(
    f"\nDataset shape: {X.shape}"
)

print(
    f"Target mean: {y.mean():.2f}"
)

print(
    f"Target std: {y.std():.2f}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(
    len(X) * 0.8
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Test samples: {len(X_test)}"
)


# ============================================================
# RANDOM FOREST
# ============================================================

rf = RandomForestRegressor(

    n_estimators=500,

    max_depth=8,

    min_samples_split=5,

    min_samples_leaf=2,

    max_features="sqrt",

    random_state=42,

    n_jobs=-1
)


rf.fit(
    X_train,
    y_train
)


rf_pred = rf.predict(
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
# GRADIENT BOOSTING
# ============================================================

gb = GradientBoostingRegressor(

    n_estimators=300,

    learning_rate=0.03,

    max_depth=3,

    min_samples_split=5,

    min_samples_leaf=3,

    loss="huber",

    random_state=42
)


gb.fit(
    X_train,
    y_train
)


gb_pred = gb.predict(
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
# RESULTS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "MODEL COMPARISON"
)

print(
    "=" * 60
)

print(
    f"\nRandom Forest"
)

print(
    f"MAE  : {rf_mae:.2f}"
)

print(
    f"RMSE : {rf_rmse:.2f}"
)

print(
    f"R²   : {rf_r2:.4f}"
)


print(
    f"\nGradient Boosting"
)

print(
    f"MAE  : {gb_mae:.2f}"
)

print(
    f"RMSE : {gb_rmse:.2f}"
)

print(
    f"R²   : {gb_r2:.4f}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TOP FEATURES"
)

print(
    "=" * 60
)

importance = pd.DataFrame({

    "feature":
        X.columns,

    "importance":
        rf.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# DO NOT SAVE AS PRODUCTION MODEL YET
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "V6 EXPERIMENT COMPLETED"
)

print(
    "=" * 60
)

print(
    "\nV1 production model remains untouched."
)

print(
    "\nCompare V6 against V1:"
)

print(
    "V1 MAE  = 27.55"
)

print(
    "V1 RMSE = 35.02"
)

print(
    "V1 R²   = 0.1560"
)