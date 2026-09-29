import pandas as pd
import numpy as np
import joblib

from collections import defaultdict, deque

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

MATCHES_PATH = "data/clean/matches.csv"
DELIVERIES_PATH = "data/clean/deliveries.csv"

MODEL_PATH = "model/ipl_score_model_v3.pkl"
FEATURE_PATH = "model/score_feature_columns_v3.pkl"


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
# FIRST INNINGS SCORE + WICKETS
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
# HISTORY STRUCTURES
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

venue_team_scores = defaultdict(
    lambda: deque(maxlen=10)
)

h2h_scores = defaultdict(
    lambda: deque(maxlen=10)
)


# ============================================================
# FEATURE STORAGE
# ============================================================

feature_rows = []
targets = []


# ============================================================
# BUILD LEAKAGE-SAFE FEATURES
# ============================================================

for _, match in data.iterrows():

    team1 = match["team1"]
    team2 = match["team2"]

    venue = match["venue"]

    batting_team = match["batting_team"]

    if batting_team == team1:
        bowling_team = team2
    else:
        bowling_team = team1


    # --------------------------------------------------------
    # Historical data BEFORE current match
    # --------------------------------------------------------

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

    venue_team_key = (
        batting_team,
        venue
    )

    venue_team_history = list(
        venue_team_scores[
            venue_team_key
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
    # BATTING FORM
    # ========================================================

    batting_last_3 = batting_history[-3:]
    batting_last_5 = batting_history[-5:]
    batting_last_10 = batting_history[-10:]


    batting_recent_score_3 = (
        np.mean(batting_last_3)
        if batting_last_3 else 0
    )

    batting_recent_score_5 = (
        np.mean(batting_last_5)
        if batting_last_5 else 0
    )

    batting_recent_score_10 = (
        np.mean(batting_last_10)
        if batting_last_10 else 0
    )

    batting_score_std_5 = (
        np.std(batting_last_5)
        if len(batting_last_5) >= 2
        else 0
    )

    batting_best_score_5 = (
        max(batting_last_5)
        if batting_last_5 else 0
    )

    batting_lowest_score_5 = (
        min(batting_last_5)
        if batting_last_5 else 0
    )

    batting_average_score = (
        np.mean(batting_history)
        if batting_history else 0
    )

    batting_recent_runs_5 = (
        np.sum(batting_last_5)
        if batting_last_5 else 0
    )


    # --------------------------------------------------------
    # Batting trend
    # --------------------------------------------------------

    if len(batting_history) >= 5:

        first_half = np.mean(
            batting_history[-5:-2]
        )

        second_half = np.mean(
            batting_history[-2:]
        )

        batting_form_trend = (
            second_half - first_half
        )

    else:

        batting_form_trend = 0


    # ========================================================
    # BATTING WICKETS
    # ========================================================

    batting_wickets_5 = (
        np.mean(
            batting_wicket_history[-5:]
        )
        if batting_wicket_history
        else 0
    )


    # ========================================================
    # BOWLING / OPPONENT DEFENSE
    # ========================================================

    bowling_recent_score_3 = (
        np.mean(
            bowling_history[-3:]
        )
        if bowling_history
        else 0
    )

    bowling_recent_score_5 = (
        np.mean(
            bowling_history[-5:]
        )
        if bowling_history
        else 0
    )

    bowling_recent_score_10 = (
        np.mean(
            bowling_history[-10:]
        )
        if bowling_history
        else 0
    )

    bowling_score_std_5 = (
        np.std(
            bowling_history[-5:]
        )
        if len(bowling_history[-5:]) >= 2
        else 0
    )

    bowling_recent_wickets_5 = (
        np.mean(
            bowling_wicket_history[-5:]
        )
        if bowling_wicket_history
        else 0
    )

    bowling_recent_wickets_10 = (
        np.mean(
            bowling_wicket_history[-10:]
        )
        if bowling_wicket_history
        else 0
    )

    bowling_average_wickets = (
        np.mean(
            bowling_wicket_history
        )
        if bowling_wicket_history
        else 0
    )


    # ========================================================
    # VENUE FEATURES
    # ========================================================

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

    venue_score_std = (
        np.std(
            venue_history
        )
        if len(venue_history) >= 2
        else 0
    )

    venue_high_score = (
        max(venue_history)
        if venue_history
        else 0
    )

    venue_low_score = (
        min(venue_history)
        if venue_history
        else 0
    )

    venue_matches = len(
        venue_history
    )


    # ========================================================
    # TEAM + VENUE HISTORY
    # ========================================================

    team_venue_average = (
        np.mean(
            venue_team_history
        )
        if venue_team_history
        else 0
    )

    team_venue_recent = (
        np.mean(
            venue_team_history[-5:]
        )
        if venue_team_history
        else 0
    )

    team_venue_matches = len(
        venue_team_history
    )


    # ========================================================
    # HEAD-TO-HEAD
    # ========================================================

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

    h2h_score_std = (
        np.std(h2h_history)
        if len(h2h_history) >= 2
        else 0
    )

    h2h_matches = len(
        h2h_history
    )


    # ========================================================
    # TOSS
    # ========================================================

    toss_winner_batting = int(
        match["toss_winner"] ==
        batting_team
    )

    toss_bat_decision = int(
        match["toss_decision"] ==
        "bat"
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

        "batting_recent_score_3":
            batting_recent_score_3,

        "batting_recent_score_5":
            batting_recent_score_5,

        "batting_recent_score_10":
            batting_recent_score_10,

        "batting_recent_runs_5":
            batting_recent_runs_5,

        "batting_average_score":
            batting_average_score,

        "batting_score_std_5":
            batting_score_std_5,

        "batting_best_score_5":
            batting_best_score_5,

        "batting_lowest_score_5":
            batting_lowest_score_5,

        "batting_form_trend":
            batting_form_trend,

        "batting_team_wickets_5":
            batting_wickets_5,

        "bowling_recent_score_3":
            bowling_recent_score_3,

        "bowling_recent_score_5":
            bowling_recent_score_5,

        "bowling_recent_score_10":
            bowling_recent_score_10,

        "bowling_score_std_5":
            bowling_score_std_5,

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

        "venue_score_std":
            venue_score_std,

        "venue_high_score":
            venue_high_score,

        "venue_low_score":
            venue_low_score,

        "venue_matches":
            venue_matches,

        "team_venue_average":
            team_venue_average,

        "team_venue_recent":
            team_venue_recent,

        "team_venue_matches":
            team_venue_matches,

        "h2h_average_score":
            h2h_average_score,

        "h2h_recent_score":
            h2h_recent_score,

        "h2h_score_std":
            h2h_score_std,

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


    # ========================================================
    # UPDATE HISTORY AFTER FEATURES
    # ========================================================

    current_score = match[
        "first_innings_score"
    ]

    current_wickets = int(
        match["first_innings_wickets"]
    )


    team_scores[
        batting_team
    ].append(
        current_score
    )

    team_wickets[
        bowling_team
    ].append(
        current_wickets
    )

    venue_scores[
        venue
    ].append(
        current_score
    )

    venue_team_scores[
        venue_team_key
    ].append(
        current_score
    )

    h2h_scores[
        h2h_key
    ].append(
        current_score
    )


    team_matches[
        team1
    ] += 1

    team_matches[
        team2
    ] += 1


# ============================================================
# DATAFRAME
# ============================================================

X = pd.DataFrame(
    feature_rows
)

y = pd.Series(
    targets,
    name="first_innings_score"
)


print("=" * 60)
print("IPL SCORE MODEL V3")
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


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# RANDOM FOREST
# ============================================================

rf = RandomForestRegressor(

    n_estimators=400,

    max_depth=10,

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


rf_predictions = rf.predict(
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
# GRADIENT BOOSTING
# ============================================================

gb = GradientBoostingRegressor(

    n_estimators=300,

    learning_rate=0.03,

    max_depth=3,

    min_samples_split=5,

    min_samples_leaf=3,

    random_state=42
)


gb.fit(
    X_train,
    y_train
)


gb_predictions = gb.predict(
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
    "\nRandom Forest"
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
    "\nGradient Boosting"
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
# SELECT MODEL
# ============================================================

if rf_mae <= gb_mae:

    best_model = rf

    best_name = "Random Forest"

    best_mae = rf_mae

else:

    best_model = gb

    best_name = "Gradient Boosting"

    best_mae = gb_mae


print(
    f"\nSelected model: {best_name}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        X.columns,

    "importance":
        best_model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print(
    "\n" + "=" * 60
)

print(
    "TOP 15 FEATURES"
)

print(
    "=" * 60
)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# RETRAIN BEST MODEL ON ALL DATA
# ============================================================

print(
    "\nRetraining best model on full dataset..."
)


if best_name == "Random Forest":

    final_model = RandomForestRegressor(

        n_estimators=400,

        max_depth=10,

        min_samples_split=5,

        min_samples_leaf=2,

        max_features="sqrt",

        random_state=42,

        n_jobs=-1
    )

else:

    final_model = GradientBoostingRegressor(

        n_estimators=300,

        learning_rate=0.03,

        max_depth=3,

        min_samples_split=5,

        min_samples_leaf=3,

        random_state=42
    )


final_model.fit(
    X,
    y
)


# ============================================================
# SAVE V3 MODEL
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
    "MODEL SAVED SUCCESSFULLY"
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
    f"\nFinal feature count: {len(X.columns)}"
)

print(
    "\nIMPORTANT:"
)

print(
    "This is a new V3 model."
)

print(
    "The old V1 model file was NOT modified."
)