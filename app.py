from flask import Flask, request, jsonify, render_template

import pandas as pd
import joblib

from prediction_engine import IPLPredictionEngine


app = Flask(__name__)


# ============================================================
# LOAD WIN PREDICTION MODEL
# ============================================================

MODEL_PATH = "model/ipl_match_model_v3.pkl"
FEATURE_PATH = "model/feature_columns_v3.pkl"

model = joblib.load(
    MODEL_PATH
)

feature_columns = joblib.load(
    FEATURE_PATH
)


# ============================================================
# LOAD SCORE PREDICTION MODEL
# ============================================================

SCORE_MODEL_PATH = "model/ipl_score_model.pkl"
SCORE_FEATURE_PATH = "model/score_feature_columns.pkl"

score_model = joblib.load(
    SCORE_MODEL_PATH
)

score_feature_columns = joblib.load(
    SCORE_FEATURE_PATH
)


# ============================================================
# LOAD CLEANED MATCH DATA
# ============================================================

MATCHES_PATH = "data/clean/matches.csv"
DELIVERIES_PATH = "data/clean/deliveries.csv"

matches = pd.read_csv(
    MATCHES_PATH
)


# ============================================================
# LOAD PREDICTION ENGINE
# ============================================================

prediction_engine = IPLPredictionEngine(
    MATCHES_PATH,
    DELIVERIES_PATH
)


# ============================================================
# BASIC DATA
# ============================================================

TEAMS = sorted(
    set(
        matches["team1"]
        .dropna()
        .unique()
    )
    |
    set(
        matches["team2"]
        .dropna()
        .unique()
    )
)


VENUES = sorted(
    matches["venue"]
    .dropna()
    .unique()
)


SEASONS = sorted(
    matches["season"]
    .dropna()
    .unique()
)


HISTORICAL_SEASONS = [
    int(x)
    for x in SEASONS
]


MAX_HISTORICAL_SEASON = max(
    HISTORICAL_SEASONS
)


# ============================================================
# SCORE PREDICTION
# ============================================================

def predict_first_innings_score(score_features):

    """
    Predict first innings score using the
    exact feature structure used during training.
    """

    # --------------------------------------------------------
    # Convert features to DataFrame
    # --------------------------------------------------------

    X_score = pd.DataFrame(
        [score_features]
    )


    # --------------------------------------------------------
    # Ensure exact training feature order
    # --------------------------------------------------------

    X_score = X_score.reindex(
        columns=score_feature_columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_score = (
        score_model
        .predict(X_score)[0]
    )


    predicted_score = round(
        float(predicted_score)
    )


    # --------------------------------------------------------
    # Keep prediction within realistic IPL range
    # --------------------------------------------------------

    predicted_score = max(
        80,
        min(
            predicted_score,
            280
        )
    )


    # --------------------------------------------------------
    # Estimated score range
    # --------------------------------------------------------

    lower_score = max(
        80,
        predicted_score - 15
    )


    upper_score = min(
        280,
        predicted_score + 15
    )


    return {

        "expected_first_innings_score":
            predicted_score,

        "estimated_score_range":
            f"{lower_score}-{upper_score}"

    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# OPTIONS
# ============================================================

@app.route(
    "/api/options",
    methods=["GET"]
)
def get_options():

    return jsonify({

        "teams": TEAMS,

        "venues": VENUES

    })


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # REQUEST DATA
        # ----------------------------------------------------

        data = request.get_json()


        if not data:

            return jsonify({

                "status": "error",

                "error":
                    "No JSON data received."

            }), 400


        team1 = data.get(
            "team1"
        )

        team2 = data.get(
            "team2"
        )

        # Team-only prediction defaults
        venue = "__NEUTRAL__"
        match_date = None
        toss_winner = None
        toss_decision = None

        if not team1 or not team2:
            return jsonify({
                "status": "error",
                "error": "Team 1 and Team 2 are required."
            }), 400

        if team1 == team2:
            return jsonify({
                "status": "error",
                "error": "Team 1 and Team 2 must be different."
            }), 400

        if team1 not in TEAMS:
            return jsonify({
                "status": "error",
                "error": "Invalid Team 1 selected."
            }), 400

        if team2 not in TEAMS:
            return jsonify({
                "status": "error",
                "error": "Invalid Team 2 selected."
            }), 400

        # Use the next season after the latest season in the dataset.
        season = int(matches["season"].max()) + 1

        # ====================================================
        # WIN PREDICTION FEATURES
        # ====================================================

        input_data = (
            prediction_engine.generate_features(

                team1=team1,

                team2=team2,

                venue=venue,

                season=season,

                toss_winner=toss_winner,

                toss_decision=toss_decision,

                match_date=match_date

            )
        )


        # ----------------------------------------------------
        # WIN MODEL FEATURES
        # ----------------------------------------------------

        X_input = pd.DataFrame(
            [input_data]
        )


        X_input = X_input.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # ====================================================
        # WIN PREDICTION
        # ====================================================

        prediction = model.predict(
            X_input
        )[0]


        probabilities = model.predict_proba(
            X_input
        )[0]


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        team1_probability = float(
            probabilities[1]
        )

        team2_probability = float(
            probabilities[0]
        )


        # ----------------------------------------------------
        # WINNER
        # ----------------------------------------------------

        predicted_team = (

            team1
            if prediction == 1
            else team2

        )


        # ====================================================
        # SCORE PREDICTION FEATURES
        # ====================================================

        score_features = (
            prediction_engine
            .generate_score_features(

                team1=team1,

                team2=team2,

                venue=venue,

                season=season,

                toss_winner=toss_winner,

                toss_decision=toss_decision,

                match_date=match_date

            )
        )


        # ====================================================
        # SCORE PREDICTION
        # ====================================================

        score_prediction = (
            predict_first_innings_score(
                score_features
            )
        )


        # ====================================================
        # MATCH ANALYSIS
        # ====================================================

        analysis = (
            prediction_engine
            .get_match_analysis(

                team1=team1,

                team2=team2,

                venue=venue,

                season=season,

                match_date=match_date

            )
        )


        # ====================================================
        # MODEL CONFIDENCE
        # ====================================================

        model_confidence = max(
            team1_probability,
            team2_probability
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({

            "status": "success",

            "match": {

                "team1": team1,

                "team2": team2,

                "venue": "Not specified (neutral baseline)",

                "season": season,

                "match_date": match_date,

                "toss_winner": toss_winner,

                "toss_decision": toss_decision

            },

            "prediction": {

                "winner": predicted_team,

                "team1_win_probability":
                    round(
                        team1_probability * 100,
                        2
                    ),

                "team2_win_probability":
                    round(
                        team2_probability * 100,
                        2
                    ),

                "model_confidence":
                    round(
                        model_confidence * 100,
                        2
                    ),

                "expected_first_innings_score":
                    score_prediction[
                        "expected_first_innings_score"
                    ],

                "estimated_score_range":
                    score_prediction[
                        "estimated_score_range"
                    ]

            },

            "analysis": analysis

        })


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print(
            "Prediction Error:",
            str(e)
        )

        return jsonify({

            "status": "error",

            "error": str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================


# ============================================================
# IPL ANALYTICS API
# ============================================================

def get_team_record(team, dataset):
    appearances = dataset[
        (dataset["team1"] == team) |
        (dataset["team2"] == team)
    ]

    decided = appearances[appearances["winner"].notna()]
    wins = int((decided["winner"] == team).sum())
    losses = len(decided) - wins

    return {
        "team": team,
        "matches": int(len(appearances)),
        "wins": wins,
        "losses": losses,
        "win_percentage": round(
            wins / len(decided) * 100, 2
        ) if len(decided) else 0
    }


@app.route("/api/analytics/overview")
def analytics_overview():
    decided = matches[matches["winner"].notna()]

    return jsonify({
        "total_matches": int(len(matches)),
        "decided_matches": int(len(decided)),
        "tied_matches": int(
            (matches["result_type"] == "tie").sum()
        ),
        "no_result_matches": int(
            (matches["result_type"] == "no result").sum()
        ),
        "total_teams": len(TEAMS),
        "total_seasons": len(SEASONS),
        "seasons": HISTORICAL_SEASONS,
        "latest_season": MAX_HISTORICAL_SEASON
    })


@app.route("/api/analytics/teams")
def analytics_teams():
    records = [
        get_team_record(team, matches)
        for team in TEAMS
    ]

    records.sort(
        key=lambda item: (
            item["wins"],
            item["win_percentage"]
        ),
        reverse=True
    )

    return jsonify({"teams": records})


@app.route("/api/analytics/head-to-head")
def analytics_head_to_head():
    team1 = request.args.get("team1")
    team2 = request.args.get("team2")

    if team1 not in TEAMS or team2 not in TEAMS or team1 == team2:
        return jsonify({
            "error": "Select two different valid IPL teams."
        }), 400

    subset = matches[
        (
            (matches["team1"] == team1) &
            (matches["team2"] == team2)
        ) |
        (
            (matches["team1"] == team2) &
            (matches["team2"] == team1)
        )
    ].copy()

    decided = subset[subset["winner"].notna()]
    team1_wins = int((decided["winner"] == team1).sum())
    team2_wins = int((decided["winner"] == team2).sum())

    recent = subset.sort_values(
        "date", ascending=False, na_position="last"
    ).head(10)

    recent_matches = []
    for _, row in recent.iterrows():
        recent_matches.append({
            "date": str(row["date"]),
            "season": int(row["season"]) if pd.notna(row["season"]) else None,
            "team1": row["team1"],
            "team2": row["team2"],
            "winner": row["winner"] if pd.notna(row["winner"]) else None,
            "result_type": row["result_type"]
        })

    return jsonify({
        "team1": team1,
        "team2": team2,
        "total_matches": int(len(subset)),
        "decided_matches": int(len(decided)),
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "tied_matches": int(
            (subset["result_type"] == "tie").sum()
        ),
        "no_result_matches": int(
            (subset["result_type"] == "no result").sum()
        ),
        "recent_matches": recent_matches
    })


@app.route("/api/analytics/season")
def analytics_season():
    year = request.args.get("year", type=int)

    if year not in HISTORICAL_SEASONS:
        return jsonify({
            "error": "Select a valid IPL season."
        }), 400

    season_matches = matches[
        matches["season"] == year
    ].copy()

    standings = []

    season_teams = sorted(
        set(season_matches["team1"].dropna()) |
        set(season_matches["team2"].dropna())
    )

    for team in season_teams:
        played = season_matches[
            (season_matches["team1"] == team) |
            (season_matches["team2"] == team)
        ]

        wins = int((played["winner"] == team).sum())
        decided = played[played["winner"].notna()]
        losses = int(len(decided) - wins)

        ties = int(
            (played["result_type"] == "tie").sum()
        )
        no_results = int(
            (played["result_type"] == "no result").sum()
        )

        standings.append({
            "team": team,
            "played": int(len(played)),
            "won": wins,
            "lost": losses,
            "tied": ties,
            "no_result": no_results,
            "points": wins * 2 + ties + no_results
        })

    standings.sort(
        key=lambda item: (
            item["points"],
            item["won"]
        ),
        reverse=True
    )

    return jsonify({
        "season": year,
        "total_matches": int(len(season_matches)),
        "decided_matches": int(
            season_matches["winner"].notna().sum()
        ),
        "standings": standings
    })


if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )