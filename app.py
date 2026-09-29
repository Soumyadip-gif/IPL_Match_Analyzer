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

        venue = data.get(
            "venue"
        )

        match_date = data.get(
            "match_date"
        )

        toss_winner = data.get(
            "toss_winner"
        )

        toss_decision = data.get(
            "toss_decision"
        )


        # ----------------------------------------------------
        # BASIC VALIDATION
        # ----------------------------------------------------

        if not team1 or not team2 or not venue:

            return jsonify({

                "status": "error",

                "error":
                    "Team 1, Team 2 and Venue "
                    "are required."

            }), 400


        if team1 == team2:

            return jsonify({

                "status": "error",

                "error":
                    "Team 1 and Team 2 "
                    "must be different."

            }), 400


        if team1 not in TEAMS:

            return jsonify({

                "status": "error",

                "error":
                    "Invalid Team 1 selected."

            }), 400


        if team2 not in TEAMS:

            return jsonify({

                "status": "error",

                "error":
                    "Invalid Team 2 selected."

            }), 400


        if venue not in VENUES:

            return jsonify({

                "status": "error",

                "error":
                    "Invalid venue selected."

            }), 400


        # ----------------------------------------------------
        # MATCH DATE
        # ----------------------------------------------------

        if not match_date:

            return jsonify({

                "status": "error",

                "error":
                    "Match date is required."

            }), 400


        try:

            parsed_match_date = pd.to_datetime(
                match_date,
                errors="raise"
            )


            match_date = (
                parsed_match_date
                .strftime("%Y-%m-%d")
            )


            # ------------------------------------------------
            # Season is automatically derived from date
            # ------------------------------------------------

            season = int(
                parsed_match_date.year
            )

        except Exception:

            return jsonify({

                "status": "error",

                "error":
                    "match_date must be a valid "
                    "date in YYYY-MM-DD format."

            }), 400


        # ----------------------------------------------------
        # TOSS VALIDATION
        # ----------------------------------------------------

        if toss_winner and toss_winner not in [
            team1,
            team2
        ]:

            return jsonify({

                "status": "error",

                "error":
                    "Toss winner must be "
                    "Team 1 or Team 2."

            }), 400


        if toss_decision and toss_decision not in [
            "bat",
            "field"
        ]:

            return jsonify({

                "status": "error",

                "error":
                    "Toss decision must be "
                    "'bat' or 'field'."

            }), 400


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

                "venue": venue,

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

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )