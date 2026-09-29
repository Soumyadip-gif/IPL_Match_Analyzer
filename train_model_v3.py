import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# IPL MATCH ANALYZER - FUTURE-READY ML TRAINING
# ============================================================

print("=" * 60)
print("IPL MATCH ANALYZER - FUTURE-READY ML TRAINING")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

data = pd.read_csv(
    "data/features/match_features_v3.csv"
)

print("\nDataset shape:", data.shape)


# ------------------------------------------------------------
# 2. Features and target
# ------------------------------------------------------------

X = data.drop("target", axis=1)
y = data["target"]

print("Features:", X.shape[1])
print("Target:", len(y))


# ------------------------------------------------------------
# 3. Chronological Train/Test split
# ------------------------------------------------------------

# IMPORTANT:
# match_features_v3.csv was generated chronologically.
# Therefore we do NOT randomly shuffle the data.
#
# Older matches -> training
# Newer matches -> future-like test set

split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print("\nChronological split:")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

if "season" in data.columns:
    train_seasons = data.iloc[:split_index]["season"]
    test_seasons = data.iloc[split_index:]["season"]

    print(
        "Training seasons:",
        int(train_seasons.min()),
        "to",
        int(train_seasons.max())
    )

    print(
        "Testing seasons :",
        int(test_seasons.min()),
        "to",
        int(test_seasons.max())
    )


# ------------------------------------------------------------
# 4. Time-aware cross-validation
# ------------------------------------------------------------

# Each validation fold only uses earlier data to predict later data.
time_cv = TimeSeriesSplit(
    n_splits=5
)

print(
    "\nTime-series CV folds:",
    time_cv.n_splits
)


# ------------------------------------------------------------
# 5. Base models
# ------------------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            C=0.5,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
}


# ------------------------------------------------------------
# 6. Evaluation function
# ------------------------------------------------------------

results = []


def evaluate_model(name, model):

    print("\n" + "=" * 60)
    print("Testing:", name)
    print("=" * 60)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Team 2 Win",
                "Team 1 Win"
            ],
            zero_division=0
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    return {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ------------------------------------------------------------
# 7. Train base models
# ------------------------------------------------------------

for name, model in models.items():

    evaluate_model(
        name,
        model
    )


# ============================================================
# 8. Random Forest tuning
# ============================================================

print("\n" + "=" * 60)
print("TUNING RANDOM FOREST - TIME AWARE")
print("=" * 60)

rf_params = {

    "n_estimators": [
        300,
        500,
        700
    ],

    "max_depth": [
        6,
        8,
        10,
        12,
        None
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_features": [
        "sqrt",
        "log2"
    ]
}


rf_search = RandomizedSearchCV(
    RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),

    param_distributions=rf_params,

    n_iter=12,

    scoring="f1",

    cv=time_cv,

    random_state=42,

    n_jobs=-1,

    verbose=1
)


rf_search.fit(
    X_train,
    y_train
)

print("\nBest Random Forest parameters:")
print(rf_search.best_params_)

print(
    f"Best Time-Series CV F1: "
    f"{rf_search.best_score_:.4f}"
)

rf_result = evaluate_model(
    "Tuned Random Forest",
    rf_search.best_estimator_
)


# ============================================================
# 9. Gradient Boosting tuning
# ============================================================

print("\n" + "=" * 60)
print("TUNING GRADIENT BOOSTING - TIME AWARE")
print("=" * 60)

gb_params = {

    "n_estimators": [
        100,
        200,
        300,
        400
    ],

    "learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1
    ],

    "max_depth": [
        2,
        3,
        4
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ]
}


gb_search = RandomizedSearchCV(
    GradientBoostingClassifier(
        random_state=42
    ),

    param_distributions=gb_params,

    n_iter=12,

    scoring="f1",

    cv=time_cv,

    random_state=42,

    n_jobs=-1,

    verbose=1
)


gb_search.fit(
    X_train,
    y_train
)

print("\nBest Gradient Boosting parameters:")
print(gb_search.best_params_)

print(
    f"Best Time-Series CV F1: "
    f"{gb_search.best_score_:.4f}"
)

gb_result = evaluate_model(
    "Tuned Gradient Boosting",
    gb_search.best_estimator_
)


# ============================================================
# 10. XGBoost tuning
# ============================================================

print("\n" + "=" * 60)
print("TUNING XGBOOST - TIME AWARE")
print("=" * 60)

xgb_params = {

    "n_estimators": [
        200,
        300,
        500
    ],

    "learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1
    ],

    "max_depth": [
        2,
        3,
        4,
        5
    ],

    "min_child_weight": [
        1,
        3,
        5
    ],

    "subsample": [
        0.7,
        0.8,
        0.9
    ],

    "colsample_bytree": [
        0.7,
        0.8,
        0.9
    ],

    "gamma": [
        0,
        0.1,
        0.2
    ]
}


xgb_search = RandomizedSearchCV(
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    ),

    param_distributions=xgb_params,

    n_iter=15,

    scoring="f1",

    cv=time_cv,

    random_state=42,

    n_jobs=-1,

    verbose=1
)


xgb_search.fit(
    X_train,
    y_train
)

print("\nBest XGBoost parameters:")
print(xgb_search.best_params_)

print(
    f"Best Time-Series CV F1: "
    f"{xgb_search.best_score_:.4f}"
)

xgb_result = evaluate_model(
    "Tuned XGBoost",
    xgb_search.best_estimator_
)


# ============================================================
# 11. Select model using TIME-SERIES CV
# ============================================================

# Model selection is based on validation performance from the
# historical training period, not on the final test set.

tuned_models = {
    "Tuned Random Forest": {
        "model": rf_search.best_estimator_,
        "cv_f1": rf_search.best_score_
    },

    "Tuned Gradient Boosting": {
        "model": gb_search.best_estimator_,
        "cv_f1": gb_search.best_score_
    },

    "Tuned XGBoost": {
        "model": xgb_search.best_estimator_,
        "cv_f1": xgb_search.best_score_
    }
}

selected_name = max(
    tuned_models,
    key=lambda name: tuned_models[name]["cv_f1"]
)

selected_model = tuned_models[selected_name]["model"]
selected_cv_f1 = tuned_models[selected_name]["cv_f1"]


# Get final test metrics for the selected model.
selected_test_row = next(
    row for row in results
    if row["Model"] == selected_name
)


# ------------------------------------------------------------
# 12. Final comparison
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "F1 Score",
    ascending=False
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSelected Model:", selected_name)

print(
    f"Time-Series CV F1: "
    f"{selected_cv_f1:.4f}"
)

print(
    f"Future-like Test Accuracy: "
    f"{selected_test_row['Accuracy']:.4f}"
)

print(
    f"Future-like Test F1: "
    f"{selected_test_row['F1 Score']:.4f}"
)


# ============================================================
# 13. Retrain selected model on ALL historical data
# ============================================================

print("\n" + "=" * 60)
print("FINAL TRAINING ON ALL AVAILABLE HISTORICAL DATA")
print("=" * 60)

# After unbiased time-based evaluation, use all known historical
# matches to train the deployed model for future IPL matches.

selected_model.fit(
    X,
    y
)


# ============================================================
# 14. Save future-ready model
# ============================================================

Path("model").mkdir(
    exist_ok=True
)

joblib.dump(
    selected_model,
    "model/ipl_match_model_v3.pkl"
)

joblib.dump(
    list(X.columns),
    "model/feature_columns_v3.pkl"
)

training_metadata = {

    "training_type": "chronological_time_series",

    "total_samples": int(len(data)),

    "evaluation_training_samples": int(len(X_train)),

    "evaluation_test_samples": int(len(X_test)),

    "feature_count": int(X.shape[1]),

    "selected_model": str(selected_name),

    "time_series_cv_f1": float(selected_cv_f1),

    "future_like_test_accuracy": float(
        selected_test_row["Accuracy"]
    ),

    "future_like_test_f1": float(
        selected_test_row["F1 Score"]
    ),

    "final_training": "all_available_historical_data"
}

joblib.dump(
    training_metadata,
    "model/training_metadata_v3.pkl"
)


print("\nFuture-ready model saved:")
print(
    "model/ipl_match_model_v3.pkl"
)

print("\nFeature columns saved:")
print(
    "model/feature_columns_v3.pkl"
)

print("\nTraining metadata saved:")
print(
    "model/training_metadata_v3.pkl"
)

print("\n" + "=" * 60)
print("FUTURE-READY ML TRAINING COMPLETED")
print("=" * 60)
