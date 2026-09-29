import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# IPL MATCH ANALYZER - IMPROVED ML TRAINING
# ============================================================

print("=" * 60)
print("IPL MATCH ANALYZER - IMPROVED ML TRAINING")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load improved features
# ------------------------------------------------------------

data = pd.read_csv(
    "data/features/match_features_v2.csv"
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
# 3. Train/Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 4. Models
# ------------------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        random_state=42
    )
}

# ------------------------------------------------------------
# 5. Optional XGBoost
# ------------------------------------------------------------

try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    print("\nXGBoost available: YES")

except ImportError:

    print("\nXGBoost available: NO")
    print("Continuing without XGBoost.")

# ------------------------------------------------------------
# 6. Train and evaluate
# ------------------------------------------------------------

results = []

best_model = None
best_model_name = None
best_accuracy = -1

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions
    ))

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        target_names=[
            "Team 2 Win",
            "Team 1 Win"
        ],
        zero_division=0
    ))

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# ------------------------------------------------------------
# 7. Model comparison
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("IMPROVED MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("\nBest Model:", best_model_name)
print(
    f"Best Accuracy: {best_accuracy:.4f}"
)

# ------------------------------------------------------------
# 8. Save best model
# ------------------------------------------------------------

Path("model").mkdir(
    exist_ok=True
)

joblib.dump(
    best_model,
    "model/ipl_match_model_v2.pkl"
)

# Save exact feature order
joblib.dump(
    list(X.columns),
    "model/feature_columns_v2.pkl"
)

print("\nBest model saved:")
print("model/ipl_match_model_v2.pkl")

print("\nFeature columns saved:")
print("model/feature_columns_v2.pkl")

print("\n" + "=" * 60)
print("IMPROVED ML TRAINING COMPLETED")
print("=" * 60)