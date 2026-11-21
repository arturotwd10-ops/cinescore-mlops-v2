import json
import os
from datetime import datetime, timezone
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/champion_model.joblib"
METRICS_PATH = "models/metrics_summary.json"
RATING_THRESHOLD = 7.0

NUMERIC_COLS = ["release_year", "imdb_rating_count"]
CATEGORICAL_COLS = ["genre"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cine_score_mlops")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if "imdb_rating" in train_df.columns:
        train_df["high_rating"] = (pd.to_numeric(train_df["imdb_rating"], errors="coerce") > RATING_THRESHOLD).astype(int)
    if "imdb_rating" in test_df.columns:
        test_df["high_rating"] = (pd.to_numeric(test_df["imdb_rating"], errors="coerce") > RATING_THRESHOLD).astype(int)

    X_train = train_df[FEATURE_COLS].copy()
    X_train["imdb_rating_count"] = np.log1p(X_train["imdb_rating_count"].clip(lower=0))
    y_train = train_df["high_rating"]

    X_test = test_df[FEATURE_COLS].copy()
    X_test["imdb_rating_count"] = np.log1p(X_test["imdb_rating_count"].clip(lower=0))
    y_test = test_df["high_rating"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=180, max_depth=10, random_state=42),
        "svm_rbf": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
    }

    best_f1 = -1.0
    best_name = None
    best_pipe = None
    results = {}

    os.makedirs("models", exist_ok=True)

    for name, clf in models.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([
                ("prep", preprocessor),
                ("clf", clf),
            ])
            pipe.fit(X_train, y_train)

            preds = pipe.predict(X_test)
            probs = pipe.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, zero_division=0)
            auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5

            mlflow.log_param("model_name", name)
            mlflow.log_param("rating_threshold", f"> {RATING_THRESHOLD}")
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("f1_score", float(f1))
            mlflow.log_metric("roc_auc", float(auc))

            results[name] = {
                "accuracy": round(float(acc), 4),
                "f1_score": round(float(f1), 4),
                "roc_auc": round(float(auc), 4),
            }
            print(f"{name}: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")

            if f1 > best_f1:
                best_f1 = float(f1)
                best_name = name
                best_pipe = pipe

    joblib.dump(best_pipe, MODEL_PATH)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating_threshold": f"> {RATING_THRESHOLD}",
        "champion": best_name,
        "best_f1": round(best_f1, 4),
        "models": results,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Champion guardado en {MODEL_PATH}: {best_name} (umbral > {RATING_THRESHOLD})")


if __name__ == "__main__":
    main()
