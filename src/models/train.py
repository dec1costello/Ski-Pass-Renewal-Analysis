"""
Train a churn-prioritized ski pass renewal model using dbt gold features.
Hyperparameter tuning via Optuna.
Models and metrics are stored directly in DuckDB.
"""

from pathlib import Path
from datetime import datetime
import json
import pickle
import logging
import sys
import warnings

import duckdb
import pandas as pd
import numpy as np
import optuna

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    fbeta_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "warehouse" / "ski.duckdb"

FEATURE_TABLE = "main.gold_customer_features"
TARGET = "renewal_in_subsequent_season"
ID_COLS = ["customer_key"]

MODEL_NAME = "lgbm_churn_first_optuna"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_TRIALS = 40

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# DuckDB setup
# ------------------------------------------------------------------
def ensure_model_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS ml_models (
            model_id TEXT,
            model_name TEXT,
            trained_at TIMESTAMP,
            threshold DOUBLE,
            f2_score DOUBLE,
            metrics JSON,
            hyperparameters JSON,
            model_blob BLOB
        )
    """)

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_data():
    con = duckdb.connect(DB_PATH.as_posix())
    df = con.execute(f"SELECT * FROM {FEATURE_TABLE}").df()
    con.close()

    df[TARGET] = df[TARGET].astype(int)

    X = df.drop(columns=ID_COLS + [TARGET])
    y = df[TARGET]

    return X, y

# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

# ------------------------------------------------------------------
# Threshold + metrics
# ------------------------------------------------------------------
def find_optimal_threshold(y_true, churn_proba):
    precision, recall, thresholds = precision_recall_curve(y_true == 0, churn_proba)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-9)
    return thresholds[np.argmax(f2)]

def compute_metrics(model, X, y, threshold):
    renewal_idx = list(model.classes_).index(1)
    churn_proba = 1 - model.predict_proba(X)[:, renewal_idx]
    y_pred = (churn_proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "churn_recall": recall_score(y, y_pred, pos_label=0),
        "churn_precision": precision_score(y, y_pred, pos_label=0),
        "roc_auc": roc_auc_score(y, churn_proba),
        "avg_precision": average_precision_score(y == 0, churn_proba),
        "f2_score": fbeta_score(y, y_pred, beta=2, pos_label=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

# ------------------------------------------------------------------
# Optuna objective
# ------------------------------------------------------------------
def objective(trial, X_train, X_valid, y_train, y_valid, preprocessor, churn_weight):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": churn_weight,
        "objective": "binary",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LGBMClassifier(**params)),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    renewal_idx = list(model.classes_).index(1)
    churn_proba = 1 - model.predict_proba(X_valid)[:, renewal_idx]
    threshold = find_optimal_threshold(y_valid, churn_proba)

    y_pred = (churn_proba >= threshold).astype(int)
    return fbeta_score(y_valid, y_pred, beta=2, pos_label=0)

# ------------------------------------------------------------------
# Training pipeline
# ------------------------------------------------------------------
def main():
    logger.info("Starting Optuna training")

    X, y = load_data()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        stratify=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    churn_weight = (y_train == 1).sum() / (y_train == 0).sum()
    preprocessor = build_preprocessor(X_train)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    study.optimize(
        lambda t: objective(
            t, X_train, X_valid, y_train, y_valid, preprocessor, churn_weight
        ),
        n_trials=N_TRIALS,
    )

    logger.info(f"Best F2: {study.best_value:.3f}")

    best_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                LGBMClassifier(
                    **study.best_params,
                    objective="binary",
                    scale_pos_weight=churn_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )

    best_model.fit(X_train, y_train)

    renewal_idx = list(best_model.classes_).index(1)
    churn_proba = 1 - best_model.predict_proba(X_valid)[:, renewal_idx]
    threshold = find_optimal_threshold(y_valid, churn_proba)
    metrics = compute_metrics(best_model, X_valid, y_valid, threshold)

    # ------------------------------------------------------------------
    # Persist model
    # ------------------------------------------------------------------
    con = duckdb.connect(DB_PATH.as_posix())
    ensure_model_table(con)

    model_id = f"{MODEL_NAME}_{datetime.now():%Y%m%d_%H%M%S}"

    con.execute(
        """
        INSERT INTO ml_models
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            model_id,
            MODEL_NAME,
            datetime.now(),
            threshold,
            metrics["f2_score"],
            json.dumps(metrics),
            json.dumps(study.best_params),
            pickle.dumps(best_model),
        ],
    )

    con.close()
    logger.info(f"Model stored in DuckDB: {model_id}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
