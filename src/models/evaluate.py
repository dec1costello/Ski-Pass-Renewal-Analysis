"""
Evaluate the latest trained renewal model.
Computes diagnostics, confusion matrices, and SHAP importance.
"""

from pathlib import Path
import pickle
import duckdb
import pandas as pd
import numpy as np
import shap
import json
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "warehouse" / "ski.duckdb"

FEATURE_TABLE = "main.gold_customer_features"
MODEL_TABLE = "ml_models"

TARGET = "renewal_in_subsequent_season"
ID_COLS = ["customer_key"]

def load_model_and_data():
    con = duckdb.connect(DB_PATH.as_posix())

    model_blob, threshold = con.execute(f"""
        SELECT model_blob, threshold
        FROM {MODEL_TABLE}
        ORDER BY trained_at DESC
        LIMIT 1
    """).fetchone()

    df = con.execute(f"SELECT * FROM {FEATURE_TABLE}").df()
    con.close()

    model = pickle.loads(model_blob)
    return model, threshold, df

def main():
    model, threshold, df = load_model_and_data()

    X = df.drop(columns=ID_COLS + [TARGET], errors="ignore")
    y = df[TARGET].astype(int)

    renewal_idx = list(model.classes_).index(1)
    churn_proba = 1 - model.predict_proba(X)[:, renewal_idx]
    y_pred = (churn_proba >= threshold).astype(int)

    metrics = {
        "f2": fbeta_score(y, y_pred, beta=2, pos_label=0),
        "precision": precision_score(y, y_pred, pos_label=0),
        "recall": recall_score(y, y_pred, pos_label=0),
        "roc_auc": roc_auc_score(y, churn_proba),
        "avg_precision": average_precision_score(y == 0, churn_proba),
    }

    print(json.dumps(metrics, indent=2))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

if __name__ == "__main__":
    main()
