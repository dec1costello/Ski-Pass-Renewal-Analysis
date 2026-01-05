"""
Predict ski pass renewal using the latest churn-prioritized LGBM model.
Scores all gold customers and stores predictions in DuckDB.
"""

from pathlib import Path
from datetime import datetime
import pickle
import logging
import sys

import duckdb
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "warehouse" / "ski.duckdb"

FEATURE_TABLE = "main.gold_customer_features"
ID_COLS = ["customer_key"]
TARGET = "renewal_in_subsequent_season"

MODEL_TABLE = "ml_models"
PREDICTION_TABLE = "ml_predictions"

STREAMLIT_SAMPLE_SIZE = 0.2  # fraction of customers for Streamlit demo

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
def ensure_prediction_table(con):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} (
            customer_key TEXT,
            predicted_renewal DOUBLE,
            predicted_churn DOUBLE,
            threshold DOUBLE,
            scored_at TIMESTAMP,
            streamlit_flag BOOLEAN
        )
    """)

# ------------------------------------------------------------------
# Load latest model
# ------------------------------------------------------------------
def load_latest_model(con):
    row = con.execute(f"""
        SELECT model_blob, threshold
        FROM {MODEL_TABLE}
        ORDER BY trained_at DESC
        LIMIT 1
    """).fetchone()

    if row is None:
        raise RuntimeError("No trained model found in DuckDB.")

    model_blob, threshold = row
    model = pickle.loads(model_blob)
    return model, threshold

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_features():
    con = duckdb.connect(DB_PATH.as_posix())
    df = con.execute(f"SELECT * FROM {FEATURE_TABLE}").df()
    con.close()
    return df

# ------------------------------------------------------------------
# Main prediction pipeline
# ------------------------------------------------------------------
def main():
    logger.info("Loading features and model...")

    con = duckdb.connect(DB_PATH.as_posix())
    ensure_prediction_table(con)

    df = load_features()
    model, threshold = load_latest_model(con)

    # Drop non-feature columns
    X = df.drop(columns=ID_COLS + [TARGET], errors="ignore")

    # ---- IMPORTANT FIX: safe class indexing ----
    renewal_idx = list(model.classes_).index(1)
    churn_proba = 1 - model.predict_proba(X)[:, renewal_idx]
    predicted_renewal = 1 - churn_proba

    # Reproducible Streamlit flag
    rng = np.random.default_rng(seed=42)
    streamlit_flag = rng.random(len(df)) < STREAMLIT_SAMPLE_SIZE

    # Build prediction dataframe
    pred_df = pd.DataFrame({
        "customer_key": df["customer_key"],
        "predicted_renewal": predicted_renewal,
        "predicted_churn": churn_proba,
        "threshold": threshold,
        "scored_at": datetime.now(),
        "streamlit_flag": streamlit_flag,
    })

    # Persist predictions (overwrite)
    con.execute(f"DELETE FROM {PREDICTION_TABLE}")
    con.register("pred_df", pred_df)
    con.execute(f"INSERT INTO {PREDICTION_TABLE} SELECT * FROM pred_df")

    con.close()
    logger.info(
        f"Predictions saved to {PREDICTION_TABLE}, rows: {len(pred_df)}"
    )

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
