import duckdb
import pandas as pd
import numpy as np
from typing import List

# -----------------------------
# Configuration & Constants
# -----------------------------
DB_PATH = "warehouse/ski.duckdb"
GOLD_TABLE = "main.gold_customer_features"

ID_COL = "customer_key"
TARGET_COL = "renewal_in_subsequent_season"
CATEGORICAL_COLS = ["guest_state"]

NULL_WARN_THRESHOLD = 0.05
CORR_THRESHOLD = 0.05
OUTLIER_IQR_MULT = 1.5
EXTREME_IMBALANCE_PCT = 2.0


# -----------------------------
# Custom Exception
# -----------------------------
class DataAuditError(Exception):
    """Raised when data is unsafe for model training."""
    pass


# -----------------------------
# Utility Functions
# -----------------------------
def fetch_sample(con, n: int = 100) -> pd.DataFrame:
    return con.execute(f"SELECT * FROM {GOLD_TABLE} LIMIT {n}").df()


def total_rows(con) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {GOLD_TABLE}").fetchone()[0]


# -----------------------------
# Audit Checks
# -----------------------------
def schema_check(df: pd.DataFrame):
    required = {ID_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise DataAuditError(f"Missing required columns: {missing}")


def primary_key_check(con):
    dupes = con.execute(f"""
        SELECT COUNT({ID_COL}) - COUNT(DISTINCT {ID_COL})
        FROM {GOLD_TABLE}
    """).fetchone()[0]

    if dupes > 0:
        raise DataAuditError(f"{dupes} duplicate {ID_COL} values found")


def null_analysis(con, df):
    total = total_rows(con)
    null_counts = con.execute(f"""
        SELECT {", ".join(
            f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}"
            for c in df.columns
        )}
        FROM {GOLD_TABLE}
    """).df()

    records = []
    for col in df.columns:
        pct = null_counts[col][0] / total
        records.append({
            "column": col,
            "null_pct": round(pct * 100, 2),
            "status": "⚠️ High" if pct > NULL_WARN_THRESHOLD else "✅ OK"
        })

    out = pd.DataFrame(records)
    if out.loc[out["column"] == TARGET_COL, "null_pct"].iloc[0] > 10:
        raise DataAuditError(f"Target '{TARGET_COL}' has >10% nulls")

    return out


def target_distribution(con):
    dist = con.execute(f"""
        SELECT {TARGET_COL},
               COUNT(*) AS count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
        FROM {GOLD_TABLE}
        GROUP BY 1
        ORDER BY 1
    """).df()

    if dist["pct"].min() < EXTREME_IMBALANCE_PCT:
        raise DataAuditError("Extreme target imbalance detected")

    return dist


def join_coverage(con):
    return con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) AS missing_demographics,
            SUM(CASE WHEN total_rentals IS NULL THEN 1 ELSE 0 END) AS missing_purchases
        FROM {GOLD_TABLE}
    """).df()


def categorical_health(con, col: str):
    return con.execute(f"""
        SELECT {col},
               COUNT(*) AS count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
        FROM {GOLD_TABLE}
        GROUP BY 1
        ORDER BY count DESC
    """).df()


def numeric_feature_stats(con, numeric_cols: List[str]):
    rows = []
    total = total_rows(con)

    for col in numeric_cols:
        corr = con.execute(
            f"SELECT CORR({col}, {TARGET_COL}::FLOAT) FROM {GOLD_TABLE}"
        ).fetchone()[0] or 0.0

        q = con.execute(f"""
            SELECT
                approx_quantile({col}, 0.25) AS q1,
                approx_quantile({col}, 0.75) AS q3
            FROM {GOLD_TABLE}
        """).df()

        q1, q3 = q.iloc[0]
        iqr = q3 - q1
        lower = q1 - OUTLIER_IQR_MULT * iqr
        upper = q3 + OUTLIER_IQR_MULT * iqr

        outliers = con.execute(f"""
            SELECT COUNT(*)
            FROM {GOLD_TABLE}
            WHERE {col} < {lower} OR {col} > {upper}
        """).fetchone()[0]

        zero_count = con.execute(
            f"SELECT COUNT(*) FROM {GOLD_TABLE} WHERE {col} = 0"
        ).fetchone()[0]

        rows.append({
            "feature": col,
            "corr_with_target": round(corr, 4),
            "outlier_pct": round(outliers / total * 100, 2),
            "zero_pct": round(zero_count / total * 100, 2)
        })

    return pd.DataFrame(rows)


# -----------------------------
# Main Audit Runner
# -----------------------------
def run_data_audit():
    con = duckdb.connect(DB_PATH)
    print(f"\n--- 🚀 Data Audit: {GOLD_TABLE} ---\n")

    df_sample = fetch_sample(con)

    print("[1/7] Schema Check")
    schema_check(df_sample)
    print("✅ Schema OK\n")

    print("[2/7] Primary Key Check")
    primary_key_check(con)
    print("✅ Primary key unique\n")

    print("[3/7] Null Analysis")
    print(null_analysis(con, df_sample).to_string(index=False), "\n")

    print("[4/7] Target Distribution")
    print(target_distribution(con).to_string(index=False), "\n")

    print("[5/7] Join Coverage")
    print(join_coverage(con).to_string(index=False), "\n")

    print("[6/7] Categorical Health")
    for col in CATEGORICAL_COLS:
        cat = categorical_health(con, col)
        print(f"\n{col} (top 10):")
        print(cat.head(10).to_string(index=False))
        rare = cat[cat["pct"] < 1.0]
        if not rare.empty:
            print(f"⚠️ {len(rare)} rare categories (<1%)\n")

    print("[7/7] Numeric Feature Health")
    numeric_cols = [
        c for c in df_sample.select_dtypes(include=[np.number]).columns
        if c not in {ID_COL, TARGET_COL}
    ]
    num_stats = numeric_feature_stats(con, numeric_cols)
    print(num_stats.to_string(index=False))

    suspicious = num_stats[num_stats["corr_with_target"].abs() > 0.9]
    if not suspicious.empty:
        raise DataAuditError("Possible target leakage detected")

    con.close()
    print("\n--- ✅ Data Audit Passed ---\n")


# -----------------------------
# Execute
# -----------------------------
if __name__ == "__main__":
    run_data_audit()
