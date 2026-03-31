"""
Model Performance Validator — Alerting Script

Usage:
    python src/validate_performance.py

Checks the saved model against a performance benchmark (AUC >= 0.75).
If performance is below the threshold, it exits with code 1 (failing CI).
"""

import logging
import pickle
import sys
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- Config ------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
DB_PATH     = ROOT / "data" / "credit.duckdb"
MODEL_PATH  = ROOT / "models" / "lgbm_credit_v1.pkl"
AUC_THRESHOLD = 0.75  # Benchmark for Home Credit default risk

def validate():
    if not MODEL_PATH.exists():
        log.error(f"Model not found at {MODEL_PATH}. Run training first.")
        sys.exit(1)

    if not DB_PATH.exists():
        log.error(f"Database not found at {DB_PATH}. Run ingestion and dbt first.")
        sys.exit(1)

    log.info(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    
    model     = artifact["model"]
    feat_cols = artifact["feature_cols"]
    encoders  = artifact["encoders"]

    log.info(f"Connecting to {DB_PATH}")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    
    # Load a sample for validation (using full data if possible, or just a large chunk)
    try:
        df = con.execute("SELECT * FROM credit_features LIMIT 50000").df()
        con.close()
    except Exception as e:
        log.error(f"Could not load features from database: {e}")
        sys.exit(1)

    log.info(f"Validating on {len(df):,} rows")
    
    # Preprocess
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].fillna("MISSING").astype(str)
            df[col] = le.transform(df[col])

    X = df[feat_cols]
    y = df["target"]

    # Evaluate
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)

    log.info(f"Current AUC: {auc:.4f}")
    log.info(f"Benchmark:   {AUC_THRESHOLD:.4f}")

    if auc < AUC_THRESHOLD:
        log.critical(f"PERFORMANCE ALERT: AUC {auc:.4f} is below threshold {AUC_THRESHOLD:.4f}!")
        # In a real production system, this could trigger a Slack alert or PagerDuty event.
        sys.exit(1)
    else:
        log.info("Performance validation passed. Model is healthy.")
        sys.exit(0)

if __name__ == "__main__":
    validate()
