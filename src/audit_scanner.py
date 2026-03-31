"""
Home Credit Pipeline — 48h Audit Hook
Automated Technical Leakage Scanner

This script scans the engineered feature set for signs of "Technical Leakage"
(AUC inflation due to accidental inclusion of labels or future data).
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --- Config ------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "credit.duckdb"

# Leakage Thresholds
CORR_THRESHOLD = 0.95  # Pearson correlation > 0.95 is highly suspicious
ID_COL         = "sk_id_curr"
TARGET_COL     = "target"

def run_audit():
    if not DB_PATH.exists():
        log.error(f"Database not found at {DB_PATH}. Run ingestion first.")
        return

    log.info("Starting Pipeline Audit Scanner...")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    
    # Load feature sample
    try:
        df = con.execute("SELECT * FROM credit_features LIMIT 100000").df()
        con.close()
    except Exception as e:
        log.error(f"Could not load features: {e}")
        return

    log.info(f"Auditing {df.shape[1]} features across {df.shape[0]:,} rows.")

    leakage_found = False
    
    # 1. Target Correlation Check
    log.info("Checking for Target Leakage (High Correlation)...")
    numeric_df = df.select_dtypes(include=['number'])
    correlations = numeric_df.corr()[TARGET_COL].abs().sort_values(ascending=False)
    
    # Drop the target itself
    correlations = correlations.drop(labels=[TARGET_COL])
    
    suspicious = correlations[correlations > CORR_THRESHOLD]
    if not suspicious.empty:
        leakage_found = True
        log.warning(f"CRITICAL: Found {len(suspicious)} suspicious features with correlation > {CORR_THRESHOLD}:")
        for feat, val in suspicious.items():
            log.warning(f"  - {feat}: correlation={val:.4f}")
    else:
        log.info("  Pass: No features exceed the correlation threshold.")

    # 2. ID Predictive Power Check
    log.info("Checking for ID-based Leakage...")
    id_corr = abs(numeric_df[ID_COL].corr(numeric_df[TARGET_COL]))
    if id_corr > 0.05:
        leakage_found = True
        log.warning(f"WARNING: ID column '{ID_COL}' has correlation {id_corr:.4f} with target.")
        log.info("  This may indicate a sequence-based leak or temporal ordering in IDs.")
    else:
        log.info(f"  Pass: ID column correlation ({id_corr:.4f}) is within safe bounds.")

    # 3. Null Flag Leakage Check
    log.info("Checking for Null-Imputation Leakage...")
    null_leakage = []
    for col in df.columns:
        if col in [TARGET_COL, ID_COL]: continue
        null_flags = df[col].isna().astype(int)
        if null_flags.std() > 0:
            n_corr = abs(null_flags.corr(df[TARGET_COL]))
            if n_corr > CORR_THRESHOLD:
                null_leakage.append((col, n_corr))
    
    if null_leakage:
        leakage_found = True
        log.warning(f"CRITICAL: Found {len(null_leakage)} columns where the 'IS_NULL' state leaks the target:")
        for col, val in null_leakage:
            log.warning(f"  - {col}: null_correlation={val:.4f}")
    else:
        log.info("  Pass: No null-imputation leakage detected.")

    if not leakage_found:
        log.info("AUDIT COMPLETE: No technical leakage detected. Pipeline is healthy and AUC is valid.")
    else:
        log.warning("AUDIT COMPLETE: Potential leakage detected. Review the flagged features before deployment.")

if __name__ == "__main__":
    run_audit()
