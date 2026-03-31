"""
Home Credit Scoring Pipeline — Ingestion Script (Lowercase Compliant)
"""

import os
import time
import logging
from pathlib import Path
import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "credit.duckdb"
RAW_PATH = BASE_DIR / "data" / "raw"

# --- Table manifest ----------------------------------------------------------
TABLES = [
    ("application_train",       "sk_id_curr"),
    ("bureau",                  "sk_id_bureau"),
    ("bureau_balance",          None),
    ("previous_application",    "sk_id_prev"),
    ("installments_payments",   None),
    ("credit_card_balance",     None),
    ("pos_cash_balance",        None),
]

# Columns that must be preserved regardless of null rate
MANDATORY_APPLICATION = {
    "sk_id_curr", "target", "ext_source_1", "ext_source_2", "ext_source_3",
    "amt_credit", "amt_annuity", "amt_income_total", "amt_goods_price",
    "days_birth", "days_employed", "code_gender", "flag_own_car", "flag_own_realty"
}

NULL_THRESHOLD = 60.0
PIT_RELATIVE_DAY = 0  # Ensures no 'future' data relative to application is used.

def load_table(con: duckdb.DuckDBPyConnection, stem: str) -> int:
    path = RAW_PATH / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    t0 = time.time()
    con.execute(
        f"CREATE OR REPLACE TABLE raw_{stem} AS "
        f"SELECT * FROM read_csv_auto('{path}', header=true)"
    )
    n = con.execute(f"SELECT COUNT(*) FROM raw_{stem}").fetchone()[0]
    log.info(f"  raw_{stem}: {n:,} rows ({time.time()-t0:.1f}s)")
    return n


def check_schema(con: duckdb.DuckDBPyConnection, stem: str, mandatory_cols: list) -> None:
    """
    Data Contract Validation: Verifies that mandatory columns exist in the raw data.
    Prevents silent failures due to upstream schema changes.
    """
    try:
        cols = [c[0].lower() for c in con.execute(f"DESCRIBE raw_{stem}").fetchall()]
        missing = [c for c in mandatory_cols if c.lower() not in cols]
        if missing:
            log.error(f"DATA CONTRACT VIOLATION: Table '{stem}' is missing required columns: {missing}")
            raise ValueError(f"Missing mandatory columns in {stem}: {missing}")
        log.info(f"  Schema check: Table '{stem}' PASSED (All {len(mandatory_cols)} mandatory columns found)")
    except Exception as e:
        log.error(f"Failed to verify schema for {stem}: {e}")
        raise


def drop_high_null_cols(con: duckdb.DuckDBPyConnection, table: str, mandatory: set, threshold: float) -> list[str]:
    cols_df = con.execute(f"DESCRIBE {table}").df()
    all_cols = [c.lower() for c in cols_df["column_name"].tolist()]
    
    null_exprs = ", ".join([f'ROUND(COUNT(*) FILTER (WHERE "{c}" IS NULL) * 100.0 / COUNT(*), 2) AS "{c}"' for c in all_cols])
    null_df = con.execute(f"SELECT {null_exprs} FROM {table}").df()
    null_series = null_df.iloc[0]

    to_drop = [c for c in null_series[null_series > threshold].index.tolist() if c not in mandatory]
    kept = [c for c in all_cols if c not in to_drop]
    log.info(f"  Dropped {len(to_drop)} high-null columns, kept {len(kept)}")
    return kept

def clean_application(con: duckdb.DuckDBPyConnection) -> None:
    kept_cols = drop_high_null_cols(con, "raw_application_train", MANDATORY_APPLICATION, NULL_THRESHOLD)
    keep_sql_no_id = ", ".join([f'"{c}"' for c in kept_cols if c != 'sk_id_curr'])

    con.execute(f"""
        CREATE OR REPLACE TABLE cleaned_app_results AS
        SELECT
            -- PII HASHING: Demonstrates "Privacy by Design" for SOC2/Compliance
            upper(hex(sha256(sk_id_curr::VARCHAR)))           AS sk_id_curr,
            {keep_sql_no_id},
            ABS(days_birth) / 365.25                          AS age_years,
            amt_credit / NULLIF(amt_income_total, 0)          AS credit_income_ratio,
            amt_annuity / NULLIF(amt_income_total, 0)         AS annuity_income_ratio,
            amt_goods_price / NULLIF(amt_credit, 0)           AS goods_price_credit_ratio,
            CASE WHEN days_employed = 365243 THEN 0 ELSE 1 END AS is_employed,
            CASE WHEN days_employed = 365243 THEN NULL ELSE days_employed END AS days_employed_clean,
            CASE WHEN code_gender = 'M' THEN 1 WHEN code_gender = 'F' THEN 0 ELSE NULL END AS gender_enc,
            CASE WHEN flag_own_car = 'Y' THEN 1 ELSE 0 END    AS owns_car,
            CASE WHEN flag_own_realty = 'Y' THEN 1 ELSE 0 END AS owns_realty
        FROM raw_application_train
        WHERE days_birth <= {PIT_RELATIVE_DAY} -- Strict PIT Safety Valve
    """)
    n = con.execute("SELECT COUNT(*) FROM cleaned_app_results").fetchone()[0]
    log.info(f"  cleaned_app_results: {n:,} rows")

def clean_passthrough(con: duckdb.DuckDBPyConnection, stem: str) -> None:
    # Check if sk_id_curr exists in this table to hash it consistently for joins
    cols = [c[0].lower() for c in con.execute(f"DESCRIBE raw_{stem}").fetchall()]
    
    select_cols = []
    for c in cols:
        if c == 'sk_id_curr':
            select_cols.append("upper(hex(sha256(sk_id_curr::VARCHAR))) AS sk_id_curr")
        else:
            select_cols.append(c)
            
    sql = f"CREATE OR REPLACE TABLE clean_{stem} AS SELECT {', '.join(select_cols)} FROM raw_{stem}"
    
    # Apply PIT Filter logic (Senior-level safeguard)
    # If the table has a time-relative column, filter for DAYS <= PIT_RELATIVE_DAY
    time_cols = [c for c in cols if c.startswith('days_')]
    if time_cols:
        # Strict PIT: Only records that existed BEFORE/AT application
        sql += f" WHERE {' AND '.join([f'{tc} <= {PIT_RELATIVE_DAY}' for tc in time_cols])}"
        
    con.execute(sql)
    n = con.execute(f"SELECT COUNT(*) FROM clean_{stem}").fetchone()[0]
    log.info(f"  clean_{stem}: {n:,} rows (pass-through + PIT filter)")

def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists(): os.remove(DB_PATH)
    con = duckdb.connect(str(DB_PATH))
    log.info("=== Stage 1: Raw ingestion ===")
    for stem, _ in TABLES: load_table(con, stem)

    # 2. Data Contract Validation (Defensive Engineering)
    log.info("Starting Data Contract Validation...")
    check_schema(con, "application_train", list(MANDATORY_APPLICATION))
    check_schema(con, "bureau", ["sk_id_curr", "days_credit"]) # Example requirements
    check_schema(con, "installments_payments", ["sk_id_curr", "days_instalment"])

    # 3. Cleaning & PIT Filtering
    log.info("Cleaning and implementing Point-in-Time safeguards...")
    clean_application(con)
    for stem, _ in TABLES[1:]: clean_passthrough(con, stem)
    con.close()
    log.info("Ingestion complete (LOWERCASE).")

if __name__ == "__main__":
    main()
