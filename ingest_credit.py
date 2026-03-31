"""
Home Credit Scoring Pipeline — Ingestion Script

Ingests 7 raw CSV tables into DuckDB, applies cleaning, and produces
cleaned tables ready for dbt transformation.

Usage:
    python src/ingest_credit.py

Inputs:  data/raw/*.csv  (7 files from Kaggle Home Credit competition)
Outputs: data/credit.duckdb with 7 raw + 7 clean tables
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
# (filename_stem, primary_key)
TABLES = [
    ("application_train",       "SK_ID_CURR"),
    ("bureau",                  "SK_ID_BUREAU"),
    ("bureau_balance",          None),           # no single PK
    ("previous_application",    "SK_ID_PREV"),
    ("installments_payments",   None),
    ("credit_card_balance",     None),
    ("POS_CASH_balance",        None),
]

# Columns that must be preserved regardless of null rate
MANDATORY_APPLICATION = {
    "SK_ID_CURR", "TARGET",
    "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE",
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "REG_CITY_NOT_WORK_CITY",
    "ORGANIZATION_TYPE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
}

NULL_THRESHOLD = 60.0  # % — looser than fraud pipeline (credit data is naturally sparser)


def load_table(con: duckdb.DuckDBPyConnection, stem: str) -> int:
    """Load a raw CSV into DuckDB. Returns row count."""
    path = RAW_PATH / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"Download from: https://www.kaggle.com/c/home-credit-default-risk/data"
        )
    t0 = time.time()
    con.execute(
        f"CREATE OR REPLACE TABLE raw_{stem} AS "
        f"SELECT * FROM read_csv_auto('{path}', header=true)"
    )
    n = con.execute(f"SELECT COUNT(*) FROM raw_{stem}").fetchone()[0]
    log.info(f"  raw_{stem}: {n:,} rows ({time.time()-t0:.1f}s)")
    return n


def drop_high_null_cols(
    con: duckdb.DuckDBPyConnection,
    table: str,
    mandatory: set,
    threshold: float,
) -> list[str]:
    """Return list of columns to keep (drop those above null threshold)."""
    cols_df = con.execute(f"DESCRIBE {table}").df()
    all_cols = cols_df["column_name"].tolist()

    null_exprs = ", ".join(
        [f'ROUND(COUNT(*) FILTER (WHERE "{c}" IS NULL) * 100.0 / COUNT(*), 2) AS "{c}"'
         for c in all_cols]
    )
    null_df = con.execute(f"SELECT {null_exprs} FROM {table}").df()
    null_series = null_df.iloc[0]

    to_drop = [
        c for c in null_series[null_series > threshold].index.tolist()
        if c not in mandatory
    ]
    kept = [c for c in all_cols if c not in to_drop]
    log.info(f"  Dropped {len(to_drop)} high-null columns, kept {len(kept)}")
    return kept


def clean_application(con: duckdb.DuckDBPyConnection) -> None:
    """Clean main application table — the core of the pipeline."""
    kept_cols = drop_high_null_cols(
        con, "raw_application_train", MANDATORY_APPLICATION, NULL_THRESHOLD
    )
    keep_sql = ", ".join([f'"{c}"' for c in kept_cols])

    con.execute(f"""
        CREATE OR REPLACE TABLE clean_application AS
        SELECT
            {keep_sql},

            -- Derived: loan-to-income ratio
            AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0)          AS credit_income_ratio,

            -- Derived: annuity-to-income ratio
            AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0)         AS annuity_income_ratio,

            -- Derived: employment vs age ratio (positive = currently employed)
            CASE
                WHEN DAYS_EMPLOYED > 0 THEN NULL  -- positive = unemployed anomaly
                ELSE ABS(DAYS_EMPLOYED) / NULLIF(ABS(DAYS_BIRTH), 0)
            END                                                AS employment_age_ratio,

            -- Derived: employment flag (DAYS_EMPLOYED = 365243 means unemployed)
            CASE WHEN DAYS_EMPLOYED = 365243 THEN 0 ELSE 1
            END                                                AS is_employed,

            -- Derived: clean employment days (null out the anomaly sentinel)
            CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL
                 ELSE DAYS_EMPLOYED
            END                                                AS days_employed_clean,

            -- Derived: age in years (DAYS_BIRTH is negative)
            ABS(DAYS_BIRTH) / 365.25                          AS age_years,

            -- Derived: goods price ratio
            AMT_GOODS_PRICE / NULLIF(AMT_CREDIT, 0)           AS goods_price_credit_ratio,

            -- Derived: gender encode
            CASE WHEN CODE_GENDER = 'M' THEN 1
                 WHEN CODE_GENDER = 'F' THEN 0
                 ELSE NULL
            END                                                AS gender_enc,

            -- Derived: own car flag
            CASE WHEN FLAG_OWN_CAR = 'Y' THEN 1 ELSE 0
            END                                                AS owns_car,

            -- Derived: own realty flag
            CASE WHEN FLAG_OWN_REALTY = 'Y' THEN 1 ELSE 0
            END                                                AS owns_realty

        FROM raw_application_train
    """)
    n = con.execute("SELECT COUNT(*) FROM clean_application").fetchone()[0]
    log.info(f"  clean_application: {n:,} rows")


def clean_passthrough(con: duckdb.DuckDBPyConnection, stem: str) -> None:
    """Pass supporting tables through with minimal transformation."""
    con.execute(
        f"CREATE OR REPLACE TABLE clean_{stem} AS SELECT * FROM raw_{stem}"
    )
    n = con.execute(f"SELECT COUNT(*) FROM clean_{stem}").fetchone()[0]
    log.info(f"  clean_{stem}: {n:,} rows (pass-through)")


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        os.remove(DB_PATH)

    log.info(f"Connecting to {DB_PATH}")
    con = duckdb.connect(str(DB_PATH))

    # --- Stage 1: Raw ingestion ----------------------------------------------
    log.info("=== Stage 1: Raw ingestion ===")
    for stem, _ in TABLES:
        load_table(con, stem)

    # --- Stage 2: Cleaning ---------------------------------------------------
    log.info("=== Stage 2: Cleaning ===")
    clean_application(con)
    for stem, _ in TABLES[1:]:  # everything except application_train
        clean_passthrough(con, stem)

    # --- Summary -------------------------------------------------------------
    log.info("=== Summary ===")
    for stem, _ in TABLES:
        n = con.execute(f"SELECT COUNT(*) FROM clean_{stem}").fetchone()[0]
        log.info(f"  clean_{stem}: {n:,} rows")

    con.close()
    log.info("Ingestion complete.")


if __name__ == "__main__":
    main()
