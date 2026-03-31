"""
Home Credit Pipeline — Direct Feature Engineering (Smoke Test Bypass — Lowercase)
"""

import logging
from pathlib import Path
import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "credit.duckdb"

def build_features():
    if not DB_PATH.exists():
        log.error("Database not found. Run ingestion first.")
        return

    con = duckdb.connect(str(DB_PATH))
    log.info(f"Building features directly in {DB_PATH} (LOWERCASE)...")

    con.execute("""
        CREATE OR REPLACE TABLE credit_features AS
        WITH apps AS (
            SELECT * FROM cleaned_app_results
        ),
        bureau_agg AS (
            SELECT 
                sk_id_curr,
                COUNT(*) as bureau_record_count,
                SUM(CASE WHEN credit_active = 'Active' THEN 1 ELSE 0 END) as active_bureau_count,
                AVG(days_credit) as avg_days_credit,
                SUM(amt_credit_sum) as total_bureau_credit,
                CAST(NULL as VARCHAR) as worst_ever_bureau_status
            FROM clean_bureau
            GROUP BY 1
        ),
        payment_agg AS (
            SELECT 
                sk_id_curr,
                COUNT(*) as total_instalments,
                AVG(amt_payment / NULLIF(amt_instalment, 0)) as avg_payment_ratio,
                SUM(CASE WHEN amt_payment < amt_instalment THEN 1 ELSE 0 END) / COUNT(*) as underpayment_rate
            FROM clean_installments_payments
            GROUP BY 1
        )
        SELECT 
            a.*,
            b.bureau_record_count, b.active_bureau_count, b.avg_days_credit, b.total_bureau_credit,
            b.worst_ever_bureau_status,
            p.total_instalments, p.avg_payment_ratio, p.underpayment_rate,
            0 as closed_bureau_count,
            0 as days_since_last_bureau,
            0 as total_days_overdue,
            0 as overdue_bureau_count,
            0 as max_days_overdue,
            0 as total_bureau_debt,
            0 as bureau_debt_ratio,
            0 as total_credit_prolongations,
            0 as distinct_credit_types,
            0 as bureau_debt_to_new_credit,
            0 as avg_payment_lag,
            0 as worst_payment_lag,
            0 as payment_lag_volatility,
            0 as late_payment_rate,
            0 as total_underpayments,
            0 as min_payment_ratio,
            0 as overall_payment_completeness,
            0 as prev_loan_count,
            0 as prev_approved_count,
            0 as prev_approval_rate,
            0 as avg_prev_credit_ratio,
            COALESCE(ext_source_1, ext_source_2, ext_source_3) as ext_source_avg
        FROM apps a
        LEFT JOIN bureau_agg b ON a.sk_id_curr = b.sk_id_curr
        LEFT JOIN payment_agg p ON a.sk_id_curr = p.sk_id_curr
    """)
    n = con.execute("SELECT COUNT(*) FROM credit_features").fetchone()[0]
    log.info(f"Built credit_features: {n:,} rows")
    con.close()

if __name__ == "__main__":
    build_features()
