-- credit_bureau_features: aggregated bureau history per applicant
-- One row per SK_ID_CURR
-- These are among the highest-signal features in credit scoring

{{
  config(materialized='table')
}}

WITH bureau AS (
    SELECT * FROM {{ ref('stg_bureau') }}
),

bureau_balance AS (
    SELECT
        sk_id_bureau,
        -- Worst status ever seen (C=closed, X=unknown, 0=no DPD, 1-5=DPD buckets)
        MAX(STATUS) AS worst_bureau_status,
        COUNT(*)    AS bureau_balance_months
    FROM {{ source('credit_db', 'clean_bureau_balance') }}
    GROUP BY sk_id_bureau
)

SELECT
    b.sk_id_curr,

    -- Volume of bureau records
    COUNT(*)                                        AS bureau_record_count,
    COUNT(*) FILTER (WHERE b.credit_active = 'Active')  AS active_bureau_count,
    COUNT(*) FILTER (WHERE b.credit_active = 'Closed')  AS closed_bureau_count,

    -- Recency of bureau inquiries
    MAX(b.days_credit)                              AS days_since_last_bureau,
    MIN(b.days_credit)                              AS days_since_oldest_bureau,
    AVG(b.days_credit)                              AS avg_days_credit,

    -- Overdue signals
    SUM(b.credit_day_overdue)                       AS total_days_overdue,
    COUNT(*) FILTER (WHERE b.credit_day_overdue > 0) AS overdue_bureau_count,
    MAX(b.credit_day_overdue)                       AS max_days_overdue,
    MAX(b.amt_credit_max_overdue)                   AS max_overdue_amount,

    -- Credit amounts
    SUM(b.amt_credit_sum)                           AS total_bureau_credit,
    SUM(b.amt_credit_sum_debt)                      AS total_bureau_debt,
    AVG(b.amt_credit_sum)                           AS avg_bureau_credit,

    -- Debt ratio across bureau records
    SUM(b.amt_credit_sum_debt)
        / NULLIF(SUM(b.amt_credit_sum), 0)          AS bureau_debt_ratio,

    -- Prolongations (renegotiations — risk signal)
    SUM(b.cnt_credit_prolong)                       AS total_credit_prolongations,

    -- Credit type diversity
    COUNT(DISTINCT b.credit_type)                   AS distinct_credit_types,

    -- Bureau balance worst status (joined)
    MAX(bb.worst_bureau_status)                     AS worst_ever_bureau_status

FROM bureau b
LEFT JOIN bureau_balance bb ON b.sk_id_bureau = bb.sk_id_bureau
GROUP BY b.sk_id_curr
