-- credit_payment_features: payment behaviour aggregated per applicant
-- Source: installments_payments (payment history for previous HC loans)
-- One row per SK_ID_CURR

{{
  config(materialized='table')
}}

WITH inst AS (
    SELECT * FROM {{ ref('stg_installments') }}
),

prev AS (
    SELECT
        sk_id_curr,
        COUNT(DISTINCT sk_id_prev)      AS prev_loan_count,
        SUM(was_approved)               AS prev_approved_count,
        AVG(was_approved)               AS prev_approval_rate,
        AVG(prev_credit_application_ratio) AS avg_prev_credit_ratio,
        MAX(days_decision)              AS days_since_last_application
    FROM {{ ref('stg_previous_application') }}
    GROUP BY sk_id_curr
)

SELECT
    i.sk_id_curr,

    -- Payment volume
    COUNT(*)                            AS total_instalments,
    COUNT(DISTINCT i.sk_id_prev)        AS loans_with_payment_history,

    -- Lateness signals (most predictive payment features)
    AVG(i.payment_lag_days)             AS avg_payment_lag,
    MAX(i.payment_lag_days)             AS worst_payment_lag,
    STDDEV(i.payment_lag_days)          AS payment_lag_volatility,

    -- Late payment rate (paid after due date)
    AVG(CASE WHEN i.payment_lag_days > 0 THEN 1.0 ELSE 0.0 END)
                                        AS late_payment_rate,

    -- Underpayment signals
    SUM(i.is_underpayment)              AS total_underpayments,
    AVG(i.is_underpayment)              AS underpayment_rate,

    -- Payment completeness
    AVG(i.payment_ratio)                AS avg_payment_ratio,
    MIN(i.payment_ratio)                AS min_payment_ratio,

    -- Total amounts
    SUM(i.amt_instalment)               AS total_instalment_amount,
    SUM(i.amt_payment)                  AS total_payment_amount,
    SUM(i.amt_payment) / NULLIF(SUM(i.amt_instalment), 0)
                                        AS overall_payment_completeness,

    -- Previous application features (joined)
    p.prev_loan_count,
    p.prev_approved_count,
    p.prev_approval_rate,
    p.avg_prev_credit_ratio,
    p.days_since_last_application

FROM inst i
LEFT JOIN prev p ON i.sk_id_curr = p.sk_id_curr
GROUP BY
    i.sk_id_curr,
    p.prev_loan_count,
    p.prev_approved_count,
    p.prev_approval_rate,
    p.avg_prev_credit_ratio,
    p.days_since_last_application
