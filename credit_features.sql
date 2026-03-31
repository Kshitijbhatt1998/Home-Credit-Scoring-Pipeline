-- credit_features: Gold layer — full feature table for model training
-- Joins application + bureau_features + payment_features
-- One row per loan application (SK_ID_CURR)
-- 70+ features ready for LightGBM training

{{
  config(materialized='table')
}}

SELECT
    -- ── Keys ────────────────────────────────────────────────────────────────
    a.sk_id_curr,
    a.target,                           -- 1 = defaulted, 0 = repaid

    -- ── Loan characteristics ─────────────────────────────────────────────
    a.name_contract_type,
    a.amt_credit,
    a.amt_annuity,
    a.amt_income_total,
    a.amt_goods_price,
    a.credit_income_ratio,
    a.annuity_income_ratio,
    a.goods_price_credit_ratio,

    -- ── Applicant demographics ───────────────────────────────────────────
    a.gender_enc,
    a.owns_car,
    a.owns_realty,
    a.cnt_children,
    a.cnt_fam_members,
    a.age_years,

    -- ── Employment ───────────────────────────────────────────────────────
    a.name_income_type,
    a.occupation_type,
    a.organization_type,
    a.is_employed,
    a.days_employed_clean,
    a.employment_age_ratio,
    a.days_registration,
    a.days_id_publish,

    -- ── Education / housing / region ─────────────────────────────────────
    a.name_education_type,
    a.name_family_status,
    a.name_housing_type,
    a.region_rating_client,
    a.reg_city_not_work_city,

    -- ── External credit scores (highest single-feature predictors) ───────
    a.ext_source_1,
    a.ext_source_2,
    a.ext_source_3,

    -- Composite external score
    (COALESCE(a.ext_source_1, 0) + COALESCE(a.ext_source_2, 0) + COALESCE(a.ext_source_3, 0))
        / NULLIF(
            (CASE WHEN a.ext_source_1 IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN a.ext_source_2 IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN a.ext_source_3 IS NOT NULL THEN 1 ELSE 0 END), 0
          )                                             AS ext_source_avg,

    -- ── Bureau features (credit history signals) ─────────────────────────
    b.bureau_record_count,
    b.active_bureau_count,
    b.closed_bureau_count,
    b.days_since_last_bureau,
    b.avg_days_credit,
    b.total_days_overdue,
    b.overdue_bureau_count,
    b.max_days_overdue,
    b.max_overdue_amount,
    b.total_bureau_credit,
    b.total_bureau_debt,
    b.bureau_debt_ratio,
    b.total_credit_prolongations,
    b.distinct_credit_types,
    b.worst_ever_bureau_status,

    -- Bureau debt vs current application (cross-table ratio)
    b.total_bureau_debt / NULLIF(a.amt_credit, 0)      AS bureau_debt_to_new_credit,

    -- ── Payment behaviour features ────────────────────────────────────────
    p.total_instalments,
    p.loans_with_payment_history,
    p.avg_payment_lag,
    p.worst_payment_lag,
    p.payment_lag_volatility,
    p.late_payment_rate,
    p.total_underpayments,
    p.underpayment_rate,
    p.avg_payment_ratio,
    p.min_payment_ratio,
    p.overall_payment_completeness,
    p.prev_loan_count,
    p.prev_approved_count,
    p.prev_approval_rate,
    p.avg_prev_credit_ratio,
    p.days_since_last_application

FROM {{ ref('stg_application') }} a
LEFT JOIN {{ ref('credit_bureau_features') }} b ON a.sk_id_curr = b.sk_id_curr
LEFT JOIN {{ ref('credit_payment_features') }} p ON a.sk_id_curr = p.sk_id_curr
