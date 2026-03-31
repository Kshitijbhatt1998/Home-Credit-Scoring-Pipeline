# 🗒️ Feature Dictionary

A comprehensive reference for the key engineered features in the **Home Credit Scoring Pipeline**. As per fintech standards, these features are designed for maximum signal-to-noise ratio in credit risk modeling.

## 📊 Bureau Features (`credit_bureau_features.sql`)

Aggregated signals from the Central Credit Bureau (CIBIL or equivalent) history per applicant.

| Feature Name | Description | Rationale |
| :--- | :--- | :--- |
| `bureau_debt_ratio` | `total_bureau_debt / total_bureau_credit` | **Highest Signal**: Measures how "maxed out" an applicant is on their external lines of credit. |
| `overdue_bureau_count` | `COUNT(*) WHERE DAYS_CREDIT_OVERDUE > 0` | Immediate flag for current delinquency. |
| `total_credit_prolongations` | `SUM(CNT_CREDIT_PROLONG)` | A signal of renegotiating distressed debt across different lenders. |
| `bureau_record_count` | `COUNT(*)` | Measures the breadth and depth of the applicant's credit history. |
| `active_bureau_count` | `COUNT(*) WHERE CREDIT_ACTIVE = 'Active'` | Count of currently open credit lines. |

---

## 💳 Payment Behaviour (`credit_payment_features.sql`)

Signals derived from historical installment payments and behavior on prior loans.

| Feature Name | Description | Rationale |
| :--- | :--- | :--- |
| `avg_payment_lag` | `AVG(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT)` | Measures the "gap" between due date and actual payment. Positive = early payment. |
| `late_payment_rate` | `AVG(PAID_AFTER_DUE)` | Percentage of historical payments that were legally late. |
| `underpayment_rate` | `AVG(AMT_PAYMENT < AMT_INSTALMENT)` | Measures partial payment frequency, a leading indicator of distress. |
| `overall_payment_completeness` | `TOTAL_PAID / TOTAL_OWED` | Final metric for reliability across all installments. |

---

## 📋 Application Ratios (`clean_application.sql`)

Derived signals from the current loan application and basic applicant profile.

| Feature Name | Description | Rationale |
| :--- | :--- | :--- |
| `credit_income_ratio` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Measures loan size-to-earnings capacity. Higher = potentially over-leveraged. |
| `annuity_income_ratio` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Measures actual monthly repayment burden against monthly income. |
| `employment_age_ratio` | `ABS(DAYS_EMPLOYED) / ABS(DAYS_BIRTH)` | Measures career stability relative to age. |
| `ext_source_avg` | `MEAN(EXT_SOURCE_1, 2, 3)` | Combines top internal and external scoring scores into a unified prior. |
| `is_employed` | `CASE WHEN DAYS_EMPLOYED = 365243 THEN 0 ELSE 1 END` | Boolean flag for employment status (handling the special 365243 null). |

---
**Lead ML Engineer**: Kshitij Bhatt
**Project Repo**: [Home-Credit-Scoring-Pipeline](https://github.com/Kshitijbhatt1998/Home-Credit-Scoring-Pipeline)
