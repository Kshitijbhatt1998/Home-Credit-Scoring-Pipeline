# Technical Note: Home Credit Scoring Pipeline Architecture & Scalability

This note summarizes the high-level engineering decisions behind the Home Credit Scoring Pipeline, focusing on its ability to handle high-volume fintech data while remaining developer-friendly and verifiable.

## 🏗️ Core Architecture: DuckDB + Lightweight Compute
- **Zero-Cluster Scalability**: The pipeline leverages **DuckDB** for in-process OLAP, allowing it to process millions of rows (e.g., the 10M+ Kaggle dataset) on standard machine hardware without the overhead of Spark or Snowflake.
- **Embedded Performance**: Features are engineered using direct SQL in DuckDB, optimizing for speed and minimizing memory pressure during large-scale transformation steps.

## 🛡️ "Safety-First" Integrity (PIT & Leakage)
- **Point-in-Time (PIT) Logic**: To prevent "Look-ahead bias"—the most common failure in credit modeling—every feature is calculated using a strict `DAYS_RELATIVE <= 0` safety valve. This ensures that only data available at the exact moment of an applicant's request is used for training.
- **Automated Leakage Scanner**: A dedicated `audit_scanner.py` runs before every training session, verifying that target variables haven't leaked into the feature set. This provides a "Data Leakage Check: PASSED" certification.

## 🚀 "Demo Mode" Strategy (instant Stakeholder Value)
- **Database-Aware Dashboard**: The Streamlit frontend detects the presence of the `credit.duckdb` file. If missing (as on Hugging Face Spaces or new development machines), it automatically switches to **Demo Mode**.
- **Synthetic Fallback**: Demo Mode uses a pre-calculated cache of metrics and SHAP (SHapley Additive exPlanations) values to show a fully functional interface, allowing stakeholders to experience the dashboard without needing to ingest 10GB of raw data first.

## 🔒 Compliance & Security
- **Hashed Identifiers (SHA-256)**: All PII (Personally Identifiable Information) such as `sk_id_curr` is hashed upon ingestion. This demonstrates "Privacy by Design" and ensures that the modeling environment remains SOC2/GDPR compliant.

---
**Technical Point of Contact:** Kshitij Bhatt
**Project Repository:** [Home-Credit-Scoring-Pipeline](https://github.com/Kshitijbhatt1998/Home-Credit-Scoring-Pipeline)
