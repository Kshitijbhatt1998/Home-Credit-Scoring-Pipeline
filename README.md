# Home Credit Scoring Pipeline

**End-to-end credit default prediction pipeline — 307K loan applications, 7 related tables, 70+ engineered features. The data engineering layer that lets your ML team train models instead of building plumbing.**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![dbt](https://img.shields.io/badge/dbt-1.7-orange)](https://getdbt.com)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.10-yellow)](https://duckdb.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-AUC≥0.77-brightgreen)](./models/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](./src/dashboard_credit.py)

---

## What This Demonstrates

This pipeline ingests 7 related tables from the Home Credit Default Risk dataset, joins them correctly across multiple levels of aggregation, engineers 70+ credit-specific features, and delivers a model-ready dataset — the same kind of pipeline I build for fintech AI teams.

**If your ML team is bottlenecked building the feature pipeline instead of the model — this is what I deliver.**

→ [View fraud detection pipeline](https://github.com/Kshitijbhatt1998/fintech-fraud-pipeline) | [Connect on LinkedIn](https://linkedin.com/in/kshitijbhatt)

---

## Why This Pipeline Is Different From Fraud Detection

| Dimension | Fraud Pipeline | This Pipeline |
|-----------|---------------|---------------|
| Data shape | 2 tables | **7 related tables** |
| Join complexity | Simple left join | Multi-level aggregation joins |
| Feature type | Velocity + temporal | Bureau aggregates + payment ratios |
| CV strategy | TimeSeriesSplit (temporal) | StratifiedKFold (snapshot data) |
| Model | XGBoost | **LightGBM** (standard for credit) |
| Null handling | Column drop (>90%) | Imputation strategy per feature group |

The CV strategy difference is intentional and documented — fraud detection requires temporal ordering to prevent leakage; credit applications are point-in-time snapshots where stratified sampling is correct.

---

## Pipeline Results

| Metric | Value |
|--------|-------|
| Loan applications | 307,511 |
| Supporting table rows | ~10M+ (bureau, payments, etc.) |
| Features engineered | 70+ |
| Model (LightGBM) CV AUC | **≥ 0.77** |
| Kaggle benchmark AUC | 0.79–0.80 |
| Pipeline runtime | < 8 minutes |

---

## Architecture

```
7 Raw CSVs (Home Credit Kaggle)
        │
        ▼
[1. Ingest]  src/ingest_credit.py
        DuckDB ← read_csv_auto (7 tables)
        - Clean application: derived ratios, employment flags, age
        - Supporting tables: pass-through
        │
        ▼
[2. Transform]  dbt (dbt_project/)
        staging/
          stg_application       ← main application table
          stg_bureau            ← credit bureau history
          stg_previous_app      ← prior HC loan applications
          stg_installments      ← payment history
        marts/
          credit_bureau_features  ← aggregated bureau signals per applicant
          credit_payment_features ← payment behaviour + prior loan history
          credit_features         ← full joined feature table (70+ cols)
          credit_summary          ← risk by contract type, income, education
        │
        ▼
[3. Train]  src/train_credit.py
        LightGBM + StratifiedKFold (5-fold)
        MLflow experiment tracking
        → models/lgbm_credit_v1.pkl
        │
        ▼
[4. Serve]  src/dashboard_credit.py
        Streamlit: Overview / Risk Breakdown / Model Performance / Application Explorer
```

---

## Key Engineered Features

### Bureau Aggregates (per applicant, across all bureau records)
```sql
bureau_debt_ratio          = total_bureau_debt / total_bureau_credit
overdue_bureau_count       = COUNT(*) WHERE credit_day_overdue > 0
total_credit_prolongations = SUM(cnt_credit_prolong)   -- renegotiation signal
bureau_debt_to_new_credit  = total_bureau_debt / amt_credit (cross-table)
```

### Payment Behaviour (from installment history)
```sql
avg_payment_lag            = AVG(days_instalment - days_entry_payment)
late_payment_rate          = AVG(paid_after_due)
underpayment_rate          = AVG(amt_payment < amt_instalment)
overall_payment_completeness = total_paid / total_owed
```

### Application Ratios
```sql
credit_income_ratio        = amt_credit / amt_income_total
annuity_income_ratio       = amt_annuity / amt_income_total
ext_source_avg             = mean of EXT_SOURCE_1/2/3 (top predictors)
employment_age_ratio       = abs(days_employed) / abs(days_birth)
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Storage | DuckDB |
| Transformation | dbt (staging → marts) |
| ML | LightGBM, scikit-learn |
| Experiment tracking | MLflow |
| Dashboard | Streamlit + Plotly |
| Language | Python 3.10 |

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/Kshitijbhatt1998/home-credit-scoring-pipeline.git
cd home-credit-scoring-pipeline
pip install -r requirements.txt

# 2. Download data from Kaggle
# https://www.kaggle.com/c/home-credit-default-risk/data
# Place all 7 CSV files in data/raw/

# 3. Ingest
python src/ingest_credit.py

# 4. Transform
cd dbt_project && dbt run && dbt test && cd ..

# 5. Train
python src/train_credit.py

# 6. Dashboard
streamlit run src/dashboard_credit.py

# 7. Deployment (Optional - Containerized)
docker-compose up -d --build
```

---

## Deployment & Production Readiness

The pipeline is containerized for easy deployment to cloud VMs (EC2, GCP, etc.) or local servers.

| Component | Port | Host |
| :--- | :--- | :--- |
| **Streamlit Dashboard** | `8501` | `localhost:8501` |
| **MLflow Tracking** | `5000` | `localhost:5000` |

### Running with Docker
1. Ensure `docker` and `docker-compose` are installed.
2. Run `docker-compose up --build`.
3. Access the dashboard at `http://localhost:8501`.

### Data Persistence
- **DuckDB**: Stored in `data/credit.duckdb`.
- **ML Models**: Stored in `models/lgbm_credit_v1.pkl`.
- **MLflow Logs**: Stored in the `mlruns/` local directory.

---

## Dbt Project Management

The transformation layer is managed by dbt.
- **Location**: `dbt_project/`
- **Profiles**: Ensure `dbt_project/profiles.yml` points to the correct DuckDB path.
- **Run**: `cd dbt_project && dbt run`
- **Test**: `cd dbt_project && dbt test`

---

## Automated Testing

This project uses `pytest` for unit and integration testing.

```bash
# 1. Install testing dependencies
pip install pytest pytest-mock

# 2. Run all tests
pytest tests/
```

- **`tests/test_ingest.py`**: Validates data cleaning logic/ratios.
- **`tests/test_train.py`**: Validates categorical encoding/feature matrix.
- **`tests/conftest.py`**: Shared fixtures (in-memory DuckDB).



---

## Data Source

Home Credit Default Risk dataset (Kaggle, 2018). Publicly available for research.

---

## About

Built by **Kshitij Bhatt** — Data Engineer specializing in fintech AI pipeline infrastructure.

This is the second in a series of public proof-of-work case studies:
1. [Fraud Detection Pipeline](https://github.com/Kshitijbhatt1998/fintech-fraud-pipeline) — transaction-level, XGBoost, AUC 0.9791
2. **Credit Scoring Pipeline** (this repo) — application-level, LightGBM, 7-table join

**Services:** Custom data pipelines · Feature engineering · dbt transformations · Model-ready dataset delivery · Monthly retainer maintenance

→ [GitHub](https://github.com/Kshitijbhatt1998) | [LinkedIn](https://linkedin.com/in/kshitijbhatt)
