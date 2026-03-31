"""
Home Credit Scoring Pipeline — Smoke Test Data Generator

Generates 7 dummy CSV files in data/raw/ to enable a minimal "Smoke Test"
of the ingestion, dbt transformation, and model training pipeline.
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np

# --- Config ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Applicants
NUM_APPS = 100
app_ids = list(range(100001, 100001 + NUM_APPS))

def generate_application_train():
    df = pd.DataFrame({
        "SK_ID_CURR":           app_ids,
        "TARGET":               np.random.choice([0, 1], NUM_APPS, p=[0.92, 0.08]),
        "NAME_CONTRACT_TYPE":   np.random.choice(["Cash loans", "Revolving loans"], NUM_APPS),
        "CODE_GENDER":          np.random.choice(["M", "F"], NUM_APPS),
        "FLAG_OWN_CAR":         np.random.choice(["Y", "N"], NUM_APPS),
        "FLAG_OWN_REALTY":      np.random.choice(["Y", "N"], NUM_APPS),
        "CNT_CHILDREN":         np.random.randint(0, 5, NUM_APPS),
        "AMT_INCOME_TOTAL":     np.random.lognormal(11.7, 0.5, NUM_APPS).round(0),
        "AMT_CREDIT":           np.random.lognormal(13.2, 0.6, NUM_APPS).round(0),
        "AMT_ANNUITY":          np.random.lognormal(10.5, 0.4, NUM_APPS).round(0),
        "AMT_GOODS_PRICE":      np.random.lognormal(13.1, 0.6, NUM_APPS).round(0),
        "NAME_INCOME_TYPE":     np.random.choice(["Working", "Commercial associate", "Pensioner"], NUM_APPS),
        "NAME_EDUCATION_TYPE":  np.random.choice(["Secondary / secondary special", "Higher education"], NUM_APPS),
        "NAME_FAMILY_STATUS":   np.random.choice(["Married", "Single / not married"], NUM_APPS),
        "NAME_HOUSING_TYPE":    np.random.choice(["House / apartment", "Rented apartment"], NUM_APPS),
        "DAYS_BIRTH":           np.random.randint(-25000, -7000, NUM_APPS),
        "DAYS_EMPLOYED":        np.random.choice([-1000, -2000, 365243], NUM_APPS),  # Includes unemployed sentinel
        "DAYS_REGISTRATION":    np.random.randint(-15000, 0, NUM_APPS),
        "DAYS_ID_PUBLISH":      np.random.randint(-5000, 0, NUM_APPS),
        "OCCUPATION_TYPE":      np.random.choice(["Laborers", "Sales staff", "Core staff"], NUM_APPS),
        "CNT_FAM_MEMBERS":      np.random.randint(1, 6, NUM_APPS),
        "REGION_RATING_CLIENT": np.random.randint(1, 4, NUM_APPS),
        "REG_CITY_NOT_WORK_CITY": np.random.randint(0, 2, NUM_APPS),
        "ORGANIZATION_TYPE":    np.random.choice(["Business Entity Type 3", "Self-employed", "Other"], NUM_APPS),
        "EXT_SOURCE_1":         np.random.beta(3, 2, NUM_APPS),
        "EXT_SOURCE_2":         np.random.beta(3, 2, NUM_APPS),
        "EXT_SOURCE_3":         np.random.beta(3, 2, NUM_APPS),
    })
    df.to_csv(RAW_DIR / "application_train.csv", index=False)
    print(f"Generated application_train.csv ({len(df)} rows)")

def generate_bureau():
    num_bureau = NUM_APPS * 3
    df = pd.DataFrame({
        "SK_ID_CURR":           np.repeat(app_ids, 3),
        "SK_ID_BUREAU":         range(200001, 200001 + num_bureau),
        "CREDIT_ACTIVE":        np.random.choice(["Closed", "Active"], num_bureau),
        "DAYS_CREDIT":          np.random.randint(-3000, 0, num_bureau),
        "CREDIT_DAY_OVERDUE":   np.random.choice([0, 10, 0], num_bureau),
        "DAYS_CREDIT_ENDDATE":  np.random.randint(-1000, 1000, num_bureau),
        "CNT_CREDIT_PROLONG":   np.random.choice([0, 1, 0, 0], num_bureau),
        "AMT_CREDIT_SUM":       np.random.lognormal(12, 1, num_bureau),
        "AMT_CREDIT_SUM_DEBT":  np.random.lognormal(11, 1, num_bureau),
        "CREDIT_TYPE":          np.random.choice(["Consumer credit", "Credit card"], num_bureau),
    })
    df.to_csv(RAW_DIR / "bureau.csv", index=False)
    print(f"Generated bureau.csv ({len(df)} rows)")

def generate_previous_application():
    num_prev = NUM_APPS * 2
    df = pd.DataFrame({
        "SK_ID_CURR":           np.repeat(app_ids, 2),
        "SK_ID_PREV":           range(300001, 300001 + num_prev),
        "NAME_CONTRACT_STATUS": np.random.choice(["Approved", "Refused", "Canceled"], num_prev),
        "AMT_CREDIT":           np.random.lognormal(12, 1, num_prev),
        "DAYS_DECISION":        np.random.randint(-3000, 0, num_prev),
    })
    df.to_csv(RAW_DIR / "previous_application.csv", index=False)
    print(f"Generated previous_application.csv ({len(df)} rows)")

def generate_installments():
    num_inst = NUM_APPS * 10
    df = pd.DataFrame({
        "SK_ID_CURR":           np.repeat(app_ids, 10),
        "SK_ID_PREV":           np.repeat(range(300001, 300001 + NUM_APPS * 2), 5),
        "NUM_INSTALMENT_NUMBER": np.tile(range(1, 6), NUM_APPS * 2),
        "DAYS_INSTALMENT":      np.random.randint(-500, 0, num_inst),
        "DAYS_ENTRY_PAYMENT":   np.random.randint(-500, 0, num_inst),
        "AMT_INSTALMENT":       np.random.lognormal(10, 1, num_inst),
        "AMT_PAYMENT":          np.random.lognormal(10, 1, num_inst),
    })
    df.to_csv(RAW_DIR / "installments_payments.csv", index=False)
    print(f"Generated installments_payments.csv ({len(df)} rows)")

def generate_others():
    # Empty but schema-correct headers for less important tables
    tables = [
        ("bureau_balance",      ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]),
        ("credit_card_balance", ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]),
        ("POS_CASH_balance",    ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT", "NAME_CONTRACT_STATUS"]),
    ]
    for name, cols in tables:
        df = pd.DataFrame(columns=cols)
        df.to_csv(RAW_DIR / f"{name}.csv", index=False)
        print(f"Generated {name}.csv (Header only)")

if __name__ == "__main__":
    generate_application_train()
    generate_bureau()
    generate_previous_application()
    generate_installments()
    generate_others()
    print("\nSmoke test data generation complete.")
