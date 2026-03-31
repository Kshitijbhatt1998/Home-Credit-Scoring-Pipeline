"""
Home Credit Scoring Pipeline — Smoke Test Data Generator (Lowercase Version)
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
        "sk_id_curr":           app_ids,
        "target":               np.random.choice([0, 1], NUM_APPS, p=[0.92, 0.08]),
        "name_contract_type":   np.random.choice(["Cash loans", "Revolving loans"], NUM_APPS),
        "code_gender":          np.random.choice(["M", "F"], NUM_APPS),
        "flag_own_car":         np.random.choice(["Y", "N"], NUM_APPS),
        "flag_own_realty":      np.random.choice(["Y", "N"], NUM_APPS),
        "cnt_children":         np.random.randint(0, 5, NUM_APPS),
        "amt_income_total":     np.random.lognormal(11.7, 0.5, NUM_APPS).round(0),
        "amt_credit":           np.random.lognormal(13.2, 0.6, NUM_APPS).round(0),
        "amt_annuity":          np.random.lognormal(10.5, 0.4, NUM_APPS).round(0),
        "amt_goods_price":      np.random.lognormal(13.1, 0.6, NUM_APPS).round(0),
        "name_income_type":     np.random.choice(["Working", "Commercial associate", "Pensioner"], NUM_APPS),
        "name_education_type":  np.random.choice(["Secondary / secondary special", "Higher education"], NUM_APPS),
        "name_family_status":   np.random.choice(["Married", "Single / not married"], NUM_APPS),
        "name_housing_type":    np.random.choice(["House / apartment", "Rented apartment"], NUM_APPS),
        "days_birth":           np.random.randint(-25000, -7000, NUM_APPS),
        "days_employed":        np.random.choice([-1000, -2000, 365243], NUM_APPS),
        "days_registration":    np.random.randint(-15000, 0, NUM_APPS),
        "days_id_publish":      np.random.randint(-5000, 0, NUM_APPS),
        "occupation_type":      np.random.choice(["Laborers", "Sales staff", "Core staff"], NUM_APPS),
        "cnt_fam_members":      np.random.randint(1, 6, NUM_APPS),
        "region_rating_client": np.random.randint(1, 4, NUM_APPS),
        "reg_city_not_work_city": np.random.randint(0, 2, NUM_APPS),
        "organization_type":    np.random.choice(["Business Entity Type 3", "Self-employed", "Other"], NUM_APPS),
        "ext_source_1":         np.random.beta(3, 2, NUM_APPS),
        "ext_source_2":         np.random.beta(3, 2, NUM_APPS),
        "ext_source_3":         np.random.beta(3, 2, NUM_APPS),
    })
    df.to_csv(RAW_DIR / "application_train.csv", index=False)
    print(f"Generated application_train.csv ({len(df)} rows)")

def generate_bureau():
    num_bureau = NUM_APPS * 3
    df = pd.DataFrame({
        "sk_id_curr":           np.repeat(app_ids, 3),
        "sk_id_bureau":         range(200001, 200001 + num_bureau),
        "credit_active":        np.random.choice(["Closed", "Active"], num_bureau),
        "days_credit":          np.random.randint(-3000, 0, num_bureau),
        "credit_day_overdue":   np.random.choice([0, 10, 0], num_bureau),
        "days_credit_enddate":  np.random.randint(-1000, 1000, num_bureau),
        "cnt_credit_prolong":   np.random.choice([0, 1, 0, 0], num_bureau),
        "amt_credit_sum":       np.random.lognormal(12, 1, num_bureau),
        "amt_credit_sum_debt":  np.random.lognormal(11, 1, num_bureau),
        "credit_type":          np.random.choice(["Consumer credit", "Credit card"], num_bureau),
    })
    df.to_csv(RAW_DIR / "bureau.csv", index=False)
    print(f"Generated bureau.csv ({len(df)} rows)")

def generate_previous_application():
    num_prev = NUM_APPS * 2
    df = pd.DataFrame({
        "sk_id_curr":           np.repeat(app_ids, 2),
        "sk_id_prev":           range(300001, 300001 + num_prev),
        "name_contract_status": np.random.choice(["Approved", "Refused", "Canceled"], num_prev),
        "amt_credit":           np.random.lognormal(12, 1, num_prev),
        "days_decision":        np.random.randint(-3000, 0, num_prev),
    })
    df.to_csv(RAW_DIR / "previous_application.csv", index=False)
    print(f"Generated previous_application.csv ({len(df)} rows)")

def generate_installments():
    num_inst = NUM_APPS * 10
    df = pd.DataFrame({
        "sk_id_curr":           np.repeat(app_ids, 10),
        "sk_id_prev":           np.repeat(range(300001, 300001 + NUM_APPS * 2), 5),
        "num_instalment_number": np.tile(range(1, 6), NUM_APPS * 2),
        "days_instalment":      np.random.randint(-500, 0, num_inst),
        "days_entry_payment":   np.random.randint(-500, 0, num_inst),
        "amt_instalment":       np.random.lognormal(10, 1, num_inst),
        "amt_payment":          np.random.lognormal(10, 1, num_inst),
    })
    df.to_csv(RAW_DIR / "installments_payments.csv", index=False)
    print(f"Generated installments_payments.csv ({len(df)} rows)")

def generate_others():
    tables = [
        ("bureau_balance",      ["sk_id_bureau", "months_balance", "status"]),
        ("credit_card_balance", ["sk_id_prev", "sk_id_curr", "months_balance", "amt_balance", "amt_credit_limit_actual"]),
        ("pos_cash_balance",    ["sk_id_prev", "sk_id_curr", "months_balance", "cnt_instalment", "name_contract_status"]),
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
    print("\nSmoke test data generation (LOWERCASE) complete.")
