import pytest
import pandas as pd
import duckdb
from src.ingest_credit import clean_application, drop_high_null_cols, MANDATORY_APPLICATION

def test_drop_high_null_cols(mock_db):
    """Verify that columns with too many nulls are dropped."""
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4, 5],
        "TARGET": [0, 0, 0, 0, 1],
        "MOSTLY_NULL": [None, None, None, None, 1],  # 80% null
        "NO_NULL": [1, 2, 3, 4, 5],                  # 0% null
    })
    mock_db.execute("CREATE TABLE raw_application_train AS SELECT * FROM df")
    
    kept = drop_high_null_cols(
        mock_db, "raw_application_train", MANDATORY_APPLICATION, threshold=60.0
    )
    
    assert "NO_NULL" in kept
    assert "MOSTLY_NULL" not in kept
    # Mandatory columns should always be kept
    assert "SK_ID_CURR" in kept

def test_clean_application_derived_features(mock_db):
    """Verify calculation of derived columns in clean_application."""
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "AMT_CREDIT": [100000, 200000],
        "AMT_ANNUITY": [10000, 20000],
        "AMT_INCOME_TOTAL": [50000, 50000],
        "DAYS_BIRTH": [-10000, -20000],
        "DAYS_EMPLOYED": [-1000, 365243],  # 365243 is special
        "AMT_GOODS_PRICE": [90000, 180000],
        "CODE_GENDER": ["M", "F"],
        "FLAG_OWN_CAR": ["Y", "N"],
        "FLAG_OWN_REALTY": ["Y", "Y"],
    })
    mock_db.execute("CREATE TABLE raw_application_train AS SELECT * FROM df")
    
    clean_application(mock_db)
    
    result = mock_db.execute("SELECT * FROM clean_application").df()
    
    # 1. Ratios
    assert result.loc[0, "credit_income_ratio"] == 2.0
    assert result.loc[0, "annuity_income_ratio"] == 0.2
    
    # 2. Employment handling
    # For DAYS_EMPLOYED = 365243, is_employed should be 0 and days_employed_clean should be NULL
    assert result.loc[0, "is_employed"] == 1
    assert result.loc[1, "is_employed"] == 0
    assert pd.isna(result.loc[1, "days_employed_clean"])
    
    # 3. Encoding
    assert result.loc[0, "gender_enc"] == 1  # 'M'
    assert result.loc[1, "gender_enc"] == 0  # 'F'
    assert result.loc[0, "owns_car"] == 1    # 'Y'
    assert result.loc[1, "owns_car"] == 0    # 'N'
