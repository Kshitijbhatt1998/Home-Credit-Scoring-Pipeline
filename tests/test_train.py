import pytest
import pandas as pd
import numpy as np
from src.train_credit import encode_categoricals, build_feature_matrix

def test_encode_categoricals():
    """Verify categorical label encoding handles missing values correctly."""
    df = pd.DataFrame({
        "cat_col": ["A", "B", "A", None, "C"],
        "num_col": [1, 2, 3, 4, 5],
    })
    
    encoded_df, encoders = encode_categoricals(df, ["cat_col"])
    
    # 1. 'MISSING' should be one of the encoded categories
    assert encoders["cat_col"] is not None
    assert "MISSING" in encoders["cat_col"].classes_
    
    # 2. Check that same labels map to same integer
    assert encoded_df.loc[0, "cat_col"] == encoded_df.loc[2, "cat_col"]
    assert encoded_df.loc[0, "cat_col"] != encoded_df.loc[1, "cat_col"]

def test_build_feature_matrix():
    """Verify that only available features are included in X."""
    df = pd.DataFrame({
        "amt_credit": [1000, 2000],
        "name_contract_type": ["Cash", "Revolving"],
        "target": [0, 1],
        "non_existent_feature": [0, 0],
    })
    
    # We pass the full set of configured columns, but only some exist
    X, y, encoders = build_feature_matrix(df)
    
    assert "amt_credit" in X.columns
    assert "name_contract_type" in X.columns
    assert "non_existent_feature" not in X.columns
    assert "target" not in X.columns
    assert len(y) == 2
    assert y.name == "target"
