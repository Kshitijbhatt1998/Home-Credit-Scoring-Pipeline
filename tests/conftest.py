import pytest
import duckdb
import pandas as pd
from pathlib import Path

@pytest.fixture
def mock_db():
    """Setup an in-memory DuckDB for testing."""
    con = duckdb.connect(":memory:")
    yield con
    con.close()

@pytest.fixture
def sample_application_data():
    """Create a small dummy application dataset."""
    return pd.DataFrame({
        "SK_ID_CURR": [100001, 100002, 100003],
        "TARGET": [0, 1, 0],
        "AMT_CREDIT": [50000, 100000, 75000],
        "AMT_ANNUITY": [5000, 10000, 7500],
        "AMT_INCOME_TOTAL": [20000, 30000, 25000],
        "DAYS_BIRTH": [-10000, -15000, -12000],
        "DAYS_EMPLOYED": [-1000, 365243, -2000],  # 365243 is the 'unemployed' sentinel
        "CODE_GENDER": ["M", "F", "F"],
        "FLAG_OWN_CAR": ["Y", "N", "Y"],
        "FLAG_OWN_REALTY": ["Y", "Y", "N"],
    })

@pytest.fixture
def mock_raw_data_dir(tmp_path):
    """Setup a temporary directory with dummy raw CSVs."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    return raw_dir
