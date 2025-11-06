"""
Test configuration and fixtures for pytest.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

@pytest.fixture
def sample_insurance_data():
    """Fixture for sample insurance data."""
    return pd.DataFrame({
        'region': ['A', 'A', 'B', 'B', 'C'],
        'year': [2020, 2021, 2020, 2021, 2020],
        'month': [1, 2, 1, 2, 1],
        'claims_count': [10, 15, 8, 12, 5],
        'total_claim_amount': [10000, 15000, 12000, 18000, 8000],
        'avg_claim_amount': [1000, 1000, 1500, 1500, 1600],
        'policy_count': [100, 100, 80, 80, 50]
    })

@pytest.fixture
def sample_climate_data():
    """Fixture for sample climate data."""
    return pd.DataFrame({
        'region': ['A', 'A', 'B', 'B', 'C'],
        'year': [2020, 2021, 2020, 2021, 2020],
        'month': [1, 2, 1, 2, 1],
        'avg_temperature': [15.5, 16.0, 14.0, 15.0, 17.0],
        'total_rainfall': [100.5, 85.2, 120.0, 90.0, 75.5],
        'flood_events': [1, 0, 2, 1, 0],
        'storm_events': [0, 1, 1, 0, 0]
    })

@pytest.fixture
def data_cleaner():
    """Fixture for DataCleaner instance."""
    from src.data_cleaning import DataCleaner
    return DataCleaner()
