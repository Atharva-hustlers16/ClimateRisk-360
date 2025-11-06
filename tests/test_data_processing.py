"""
Unit tests for the data_processing module.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import the module to test
from data_processing import DataProcessor


class TestDataProcessing:
    """Test cases for the DataProcessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        return pd.DataFrame({
            'region': ['A'] * 6 + ['B'] * 6,
            'date': dates,
            'avg_temperature': [15 + i * 0.5 for i in range(12)],
            'total_rainfall': [100 - i * 5 for i in range(12)],
            'flood_events': [i % 3 for i in range(12)],
            'claims_count': [i + 5 for i in range(12)],
            'total_claim_amount': [1000 * (i + 1) for i in range(12)]
        })
    
    def test_process_merged_data(self, sample_data):
        """Test the process_merged_data method."""
        processor = DataProcessor()
        processed = processor.process_merged_data(sample_data)
        
        # Check that time features were added
        assert 'year' in processed.columns
        assert 'month' in processed.columns
        assert 'quarter' in processed.columns
        assert 'season' in processed.columns
        
        # Check that rolling features were added
        assert 'avg_temperature_rolling_mean' in processed.columns
        assert 'total_rainfall_rolling_std' in processed.columns
        
        # Check that lagged features were added
        assert 'avg_temperature_lag1' in processed.columns
        assert 'total_rainfall_lag3' in processed.columns
        
        # Check that derived features were added
        assert 'temp_anomaly' in processed.columns
        assert 'rainfall_anomaly' in processed.columns
        assert 'claim_severity' in processed.columns
        
        # Check that risk score was calculated
        assert 'risk_score' in processed.columns
        assert processed['risk_score'].between(0, 100).all()
    
    def test_aggregate_to_region_level(self, sample_data):
        """Test the aggregate_to_region_level method."""
        processor = DataProcessor()
        processed = processor.process_merged_data(sample_data)
        aggregated = processor.aggregate_to_region_level(processed, freq='Q')
        
        # Check that we have the expected columns
        expected_columns = [
            'region', 'date', 'avg_temperature', 'total_rainfall',
            'flood_events', 'claims_count', 'total_claim_amount',
            'avg_claim_amount', 'claim_frequency', 'risk_score'
        ]
        
        for col in expected_columns:
            assert col in aggregated.columns
        
        # Check that we have the expected number of rows (2 regions * 4 quarters = 8)
        assert len(aggregated) == 8
        
        # Check that aggregation worked correctly
        assert (aggregated.groupby('region').size() == 4).all()
    
    def test_calculate_risk_scores(self, sample_data):
        """Test risk score calculation."""
        processor = DataProcessor()
        
        # Test with default weights
        processed = processor.process_merged_data(sample_data)
        assert 'risk_score' in processed.columns
        assert processed['risk_score'].between(0, 100).all()
        
        # Test with custom weights
        custom_weights = {
            'temperature': 0.5,
            'rainfall': 0.3,
            'flood_events': 0.2
        }
        
        processor.config['risk_weights'] = custom_weights
        processed_custom = processor.process_merged_data(sample_data)
        
        # Check that risk scores changed with different weights
        assert not processed['risk_score'].equals(processed_custom['risk_score'])
        
    def test_scale_features(self, sample_data):
        """Test feature scaling."""
        processor = DataProcessor()
        processed = processor.process_merged_data(sample_data)
        
        # Check that numerical features were scaled
        numeric_cols = [
            'avg_temperature', 'total_rainfall',
            'flood_events', 'claims_count', 'total_claim_amount'
        ]
        
        for col in numeric_cols:
            if col in processed.columns:
                # Check that values are within a reasonable range for standardized data
                assert abs(processed[col].mean()) < 1e-10 or abs(1 - processed[col].mean()) < 1e-10
