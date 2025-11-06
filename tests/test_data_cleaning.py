"""
Unit tests for the data_cleaning module.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_cleaning import DataCleaner

class TestDataCleaning(unittest.TestCase):
    """Test cases for the DataCleaner class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data before any tests are run."""
        # Create sample data for testing
        cls.sample_insurance_data = pd.DataFrame({
            'region': ['A', 'A', 'B', 'B', 'C'],
            'year': [2020, 2021, 2020, 2021, 2020],
            'month': [1, 2, 1, 2, 1],
            'claims_count': [10, 15, 8, 12, 5],
            'total_claim_amount': [10000, 15000, 12000, 18000, 8000],
            'avg_claim_amount': [1000, 1000, 1500, 1500, 1600],
            'policy_count': [100, 100, 80, 80, 50]
        })
        
        cls.sample_climate_data = pd.DataFrame({
            'region': ['A', 'A', 'B', 'B', 'C'],
            'year': [2020, 2021, 2020, 2021, 2020],
            'month': [1, 2, 1, 2, 1],
            'avg_temperature': [15.5, 16.0, 14.0, 15.0, 17.0],
            'total_rainfall': [100.5, 85.2, 120.0, 90.0, 75.5],
            'flood_events': [1, 0, 2, 1, 0],
            'storm_events': [0, 1, 1, 0, 0]
        })
        
        # Initialize the cleaner
        cls.cleaner = DataCleaner()
    
    def test_clean_insurance_data(self):
        """Test cleaning insurance data."""
        # Create a copy with some test issues
        test_data = self.sample_insurance_data.copy()
        test_data.loc[0, 'claims_count'] = np.nan
        test_data.loc[1, 'avg_claim_amount'] = 1000000  # Outlier
        
        cleaned = self.cleaner.clean_insurance_data(test_data)
        
        # Check that missing values are handled
        self.assertFalse(cleaned['claims_count'].isna().any())
        
        # Check that outliers are capped
        self.assertLessEqual(
            cleaned['avg_claim_amount'].max(),
            self.cleaner.config['max_claim_amount']
        )
        
        # Check derived feature
        self.assertIn('claim_frequency', cleaned.columns)
    
    def test_clean_climate_data(self):
        """Test cleaning climate data."""
        # Create a copy with some test issues
        test_data = self.sample_climate_data.copy()
        test_data.loc[0, 'avg_temperature'] = -100  # Outlier
        test_data.loc[1, 'total_rainfall'] = np.nan
        
        cleaned = self.cleaner.clean_climate_data(test_data)
        
        # Check that missing values are handled
        self.assertFalse(cleaned['total_rainfall'].isna().any())
        
        # Check that temperature is within valid range
        min_temp, max_temp = self.cleaner.config['valid_temperature_range']
        self.assertGreaterEqual(cleaned['avg_temperature'].min(), min_temp)
        self.assertLessEqual(cleaned['avg_temperature'].max(), max_temp)
        
        # Check derived feature
        self.assertIn('temperature_range', cleaned.columns)
    
    def test_merge_datasets(self):
        """Test merging climate and insurance data."""
        cleaned_insurance = self.cleaner.clean_insurance_data(self.sample_insurance_data)
        cleaned_climate = self.cleaner.clean_climate_data(self.sample_climate_data)
        
        merged = self.cleaner.merge_datasets(cleaned_climate, cleaned_insurance)
        
        # Check that we have the expected columns from both datasets
        self.assertIn('avg_temperature', merged.columns)
        self.assertIn('claims_count', merged.columns)
        
        # Check that the merge was successful
        self.assertEqual(len(merged), len(self.sample_insurance_data))

if __name__ == '__main__':
    unittest.main()