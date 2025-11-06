"""Data cleaning module for ClimateRisk360.

This module handles cleaning and preprocessing of the raw data.
"""
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Handles cleaning and preprocessing of climate and insurance data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data cleaner with optional configuration.
        
        Args:
            config: Configuration dictionary with cleaning parameters
        """
        self.config = config or {
            'outlier_threshold': 3.0,  # Z-score threshold for outliers
            'min_claim_amount': 100,   # Minimum valid claim amount
            'max_claim_amount': 1_000_000,  # Maximum valid claim amount
            'valid_temperature_range': (-50, 60),  # Valid temperature range in Celsius
            'valid_rainfall_range': (0, 2000),     # Valid monthly rainfall in mm
        }
    
    def clean_insurance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess insurance claims data.
        
        Args:
            df: Raw insurance claims data
            
        Returns:
            Cleaned and processed insurance data
        """
        if df.empty:
            logger.warning("Empty insurance data frame provided")
            return df
            
        logger.info(f"Cleaning insurance data with {len(df)} rows")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean, 'insurance')
        
        # Convert data types
        numeric_cols = ['claims_count', 'total_claim_amount', 'avg_claim_amount', 'policy_count']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean, ['avg_claim_amount', 'claims_count'])
        
        # Validate claim amounts
        if 'avg_claim_amount' in df_clean.columns:
            mask = ((df_clean['avg_claim_amount'] < self.config['min_claim_amount']) | 
                    (df_clean['avg_claim_amount'] > self.config['max_claim_amount']))
            df_clean.loc[mask, 'avg_claim_amount'] = np.nan
        
        # Add derived features
        if all(col in df_clean.columns for col in ['claims_count', 'policy_count']):
            df_clean['claim_frequency'] = df_clean['claims_count'] / df_clean['policy_count']
        
        logger.info(f"Cleaned insurance data: {len(df_clean)} rows")
        return df_clean
    
    def clean_climate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess climate data.
        
        Args:
            df: Raw climate data
            
        Returns:
            Cleaned and processed climate data
        """
        if df.empty:
            logger.warning("Empty climate data frame provided")
            return df
            
        logger.info(f"Cleaning climate data with {len(df)} rows")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean, 'climate')
        
        # Convert data types
        numeric_cols = [
            'avg_temperature', 'total_rainfall', 'max_temperature', 
            'min_temperature', 'flood_events', 'storm_events'
        ]
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Validate ranges
        if 'avg_temperature' in df_clean.columns:
            min_temp, max_temp = self.config['valid_temperature_range']
            df_clean['avg_temperature'] = df_clean['avg_temperature'].clip(min_temp, max_temp)
        
        if 'total_rainfall' in df_clean.columns:
            min_rain, max_rain = self.config['valid_rainfall_range']
            df_clean['total_rainfall'] = df_clean['total_rainfall'].clip(min_rain, max_rain)
        
        # Handle outliers
        df_clean = self._handle_outliers(
            df_clean, 
            ['avg_temperature', 'total_rainfall', 'flood_events', 'storm_events']
        )
        
        # Add derived features
        if all(col in df_clean.columns for col in ['max_temperature', 'min_temperature']):
            df_clean['temperature_range'] = df_clean['max_temperature'] - df_clean['min_temperature']
        
        logger.info(f"Cleaned climate data: {len(df_clean)} rows")
        return df_clean
    
    def merge_datasets(
        self, 
        climate_df: pd.DataFrame, 
        insurance_df: pd.DataFrame,
        on: List[str] = ['region', 'year', 'month']
    ) -> pd.DataFrame:
        """Merge climate and insurance data on specified columns.
        
        Args:
            climate_df: Cleaned climate data
            insurance_df: Cleaned insurance data
            on: List of columns to merge on
            
        Returns:
            Merged dataset
        """
        if climate_df.empty or insurance_df.empty:
            logger.warning("One or both input dataframes are empty")
            return pd.DataFrame()
        
        # Ensure merge columns exist in both dataframes
        common_cols = [col for col in on if col in climate_df.columns and col in insurance_df.columns]
        
        if not common_cols:
            logger.error("No common columns to merge on")
            return pd.DataFrame()
        
        # Perform the merge
        merged_df = pd.merge(
            climate_df, 
            insurance_df, 
            on=common_cols,
            how='outer',
            suffixes=('_climate', '_insurance')
        )
        
        logger.info(f"Merged dataset has {len(merged_df)} rows")
        return merged_df
    
    def _handle_missing_values(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Handle missing values based on data type.
        
        Args:
            df: Input dataframe
            data_type: Type of data ('climate' or 'insurance')
            
        Returns:
            Dataframe with handled missing values
        """
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        if data_type == 'climate':
            # For climate data, use forward fill for time series within each region
            if 'region' in df_clean.columns:
                df_clean = df_clean.groupby('region').apply(
                    lambda x: x.ffill().bfill()
                ).reset_index(drop=True)
            
            # Fill any remaining NAs with column means
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isna().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
        elif data_type == 'insurance':
            # For insurance data, fill with zeros or means as appropriate
            if 'claims_count' in df_clean.columns:
                df_clean['claims_count'] = df_clean['claims_count'].fillna(0)
            
            if 'avg_claim_amount' in df_clean.columns:
                mean_claim = df_clean['avg_claim_amount'].mean()
                df_clean['avg_claim_amount'] = df_clean['avg_claim_amount'].fillna(mean_claim)
            
            if 'total_claim_amount' in df_clean.columns:
                if 'claims_count' in df_clean.columns and 'avg_claim_amount' in df_clean.columns:
                    mask = df_clean['total_claim_amount'].isna()
                    df_clean.loc[mask, 'total_claim_amount'] = (
                        df_clean.loc[mask, 'claims_count'] * 
                        df_clean.loc[mask, 'avg_claim_amount']
                    )
        
        return df_clean
    
    def _handle_outliers(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """Handle outliers in the specified columns.
        
        Args:
            df: Input dataframe
            columns: List of column names to process
            method: Method to use for outlier detection ('zscore' or 'iqr')
            
        Returns:
            Dataframe with handled outliers
        """
        if df.empty or not columns:
            return df
            
        df_clean = df.copy()
        columns = [col for col in columns if col in df_clean.columns]
        
        if method == 'zscore':
            for col in columns:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    mask = z_scores > self.config['outlier_threshold']
                    
                    if mask.any():
                        logger.info(f"Found {mask.sum()} outliers in column '{col}'")
                        # Cap outliers at threshold
                        upper_bound = df_clean[col].mean() + self.config['outlier_threshold'] * df_clean[col].std()
                        lower_bound = df_clean[col].mean() - self.config['outlier_threshold'] * df_clean[col].std()
                        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        elif method == 'iqr':
            for col in columns:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    q1 = df_clean[col].quantile(0.25)
                    q3 = df_clean[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                    if mask.any():
                        logger.info(f"Found {mask.sum()} outliers in column '{col}'")
                        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        return df_clean
