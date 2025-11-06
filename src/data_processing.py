"""Data processing module for ClimateRisk360.

This module handles data transformations, feature engineering, and aggregations.
"""
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data transformations and feature engineering."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data processor with optional configuration.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {
            'rolling_window': 3,  # Months for rolling averages
            'lag_periods': [1, 3, 6, 12],  # Lag periods in months
            'risk_weights': {
                'temperature': 0.3,
                'rainfall': 0.4,
                'flood_events': 0.3
            },
            'scaler': 'standard'  # 'standard' or 'minmax'
        }
        
        # Initialize scaler
        self.scaler = StandardScaler() if self.config['scaler'] == 'standard' else MinMaxScaler()
        self.scaler_fitted = False
    
    def process_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the merged climate and insurance data.
        
        Args:
            df: Merged dataframe from DataCleaner
            
        Returns:
            Processed dataframe with features and targets
        """
        if df.empty:
            logger.warning("Empty dataframe provided for processing")
            return df
            
        logger.info(f"Processing merged data with {len(df)} rows")
        
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Ensure date is in datetime format
        if 'date' not in processed_df.columns and all(col in processed_df.columns for col in ['year', 'month']):
            processed_df['date'] = pd.to_datetime(
                processed_df[['year', 'month']].assign(day=1)
            )
        
        # Sort by region and date
        sort_cols = ['region', 'date'] if 'date' in processed_df.columns else ['region', 'year', 'month']
        processed_df = processed_df.sort_values(sort_cols).reset_index(drop=True)
        
        # Add time-based features
        processed_df = self._add_time_features(processed_df)
        
        # Add rolling statistics
        processed_df = self._add_rolling_features(processed_df)
        
        # Add lagged features
        processed_df = self._add_lagged_features(processed_df)
        
        # Add derived features
        processed_df = self._add_derived_features(processed_df)
        
        # Calculate risk scores
        processed_df = self._calculate_risk_scores(processed_df)
        
        # Scale features
        processed_df = self._scale_features(processed_df)
        
        logger.info(f"Processed data shape: {processed_df.shape}")
        return processed_df
    
    def aggregate_to_region_level(self, df: pd.DataFrame, freq: str = 'Q') -> pd.DataFrame:
        """Aggregate data to region level with the specified frequency.
        
        Args:
            df: Processed dataframe
            freq: Aggregation frequency ('M' for monthly, 'Q' for quarterly, 'Y' for yearly)
            
        Returns:
            Aggregated dataframe
        """
        if df.empty:
            return df
            
        if 'date' not in df.columns:
            logger.error("Cannot aggregate: 'date' column not found")
            return df
            
        # Make a copy and set date as index
        agg_df = df.copy()
        agg_df = agg_df.set_index('date')
        
        # Define aggregation functions
        agg_funcs = {
            'avg_temperature': 'mean',
            'total_rainfall': 'sum',
            'flood_events': 'sum',
            'storm_events': 'sum',
            'claims_count': 'sum',
            'total_claim_amount': 'sum',
            'avg_claim_amount': 'mean',
            'policy_count': 'sum',
            'claim_frequency': 'mean',
            'risk_score': 'mean'
        }
        
        # Filter to only include columns that exist in the dataframe
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in agg_df.columns}
        
        # Group by region and resample
        grouped = agg_df.groupby('region').resample(freq)
        
        # Apply aggregation
        agg_df = grouped.agg(agg_funcs).reset_index()
        
        # Calculate derived metrics after aggregation
        if 'claims_count' in agg_df.columns and 'policy_count' in agg_df.columns:
            agg_df['claim_frequency'] = agg_df['claims_count'] / agg_df['policy_count']
        
        if 'total_claim_amount' in agg_df.columns and 'claims_count' in agg_df.columns:
            mask = agg_df['claims_count'] > 0
            agg_df.loc[mask, 'avg_claim_amount'] = (
                agg_df.loc[mask, 'total_claim_amount'] / 
                agg_df.loc[mask, 'claims_count']
            )
        
        return agg_df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataframe."""
        if 'date' not in df.columns:
            return df
            
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Add season based on month
        df['season'] = df['month'].apply(
            lambda m: (m % 12 + 3) // 3  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
        )
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics features."""
        if 'region' not in df.columns or 'date' not in df.columns:
            return df
            
        df = df.copy()
        window = self.config['rolling_window']
        
        # Define columns for rolling stats
        roll_cols = [
            'avg_temperature', 'total_rainfall', 'flood_events',
            'claims_count', 'total_claim_amount'
        ]
        roll_cols = [col for col in roll_cols if col in df.columns]
        
        # Sort by region and date
        df = df.sort_values(['region', 'date'])
        
        # Calculate rolling statistics for each region
        for col in roll_cols:
            df[f'{col}_rolling_mean'] = df.groupby('region')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'{col}_rolling_std'] = df.groupby('region')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            # Calculate z-score within each region
            df[f'{col}_zscore'] = df.groupby('region')[col].transform(
                lambda x: (x - x.rolling(window=window, min_periods=1).mean()) / 
                         x.rolling(window=window, min_periods=1).std().replace(0, 1)
            )
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for time series analysis."""
        if 'region' not in df.columns or 'date' not in df.columns:
            return df
            
        df = df.copy().sort_values(['region', 'date'])
        
        # Define columns for lagging
        lag_cols = [
            'avg_temperature', 'total_rainfall', 'flood_events',
            'claims_count', 'total_claim_amount'
        ]
        lag_cols = [col for col in lag_cols if col in df.columns]
        
        # Add lagged features for each period
        for period in self.config['lag_periods']:
            for col in lag_cols:
                df[f'{col}_lag{period}'] = df.groupby('region')[col].shift(period)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataframe."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Add temperature anomaly (deviation from region's historical average)
        if 'avg_temperature' in df.columns:
            df['temp_anomaly'] = df.groupby('region')['avg_temperature'].transform(
                lambda x: x - x.mean()
            )
        
        # Add rainfall anomaly
        if 'total_rainfall' in df.columns:
            df['rainfall_anomaly'] = df.groupby('region')['total_rainfall'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        
        # Add claim severity (average claim amount)
        if 'total_claim_amount' in df.columns and 'claims_count' in df.columns:
            mask = df['claims_count'] > 0
            df['claim_severity'] = 0
            df.loc[mask, 'claim_severity'] = (
                df.loc[mask, 'total_claim_amount'] / 
                df.loc[mask, 'claims_count']
            )
        
        # Add claim frequency (claims per policy)
        if 'claims_count' in df.columns and 'policy_count' in df.columns:
            df['claim_frequency'] = df['claims_count'] / df['policy_count'].replace(0, np.nan)
        
        return df
    
    def _calculate_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate climate risk scores."""
        if df.empty:
            return df
            
        df = df.copy()
        weights = self.config['risk_weights']
        
        # Initialize risk score
        df['risk_score'] = 0.0
        
        # Add temperature component
        if 'avg_temperature' in df.columns and 'temperature' in weights:
            temp_std = df['avg_temperature'].std()
            if temp_std > 0:
                df['temp_risk'] = (
                    (df['avg_temperature'] - df['avg_temperature'].mean()) / temp_std
                ).abs()
                df['risk_score'] += df['temp_risk'] * weights['temperature']
        
        # Add rainfall component
        if 'total_rainfall' in df.columns and 'rainfall' in weights:
            rain_std = df['total_rainfall'].std()
            if rain_std > 0:
                df['rain_risk'] = (
                    (df['total_rainfall'] - df['total_rainfall'].mean()) / rain_std
                ).abs()
                df['risk_score'] += df['rain_risk'] * weights['rainfall']
        
        # Add flood component
        if 'flood_events' in df.columns and 'flood_events' in weights:
            flood_std = df['flood_events'].std()
            if flood_std > 0:
                df['flood_risk'] = (
                    (df['flood_events'] - df['flood_events'].mean()) / flood_std
                )
                df['risk_score'] += df['flood_risk'] * weights['flood_events']
        
        # Scale risk score to 0-100
        if 'risk_score' in df.columns:
            df['risk_score'] = (
                (df['risk_score'] - df['risk_score'].min()) / 
                (df['risk_score'].max() - df['risk_score'].min() + 1e-10) * 100
            ).clip(0, 100).round(2)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Select numerical columns to scale
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove columns that shouldn't be scaled
        exclude_cols = [
            'year', 'month', 'day', 'quarter', 'season', 'day_of_year',
            'claims_count', 'total_claim_amount', 'policy_count',
            'flood_events', 'storm_events', 'risk_score'
        ]
        
        scale_cols = [col for col in numeric_cols if col not in exclude_cols and 
                     not any(x in col for x in ['_lag', '_rolling', '_zscore', 'anomaly'])]
        
        if not scale_cols:
            return df
        
        # Fit scaler if not already fitted
        if not self.scaler_fitted:
            self.scaler.fit(df[scale_cols])
            self.scaler_fitted = True
        
        # Scale features
        df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        return df
