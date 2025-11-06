"""
Data loading utilities for ClimateRisk360.

This module provides functions to load datasets from various sources including
local files, remote URLs, and APIs.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional
import requests
from datetime import datetime
import json

# Default data directory
DEFAULT_DATA_DIR = Path("data")

class DataLoader:
    """A class to handle loading of climate and insurance datasets."""
    
    def __init__(self, data_dir: Union[str, Path] = None):
        """Initialize the data loader with a data directory.
        
        Args:
            data_dir: Directory where data files are stored. If None, uses 'data/' in the project root.
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Default file names
        self.default_files = {
            'climate_data': 'climate_data.parquet',
            'insurance_data': 'insurance_data.parquet',
            'regions': 'regions.geojson'
        }
    
    def load_climate_data(self, source: str = None, **kwargs) -> pd.DataFrame:
        """Load climate data from a source.
        
        Args:
            source: Path to the data source. If None, uses the default location.
            **kwargs: Additional arguments passed to the loading function.
            
        Returns:
            DataFrame containing the climate data.
        """
        if source is None:
            source = self.data_dir / self.default_files['climate_data']
        
        if str(source).startswith(('http://', 'https://')):
            return self._load_from_url(source, **kwargs)
        else:
            return self._load_local_file(source, **kwargs)
    
    def load_insurance_data(self, source: str = None, **kwargs) -> pd.DataFrame:
        """Load insurance data from a source.
        
        Args:
            source: Path to the data source. If None, uses the default location.
            **kwargs: Additional arguments passed to the loading function.
            
        Returns:
            DataFrame containing the insurance data.
        """
        if source is None:
            source = self.data_dir / self.default_files['insurance_data']
        
        if str(source).startswith(('http://', 'https://')):
            return self._load_from_url(source, **kwargs)
        else:
            return self._load_local_file(source, **kwargs)
    
    def _load_local_file(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from a local file.
        
        Args:
            filepath: Path to the file to load.
            **kwargs: Additional arguments passed to pandas read function.
            
        Returns:
            DataFrame containing the loaded data.
        """
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.data_dir / filepath
        
        suffix = filepath.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(filepath, **kwargs)
        elif suffix == '.parquet':
            return pd.read_parquet(filepath, **kwargs)
        elif suffix == '.xlsx' or suffix == '.xls':
            return pd.read_excel(filepath, **kwargs)
        elif suffix == '.json':
            return pd.read_json(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_from_url(self, url: str, **kwargs) -> pd.DataFrame:
        """Load data from a URL.
        
        Args:
            url: URL to load the data from.
            **kwargs: Additional arguments passed to pandas read function.
            
        Returns:
            DataFrame containing the loaded data.
        """
        # Check if we should cache the file
        cache_file = self.data_dir / f"cache_{hash(url)}.parquet"
        
        # If file is already cached and less than 1 day old, use the cache
        if cache_file.exists():
            file_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if file_age < 86400:  # 1 day in seconds
                return self._load_local_file(cache_file)
        
        # Otherwise download and cache the file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Determine file type from URL or content type
        content_type = response.headers.get('content-type', '')
        if 'csv' in content_type or url.endswith('.csv'):
            df = pd.read_csv(pd.compat.StringIO(response.text), **kwargs)
        elif 'parquet' in content_type or url.endswith(('.parquet', '.parq')):
            import io
            df = pd.read_parquet(io.BytesIO(response.content), **kwargs)
        elif 'excel' in content_type or url.endswith(('.xlsx', '.xls')):
            import io
            df = pd.read_excel(io.BytesIO(response.content), **kwargs)
        elif 'json' in content_type or url.endswith('.json'):
            df = pd.read_json(response.text, **kwargs)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Cache the result
        df.to_parquet(cache_file)
        return df
    
    def load_sample_data(self, dataset: str = 'climate') -> pd.DataFrame:
        """Load sample data for testing and development.
        
        Args:
            dataset: Type of sample data to load ('climate' or 'insurance').
            
        Returns:
            DataFrame containing the sample data.
        """
        if dataset == 'climate':
            # Generate sample climate data
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
            regions = ['North', 'South', 'East', 'West']
            
            data = {
                'date': np.tile(dates, len(regions)),
                'region': np.repeat(regions, len(dates)),
                'avg_temperature': np.random.normal(20, 5, len(dates) * len(regions)),
                'total_rainfall': np.random.gamma(2, 10, len(dates) * len(regions)),
                'flood_events': np.random.poisson(0.2, len(dates) * len(regions)),
                'storm_events': np.random.poisson(0.1, len(dates) * len(regions)),
            }
            
            return pd.DataFrame(data)
            
        elif dataset == 'insurance':
            # Generate sample insurance data
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
            regions = ['North', 'South', 'East', 'West']
            
            data = {
                'date': np.tile(dates, len(regions)),
                'region': np.repeat(regions, len(dates)),
                'policies_in_force': np.random.randint(1000, 10000, len(dates) * len(regions)),
                'claims_count': np.random.poisson(50, len(dates) * len(regions)),
                'total_claims': np.random.gamma(10000, 10, len(dates) * len(regions)),
                'avg_claim_amount': np.random.normal(2000, 500, len(dates) * len(regions)),
            }
            
            return pd.DataFrame(data)
            
        else:
            raise ValueError(f"Unknown dataset: {dataset}")


def get_data_loader(data_dir: Union[str, Path] = None) -> DataLoader:
    """Get a configured DataLoader instance.
    
    Args:
        data_dir: Directory where data files are stored. If None, uses 'data/' in the project root.
        
    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(data_dir)
