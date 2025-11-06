"""Data ingestion module for ClimateRisk360.

This module handles loading data from various sources including CSV files and APIs.
"""
from pathlib import Path
from typing import Dict, Union
import pandas as pd
import requests
from datetime import datetime
import os

class DataIngestion:
    """Handles data loading from various sources."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """Initialize with the data directory path.
        
        Args:
            data_dir: Path to the directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.ensure_data_dir()
    
    def ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_insurance_data(self, filepath: Union[str, Path] = None) -> pd.DataFrame:
        """Load insurance claims data.
        
        Args:
            filepath: Path to the insurance data file. If None, uses default location.
            
        Returns:
            DataFrame containing insurance claims data
        """
        if filepath is None:
            filepath = self.data_dir / "insurance_claims.csv"
            
        if not Path(filepath).exists():
            # In a real application, this would load from your actual data source
            return self._generate_sample_insurance_data()
            
        return pd.read_csv(filepath)
    
    def load_climate_data(self, filepath: Union[str, Path] = None) -> pd.DataFrame:
        """Load climate data.
        
        Args:
            filepath: Path to the climate data file. If None, uses default location.
            
        Returns:
            DataFrame containing climate data
        """
        if filepath is None:
            filepath = self.data_dir / "climate_data.csv"
            
        if not Path(filepath).exists():
            # In a real application, this would load from your actual data source
            return self._generate_sample_climate_data()
            
        return pd.read_csv(filepath)
    
    def fetch_weather_api_data(self, 
                             location: str, 
                             start_date: str, 
                             end_date: str,
                             api_key: str = None) -> Dict:
        """Fetch weather data from a public API (example implementation).
        
        Args:
            location: Location to fetch data for (e.g., 'New York,US')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            api_key: API key for the weather service
            
        Returns:
            Dictionary containing the API response
        """
        # This is a placeholder for a real API call
        # Example using OpenWeatherMap (would require API key)
        base_url = "http://api.openweathermap.org/data/2.5/onecall/timemachine"
        
        params = {
            'lat': 0,  # Would be calculated from location
            'lon': 0,  # Would be calculated from location
            'dt': int(datetime.now().timestamp()),
            'appid': api_key or os.getenv('OPENWEATHER_API_KEY')
        }
        
        # In a real implementation, we would make the API call here
        # response = requests.get(base_url, params=params)
        # return response.json()
        
        # For now, return sample data
        return {
            'lat': 0,
            'lon': 0,
            'timezone': 'UTC',
            'current': {
                'dt': int(datetime.now().timestamp()),
                'temp': 25.5,
                'humidity': 65,
                'wind_speed': 3.5,
                'weather': [{'main': 'Clear', 'description': 'clear sky'}]
            }
        }
    
    def _generate_sample_insurance_data(self) -> pd.DataFrame:
        """Generate sample insurance claims data for demonstration."""
        import numpy as np
        
        np.random.seed(42)
        regions = [f"Region_{i}" for i in range(1, 6)]
        years = [2020, 2021, 2022, 2023]
        
        data = []
        for year in years:
            for region in regions:
                base_claims = np.random.randint(5, 20)
                base_amount = np.random.uniform(2000, 10000)
                
                for month in range(1, 13):
                    claims = max(0, int(np.random.normal(base_claims, 3)))
                    avg_amount = max(1000, base_amount * np.random.uniform(0.8, 1.2))
                    
                    data.append({
                        'region': region,
                        'year': year,
                        'month': month,
                        'claims_count': claims,
                        'total_claim_amount': claims * avg_amount,
                        'avg_claim_amount': avg_amount,
                        'policy_count': np.random.randint(1000, 5000),
                    })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        return df
    
    def _generate_sample_climate_data(self) -> pd.DataFrame:
        """Generate sample climate data for demonstration."""
        import numpy as np
        
        np.random.seed(42)
        regions = [f"Region_{i}" for i in range(1, 6)]
        years = [2020, 2021, 2022, 2023]
        
        data = []
        for year in years:
            for region in regions:
                base_temp = 15 + (hash(region) % 10)
                base_rain = 50 + (hash(region) % 100)
                
                for month in range(1, 13):
                    # Seasonal variation
                    temp_variation = 10 * np.sin(2 * np.pi * (month - 1) / 12)
                    rain_variation = 50 * np.sin(2 * np.pi * (month - 1) / 12 - np.pi/2)
                    
                    # Climate change trend
                    year_factor = (year - 2020) * 0.3
                    
                    # Random noise
                    temp_noise = np.random.normal(0, 2)
                    rain_noise = np.random.normal(0, 10)
                    
                    # Calculate values
                    avg_temp = base_temp + temp_variation + year_factor + temp_noise
                    total_rain = max(0, base_rain + rain_variation + year_factor * 5 + rain_noise)
                    
                    data.append({
                        'region': region,
                        'year': year,
                        'month': month,
                        'avg_temperature': round(avg_temp, 1),
                        'total_rainfall': round(total_rain, 1),
                        'max_temperature': round(avg_temp + np.random.uniform(3, 8), 1),
                        'min_temperature': round(avg_temp - np.random.uniform(3, 8), 1),
                        'flood_events': int(np.random.poisson(0.5)),
                        'storm_events': int(np.random.poisson(0.3)),
                    })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        return df
