"""
Script to download and process climate and insurance datasets from various sources.
"""
import os
import pandas as pd
import requests
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_worldbank_climate_data():
    """Download climate data from World Bank Climate Data API."""
    print("Downloading climate data from World Bank...")
    
    # World Bank API endpoint for climate data
    base_url = "https://climateknowledgeportal.worldbank.org/api/data/get-download-data"
    
    # Parameters for the API request
    params = {
        'model': 'mohc_hadgem2_es',
        'gcm': 'rcp8_5',
        'variable': 'tas',  # Temperature
        'from': '1980',
        'to': '2023',
        'country': 'USA',  # Example country
        'type': 'mavg',   # Monthly average
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Save raw data
        climate_file = DATA_DIR / "worldbank_climate_data.csv"
        with open(climate_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Process the data
        df = pd.read_csv(climate_file)
        
        # Basic cleaning and transformation
        df = df.rename(columns={
            'Year': 'year',
            'Value': 'temperature_change',
            'ISO3': 'country_code',
            'Country': 'country'
        })
        
        # Save processed data
        processed_file = DATA_DIR / "processed_climate_data.parquet"
        df.to_parquet(processed_file)
        print(f"Climate data saved to {processed_file}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading climate data: {e}")
        return None

def download_noaa_weather_data():
    """Download historical weather data from NOAA's NCEI API."""
    print("Downloading weather data from NOAA NCEI...")
    
    # You'll need to get an API token from https://www.ncdc.noaa.gov/cdo-web/token
    # For now, we'll use a sample dataset
    try:
        # Example: Using a sample dataset from NOAA
        url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/1/1/1850-2023/data.json"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Process the data
        df = pd.DataFrame({
            'year': [int(k) for k in data['data'].keys()],
            'temperature_anomaly': [float(v) for v in data['data'].values()]
        })
        
        # Save the data
        weather_file = DATA_DIR / "noaa_global_temps.parquet"
        df.to_parquet(weather_file)
        print(f"Weather data saved to {weather_file}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading weather data: {e}")
        return None

def download_insurance_data():
    """Generate sample insurance data since real insurance data is often proprietary."""
    print("Generating sample insurance data...")
    
    try:
        # Generate sample data for insurance claims
        np.random.seed(42)
        
        # Create date range for the last 10 years
        dates = pd.date_range(end=datetime.now(), periods=120, freq='M')
        
        # Create regions
        regions = ['North', 'South', 'East', 'West']
        
        # Generate data
        data = []
        for date in dates:
            for region in regions:
                base_claims = np.random.poisson(50)
                weather_factor = 1.0
                
                # Simulate more claims in certain conditions
                if date.month in [6, 7, 8]:  # Summer months
                    weather_factor *= 1.3
                if region in ['South', 'East']:  # Regions with more extreme weather
                    weather_factor *= 1.2
                
                claims = int(base_claims * weather_factor)
                avg_claim = np.random.normal(5000, 1000)
                
                data.append({
                    'date': date,
                    'region': region,
                    'claims_count': max(0, int(claims + np.random.normal(0, 5))),
                    'avg_claim_amount': max(1000, avg_claim),
                    'total_claims': max(1000, claims * avg_claim)
                })
        
        df = pd.DataFrame(data)
        
        # Save the data
        insurance_file = DATA_DIR / "sample_insurance_data.parquet"
        df.to_parquet(insurance_file)
        print(f"Sample insurance data saved to {insurance_file}")
        
        return df
        
    except Exception as e:
        print(f"Error generating insurance data: {e}")
        return None

def main():
    """Main function to download all datasets."""
    print("Starting dataset download...")
    
    # Download climate data
    climate_df = download_worldbank_climate_data()
    
    # Download weather data
    weather_df = download_noaa_weather_data()
    
    # Generate insurance data
    insurance_df = download_insurance_data()
    
    print("\nDataset download complete!")
    
    # Print dataset summaries
    if climate_df is not None:
        print(f"\nClimate data shape: {climate_df.shape}")
        print(climate_df.head())
    
    if weather_df is not None:
        print(f"\nWeather data shape: {weather_df.shape}")
        print(weather_df.head())
    
    if insurance_df is not None:
        print(f"\nSample insurance data shape: {insurance_df.shape}")
        print(insurance_df.head())

if __name__ == "__main__":
    main()
