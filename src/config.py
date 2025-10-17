# Parameters, paths, constants
import os
from dataclasses import dataclass

@dataclass
class Config:
    # API keys: fill or set as env vars
    ALPHA_VANTAGE_KEY: str = os.getenv('ALPHAVANTAGE_API_KEY', 'YOUR_ALPHA_VANTAGE_KEY')
    NASDAQ_DATALINK_KEY: str = os.getenv('NASDAQ_DATALINK_KEY', 'YOUR_NASDAQ_DATALINK_KEY')
    FRED_API_KEY: str = os.getenv('FRED_API_KEY', 'YOUR_FRED_API_KEY')

    # Storage
    RAW_DATA_DIR: str = os.getenv('RAW_DATA_DIR', './data/raw')
    PROCESSED_DIR: str = os.getenv('PROCESSED_DIR', './data/processed')

    # Defaults
    START_DATE: str = os.getenv('START_DATE', '2000-01-01')
    END_DATE: str = os.getenv('END_DATE', None)  # None means up to today

config = Config()
print(config.FRED_API_KEY)
