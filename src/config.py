# Parameters, paths, constants
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class Config:
    # API keys: fill or set as env vars
    ALPHA_VANTAGE_KEY: str = os.getenv('ALPHAVANTAGE_API_KEY')
    NASDAQ_DATALINK_KEY: str = os.getenv('NASDAQ_DATALINK_KEY')
    FRED_API_KEY: str = os.getenv('FRED_API_KEY')

    # Storage
    RAW_DATA_DIR: str = os.getenv('RAW_DATA_DIR', os.path.join(os.getcwd(), 'data', 'raw'))
    PROCESSED_DIR: str = os.getenv('PROCESSED_DIR', os.path.join(os.getcwd(), 'data', 'processed'))

    # Defaults
    START_DATE: str = os.getenv('START_DATE', '2000-01-01')
    END_DATE: str = os.getenv('END_DATE', '2025-12-31')

config = Config()
