# file: data_utils.py

import os
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

# Configure logging

LOG_DIR = "./logs"
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=Path(LOG_DIR) / "data_pipeline.log",
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

# Utility functions

def ensure_dirs(*dirs):
    """Ensure directories exist; create them if not."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logging.debug(f"Ensured directory exists: {d}")


def save_parquet(df: pd.DataFrame, path: str, partition_cols: Optional[list] = None):
    """Save a DataFrame to a Parquet file."""
    try:
        ensure_dirs(Path(path).parent)
        df.to_parquet(path, index=True)
        logging.info(f"Saved DataFrame to Parquet: {path}")
    except Exception as e:
        logging.error(f"Error saving Parquet {path}: {e}")
        raise


def save_csv(df: pd.DataFrame, path: str):
    """Save a DataFrame to a CSV file."""
    try:
        ensure_dirs(Path(path).parent)
        df.to_csv(path, index=True)
        logging.info(f"Saved DataFrame to CSV: {path}")
    except Exception as e:
        logging.error(f"Error saving CSV {path}: {e}")
        raise


def read_parquet(path: str) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    try:
        df = pd.read_parquet(path)
        logging.info(f"Loaded Parquet file: {path}")
        return df
    except Exception as e:
        logging.error(f"Error reading Parquet {path}: {e}")
        raise


def read_csv(path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(path, parse_dates=parse_dates)
        logging.info(f"Loaded CSV file: {path}")
        return df
    except Exception as e:
        logging.error(f"Error reading CSV {path}: {e}")
        raise


def load_data(path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """
    Load data automatically based on file extension.
    Supports CSV and Parquet.
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return read_csv(path, parse_dates=parse_dates)
    elif ext in [".parquet", ".parq"]:
        return read_parquet(path)
    else:
        logging.error(f"Unsupported file type: {ext}")
        raise ValueError(f"Unsupported file type: {ext}")



if __name__ == "__main__":
    # Example DataFrame
    df_example = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=5),
        "Value": [10, 20, 30, 40, 50]
    })

    # Paths
    csv_path = "./data/processed/example.csv"
    parquet_path = "./data/processed/example.parquet"

    # Save
    save_csv(df_example, csv_path)
    save_parquet(df_example, parquet_path)

    # Load
    df_csv = load_data(csv_path, parse_dates=["Date"])
    df_parquet = load_data(parquet_path)

    print(df_csv)
    print(df_parquet)