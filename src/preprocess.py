import pandas as pd
import numpy as np
import config


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_df / price_df.shift(1)).dropna(how='all')


def align_and_forward_fill(dfs: dict) -> pd.DataFrame:
    """Given a dict of series/dataframes, produce a single DataFrame aligned on index with forward-fill for missing values."""
    merged = pd.concat(dfs.values(), axis=1)
    merged = merged.sort_index()
    # Basic cleaning: remove rows that are all NaN
    merged = merged.dropna(how='all')
    # For financial series, forward fill short gaps
    merged = merged.ffill().bfill()
    return merged

# ------------------------- storage.py -------------------------
import os
from pathlib import Path
import pandas as pd


def write_series_parquet(df: pd.DataFrame, name: str, base_dir: str = None):
    base = base_dir or config.RAW_DATA_DIR
    Path(base).mkdir(parents=True, exist_ok=True)
    path = Path(base) / f"{name}.parquet"
    df.to_parquet(path)
    print(f"Saved {name} -> {path}")