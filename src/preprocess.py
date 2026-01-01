import os
import glob
import pandas as pd
import numpy as np
import logging
from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_macro_data(raw_dir: str, output_path: str, start_date: str = "2000-01-01"):
    """
    Consolidate individual macro CSV files into a single, daily-aligned dataset.
    """
    logger.info(f"Starting macro preprocessing from directory: {raw_dir}")
    
    # Discover all CSV files in the raw macro directory
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {raw_dir}. Aborting.")
        return

    logger.info(f"Found {len(csv_files)} macro indicators to process.")
    
    all_series = []
    
    # Load each file and store as a series
    for file_path in csv_files:
        indicator_name = os.path.basename(file_path).replace(".csv", "")
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if df.empty:
                logger.warning(f"File {file_path} is empty. Skipping.")
                continue
            
            # Ensure it's a series or single column
            series = df.iloc[:, 0]
            series.name = indicator_name
            all_series.append(series)
            logger.debug(f"Loaded {indicator_name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    if not all_series:
        logger.error("No valid data loaded. Aborting.")
        return

    # Merge all series into a single DataFrame
    merged_df = pd.concat(all_series, axis=1)
    merged_df = merged_df.sort_index()
    
    # Create a daily business day index for alignment
    # End date should be today or the max date in the data
    end_date = merged_df.index.max()
    daily_index = pd.date_range(start=start_date, end=end_date, freq='B')
    
    logger.info(f"Aligning data to daily business frequency from {start_date} to {end_date.date()}")
    
    # Reindex and fill missing values
    processed_df = merged_df.reindex(daily_index)
    
    # Forward fill: carry over the last known value
    processed_df = processed_df.ffill()
    
    # Backward fill: for the very beginning of the series if they start late
    processed_df = processed_df.bfill()
    
    # Save final result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        processed_df.to_csv(output_path)
        logger.info(f"Successfully saved processed macro data to: {output_path}")
        logger.info(f"Final shape: {processed_df.shape}")
    except Exception as e:
        logger.error(f"Failed to save output to {output_path}: {e}")

    return processed_df

if __name__ == "__main__":
    # Define paths based on project structure
    RAW_MACRO_DIR = os.path.join("data", "raw", "macro")
    OUTPUT_FILE = os.path.join(config.PROCESSED_DIR, "macro_processed.csv")
    
    # Run the preprocessing
    df_processed = preprocess_macro_data(RAW_MACRO_DIR, OUTPUT_FILE)
    
    if df_processed is not None:
        # Show preview for the user (30 rows)
        logger.info(f"\nProcessed Macro Data Preview (First 30 rows):\n{df_processed.head(30).to_string()}")
        logger.info(f"\nProcessed Macro Data Preview (Last 30 rows):\n{df_processed.tail(30).to_string()}")
        
        # Determine project root dynamically (one level up from src/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Construct absolute path for saving
        abs_output = os.path.join(project_root, "data", "processed", "macro_processed.csv")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(abs_output), exist_ok=True)
        
        # Save the data
        df_processed.to_csv(abs_output)
        logger.info(f"Successfully saved processed macro data to: {abs_output}")
