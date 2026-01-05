import logging
import os
import argparse
from config import config
from fetcher import DataCollector
from preprocess import preprocess_macro_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(start_date: str, end_date: str):
    """
    Orchestrates the macro-financial data pipeline:
    1. Fetch asset and macro data.
    2. Save raw data to categorized directory structure.
    3. Preprocess and align macro data.
    """
    logger.info(f"Starting pipeline from {start_date} to {end_date}")

    # 1. Fetch Data
    collector = DataCollector(start=start_date, end=end_date)
    data = collector.run_full_collection()

    if not data:
        logger.error("Data collection failed. Aborting pipeline.")
        return

    # 2. Save Raw Data
    collector.save_data_to_disk(data)

    # 3. Preprocess Macro Data
    raw_macro_dir = os.path.join(config.RAW_DATA_DIR, "macro")
    output_macro_file = os.path.join(config.PROCESSED_DIR, "macro_processed.csv")
    
    logger.info("Starting preprocessing step...")
    preprocess_macro_data(raw_macro_dir, output_macro_file, start_date=start_date)

    logger.info("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate the macro-financial data pipeline.")
    parser.add_argument('--start', type=str, default=config.START_DATE, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=config.END_DATE, help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    run_pipeline(args.start, args.end)
