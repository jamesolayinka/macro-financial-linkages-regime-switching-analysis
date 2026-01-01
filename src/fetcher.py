import numpy as np
import pandas as pd
import logging

import yfinance as yf
from fredapi import Fred
import nasdaqdatalink
from alpha_vantage.timeseries import TimeSeries

from pandas_datareader import data as pdr
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")
from typing import List, Optional
from config import config

import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:

    def __init__(self, start: str="2003-01-01", end: str=datetime.now().strftime('%Y-%m-%d'), interval: str = '1d'):
        self.start = start
        self.end = end
        self.interval = interval
        self.fred_api_key = config.FRED_API_KEY
        self.alpha_vantage_api_key = config.ALPHA_VANTAGE_KEY
        self.nasdaq_api_key = config.NASDAQ_DATALINK_KEY

        self.assets = {"equities": ['SPY', 'QQQ', 'DIA', 'IWM'],
                       "commodities": ['GLD', 'SLV', 'USO', 'DBC'],
                       "sectors": ['XLE', 'XLF', 'XLI', 'XLK', 'XLV']}
        
        self.macro_series = {
            "GDP": "Gross Domestic Product",
            "UNRATE": "Unemployment Rate",
            "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
            "FEDFUNDS": "Effective Federal Funds Rate",
            "DGS10": "10-Year Treasury Constant Maturity Rate",
            "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate (WTI)",
            "INDPRO": "Industrial Production Index",
            "UMCSENT": "University of Michigan Consumer Sentiment Index",
            "VIXCLS": "CBOE Volatility Index"
        }
    
    def get_asset_prices(self) -> pd.DataFrame:
        """
        Download OHLCV for multiple tickers and return a multiindex (Date, Ticker) DataFrame of adjusted close and volume.
        """
        all_tickers = []
        for category in self.assets:
            all_tickers.extend(self.assets[category])
        
        logger.info(f"Downloading {len(all_tickers)} assets from {self.start} to {self.end}")
   
        try:
            raw = yf.download(all_tickers, start=self.start, end=self.end, interval=self.interval, group_by='ticker', threads=True)
        except Exception as e:
            logger.error(f"Failed to download asset prices via yfinance: {e}")
            return pd.DataFrame()

        # If single ticker, normalize to multiindex
        if len(all_tickers) == 1:
            df = raw.copy()
            df.columns = pd.MultiIndex.from_product([[all_tickers[0]], df.columns])

        # Build adjusted close matrix
        adj_close = pd.DataFrame()
        close = pd.DataFrame()
        vol = pd.DataFrame()

        for t in all_tickers:
            try:
                adj = raw[t]['Adj Close'] if isinstance(raw.columns, pd.MultiIndex) else raw['Adj Close']
                cl = raw[t]['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
                v = raw[t]['Volume'] if isinstance(raw.columns, pd.MultiIndex) else raw['Volume']
            except KeyError:
                logger.warning(f"Ticker {t} not found in primary download, attempting fallback...")
                try:
                    tk = yf.Ticker(t)
                    hist = tk.history(start=self.start, end=self.end, interval=self.interval)
                    if hist.empty:
                        logger.error(f"No data found for {t} even during fallback.")
                        continue
                    adj = hist['Close'] * (hist.get('Adj Close', hist['Close']) / hist['Close'])
                    cl = hist['Close']
                    v = hist.get('Volume', pd.Series(index=hist.index))
                except Exception as ex:
                    logger.error(f"Error during fallback for {t}: {ex}")
                    continue
            except Exception as e:
                logger.error(f"Unexpected error processing {t}: {e}")
                continue

            adj_close[t] = adj
            close[t] = cl
            vol[t] = v

        # Combine into a long-format DataFrame
        if not adj_close.empty:
            adj_close.index = pd.to_datetime(adj_close.index).tz_localize(None).normalize()
            logger.info(f"Successfully processed {len(adj_close.columns)} assets with {len(adj_close)} records each.")
        else:
            logger.error("No asset price data collected.")
            
        return adj_close.sort_index()
    
    def get_macro_data(self) -> pd.DataFrame:
        if not self.fred_api_key:
            logger.error("FRED API key is missing. Skipping macro data collection.")
            return pd.DataFrame()

        try:
            fred = Fred(api_key=self.fred_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize FRED API: {e}")
            return pd.DataFrame()

        macro_df = pd.DataFrame()
        for fred_code, name in self.macro_series.items():
            try:
                series = fred.get_series(fred_code, observation_start=self.start, observation_end=self.end)
                series.name = fred_code
                macro_df = pd.concat([macro_df, series], axis=1)
                logger.debug(f"Fetched macro series: {fred_code}")
            except Exception as e:
                logger.error(f"Error fetching {fred_code} from FRED: {e}")
        
        if not macro_df.empty:
            logger.info(f"Downloaded {macro_df.shape[1]} macroeconomic indicators with {len(macro_df)} records.")
            macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).normalize()
        else:
            logger.warning("No macroeconomic data was collected.")
            
        return macro_df.sort_index()

    def fetch_nasdaq_dataset(self) -> pd.DataFrame:
        if not self.nasdaq_api_key:
            logger.warning("Nasdaq API key is missing. Skipping Nasdaq datasets.")
            return pd.DataFrame()

        nasdaqdatalink.ApiConfig.api_key = self.nasdaq_api_key
        
        combined_df = pd.DataFrame()
        try:
            # Note: _fetch_nasdaq_dataset was previously called but not clearly defined in the snippet or it was a self call
            # Re-implementing with direct nasdaqdatalink call for known datasets
            for dataset_code in ['CHRIS/CME_CL1', 'CHRIS/CME_GC1', 'CHRIS/CME_SI1']:
                try:
                    df = nasdaqdatalink.get(dataset_code, start_date=self.start, end_date=self.end)
                    if not df.empty:
                        logger.info(f"Fetched Nasdaq dataset {dataset_code} with {len(df)} records.")
                        # This logic might need refinement if you want to join them
                        combined_df = pd.concat([combined_df, df], axis=1)
                except Exception as inner_e:
                    logger.error(f"Error fetching {dataset_code} from Nasdaq: {inner_e}")
                    
            return combined_df.sort_index()
        except Exception as e:
            logger.error(f"Critical error in fetch_nasdaq_dataset: {e}")
            return pd.DataFrame()

    def preprocess_data(self, prices, macro_df):
        if prices.empty or macro_df.empty:
            logger.warning("Empty dataframes passed to preprocess_data. Skipping.")
            return {}

        logger.info("Beginning data preprocessing...")
        try:
            macro_daily = macro_df.reindex(prices.index, method='ffill').ffill().bfill()

            returns = prices.pct_change().dropna()
            log_returns = np.log(prices/prices.shift(1)).dropna()

            macro_daily_changes = macro_daily.pct_change().dropna()

            data = {
                "prices": prices,
                "returns": returns,
                "log_returns": log_returns,
                "macro": macro_daily,
                "macro_changes": macro_daily_changes
            }
            logger.info("Preprocessing complete.")
            return data
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return {}
    
    def get_events_dates(self):
        # Placeholder for event dates fetching logic
        events = {
            "FOMC": [], # List of FOMC meeting dates
            "NFP": [],  # List of CPI release dates
            "Earnings": [], # List of major earnings dates
            "CPI": [] # Non-farm payrolls
        }
        return events
    
    def run_full_collection(self):
        logger.info("=== Starting Data Collection Pipeline ===")
        
        prices = self.get_asset_prices()
        macro_data = self.get_macro_data()
        
        if prices.empty:
            logger.error("Asset price collection failed. Data collection aborted.")
            return {}

        data = self.preprocess_data(prices, macro_data)
        return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fetch and preprocess financial and macroeconomic data.")
    parser.add_argument('--start', type=str, default="2000-01-01", help='Start date for data fetching in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, default="2025-12-31", help='End date for data fetching in YYYY-MM-DD format')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d, 1wk, 1mo)')

    args = parser.parse_args()

    collector = DataCollector(start=args.start, end=args.end, interval=args.interval)
    data = collector.run_full_collection()

    if data:
        import os
        
        # 1. Save individual ticker CSVs in data/raw/{category}/
        logger.info("Extracting and saving individual ticker CSVs...")
        prices = data['prices']
        for category, tickers in collector.assets.items():
            category_dir = os.path.join("data", "raw", category)
            os.makedirs(category_dir, exist_ok=True)
            for ticker in tickers:
                if ticker in prices.columns:
                    ticker_path = os.path.join(category_dir, f"{ticker}.csv")
                    prices[ticker].to_csv(ticker_path)
                    logger.debug(f"Saved {ticker} -> {ticker_path}")
        
        # 2. Save individual macro series in data/raw/macro/
        logger.info("Extracting and saving individual macro CSVs...")
        macro_all = data['macro']
        macro_dir = os.path.join("data", "raw", "macro")
        os.makedirs(macro_dir, exist_ok=True)
        for fred_code in collector.macro_series.keys():
            if fred_code in macro_all.columns:
                macro_path = os.path.join(macro_dir, f"{fred_code}.csv")
                macro_all[fred_code].to_csv(macro_path)
                logger.debug(f"Saved {fred_code} -> {macro_path}")

        # 3. Save consolidated/processed data
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        try:
            data['prices'].to_csv(os.path.join(config.PROCESSED_DIR, "consolidated_prices.csv"))
            data['macro'].to_csv(os.path.join(config.PROCESSED_DIR, "consolidated_macro.csv"))
            logger.info(f"Consolidated data saved to {config.PROCESSED_DIR}")
        except Exception as e:
            logger.error(f"Failed to save consolidated data: {e}")

        # Quick tail check
        logger.info(f"\nLast 5 days of GLD (Commodity):\n{prices['GLD'].tail() if 'GLD' in prices.columns else 'N/A'}")
        logger.info(f"\nLast 5 observations of GDP (Macro):\n{macro_all['GDP'].tail() if 'GDP' in macro_all.columns else 'N/A'}")
    else:
        logger.error("No data collected in this run.")

