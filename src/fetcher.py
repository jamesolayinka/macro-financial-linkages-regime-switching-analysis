import numpy as np
import pandas as pd

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

import argparse

class DataCollector:

    def __init__(self, start: str="2003-01-01", end: str=datetime.now().strftime('%Y-%m-%d'), interval: str = '1d'):
        self.start = start
        self.end = end
        self.interval = interval
        self.fred_api_key = "d66f326e45966a51f8799ede96b679e5"
        self.alpha_vantage_api_key = "2EI9MDGQKVDRD2X9"
        self.nasdaq_api_key = "vC4umV-rynDZj6mtcotw"

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
        
        print(f"Downloading {len(all_tickers)} assets from {self.start} to {self.end}")
   
        raw = yf.download(all_tickers, start=self.start, end=self.end, interval=self.interval, group_by='ticker', threads=True)

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
            except Exception:
                # fallback: use yf.Ticker
                tk = yf.Ticker(t)
                hist = tk.history(start=self.start, end=self.end, interval=self.interval)
                adj = hist['Close'] * (hist.get('Adj Close', hist['Close']) / hist['Close'])
                cl = hist['Close']
                v = hist.get('Volume', pd.Series(index=hist.index))

            adj_close[t] = adj
            close[t] = cl
            vol[t] = v

        # Combine into a long-format DataFrame
        adj_close.index = pd.to_datetime(adj_close.index).tz_localize(None).normalize()
        print(f"Downloaded data for {len(adj_close.columns)} assets with {len(adj_close)} records each.")
        return adj_close.sort_index()
    
    def get_macro_data(self) -> pd.DataFrame:
        
        fred = Fred(api_key=self.fred_api_key)
        macro_df = pd.DataFrame()
        for fred_code, name in self.macro_series.items():
            try:
                series = fred.get_series(fred_code, observation_start=self.start, observation_end=self.end)
                series.name = fred_code
                macro_df = pd.concat([macro_df, series], axis=1)
            except Exception as e:
                print(f"Error fetching {fred_code}: {e}")
        print(f"Downloaded {len(self.macro_series)} macroeconomic data with {len(macro_df)} records.")
        macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).normalize()
        return macro_df.sort_index()

    # Use nasdaq-data-link (Quandl's new name) for commodity data / alternative sources
    
    def fetch_nasdaq_dataset(self) -> pd.DataFrame:
        if self.alpha_vantage_api_key:
            nasdaqdatalink.ApiConfig.api_key = self.nasdaq_api_key
        # dataset_code example: 'CHRIS/CME_CL1' for crude oil continuous futures on Quandl (if available)
        try:
            for dataset_code in ['CHRIS/CME_CL1', 'CHRIS/CME_GC1', 'CHRIS/CME_SI1']:
                df = self._fetch_nasdaq_dataset(dataset_code, self.start, self.end)
                if not df.empty:
                    print(f"Fetched dataset {dataset_code} with {len(df)} records.")
            return df
        except Exception as e:
            print(f"Error fetching Nasdaq dataset: {e}")
            df = nasdaqdatalink.get(dataset_code, start_date=self.start, end_date=self.end)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()

    def preprocess_data(self, prices, macro_df):

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
        return data
    
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
        print("\n=== Starting Data Collection ===\n")
        print("Downloading asset prices...")
        prices = self.get_asset_prices()
        print("\nDownloading macro indicators...")
        macro_data = self.get_macro_data()
        print("\nPreprocessing data...")
        data = self.preprocess_data(prices, macro_data)
        return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fetch and preprocess financial and macroeconomic data.")
    parser.add_argument('--start', type=str, default="2000-01-01", help='Start date for data fetching in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date for data fetching in YYYY-MM-DD format')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d, 1wk, 1mo)')

    args = parser.parse_args()

    collector = DataCollector(start=args.start, end=args.end, interval=args.interval)
    data = collector.run_full_collection()

    #save to disk
    data['prices'].to_csv("data/processed/asset_prices.csv")
    data['macro'].to_csv("data/processed/macro_data.csv")

    # Quick visualization check
    print("\nSample prices (last 5 days):")
    print(data['prices'].tail())
    
    print("\nSample macro data (last 5 observations):")
    print(data['macro'].tail())

