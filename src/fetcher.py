import yfinance as yf
import pandas as pd
from typing import List, Optional


def fetch_yahoo(tickers: List[str], start: str = None, end: str = None, interval: str = '1d') -> pd.DataFrame:
    """
    Download OHLCV for multiple tickers and return a multiindex (Date, Ticker) DataFrame of adjusted close and volume.
    """
    # yfinance can return a multi-column DataFrame; we'll normalize it.
    raw = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', threads=True)

    # If single ticker, normalize to multiindex
    if len(tickers) == 1:
        df = raw.copy()
        df.columns = pd.MultiIndex.from_product([ [tickers[0]], df.columns ])

    # Build adjusted close matrix
    adj_close = pd.DataFrame()
    close = pd.DataFrame()
    vol = pd.DataFrame()

    for t in tickers:
        try:
            adj = raw[t]['Adj Close'] if isinstance(raw.columns, pd.MultiIndex) else raw['Adj Close']
            cl = raw[t]['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
            v = raw[t]['Volume'] if isinstance(raw.columns, pd.MultiIndex) else raw['Volume']
        except Exception:
            # fallback: use yf.Ticker
            tk = yf.Ticker(t)
            hist = tk.history(start=start, end=end, interval=interval)
            adj = hist['Close'] * (hist.get('Adj Close', hist['Close']) / hist['Close'])
            cl = hist['Close']
            v = hist.get('Volume', pd.Series(index=hist.index))

        adj_close[t] = adj
        close[t] = cl
        vol[t] = v

    # Combine into a long-format DataFrame
    adj_close.index = pd.to_datetime(adj_close.index)
    return adj_close.sort_index()

# ------------------------- fetchers/fred_fetcher.py -------------------------
# Fetch macroeconomic series from FRED
from fredapi import Fred
import pandas as pd


def fetch_fred_series(series_ids: list, api_key: str, start: str = None, end: str = None) -> pd.DataFrame:
    fred = Fred(api_key=api_key)
    out = {}
    for s in series_ids:
        try:
            df = fred.get_series(s, observation_start=start, observation_end=end)
            out[s] = pd.Series(df, name=s)
        except Exception as e:
            print(f"Error fetching {s}: {e}")
    if not out:
        return pd.DataFrame()
    df = pd.concat(out.values(), axis=1)
    df.columns = list(out.keys())
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

# Use nasdaq-data-link (Quandl's new name) for commodity data / alternative sources
import nasdaq_data_link
import pandas as pd


def fetch_nasdaq_dataset(dataset_code: str, start_date: str = None, end_date: str = None, api_key: str = None) -> pd.DataFrame:
    if api_key:
        nasdaq_data_link.ApiConfig.api_key = api_key
    # dataset_code example: 'CHRIS/CME_CL1' for crude oil continuous futures on Quandl (if available)
    try:
        df = nasdaq_data_link.get(dataset_code, start_date=start_date, end_date=end_date)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        print(f"Error fetching dataset {dataset_code}: {e}")
        return pd.DataFrame()

# Alpha Vantage can provide time series and some commodity/economic indicators. Note free tier limits.
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time


def fetch_alpha_vantage_daily(symbol: str, api_key: str, outputsize: str = 'full') -> pd.DataFrame:
    ts = TimeSeries(key=api_key, output_format='pandas')
    # NOTE: AlphaVantage free tier is limited to 5 requests/min and 500 requests/day
    try:
        data, meta = ts.get_daily_adjusted(symbol=symbol, outputsize=outputsize)
    except Exception as e:
        print(f"AlphaVantage error for {symbol}: {e}")
        return pd.DataFrame()
    data.index = pd.to_datetime(data.index)
    return data.sort_index()
