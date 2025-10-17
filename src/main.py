import datetime
import config
from preprocess import  write_series_parquet
from fetcher import fetch_alpha_vantage_daily, fetch_nasdaq_dataset, fetch_yahoo, fetch_fred_series
import logging


def example_run():
    ensure_dirs(config.RAW_DATA_DIR, config.PROCESSED_DIR)

    # 1) Equity & commodity tickers (example)
    equities = ['^GSPC', '^FTSE', 'MSFT']  # S&P500, FTSE 100 (Yahoo tickers), Microsoft
    commodities = ['CL=F', 'GC=F']  # Crude Oil and Gold futures (Yahoo tickers)

    start = config.START_DATE
    end = config.END_DATE

    print('Fetching equities...')
    eq_prices = fetch_yahoo(equities, start=start, end=end)
    write_series_parquet(eq_prices, 'equities_yahoo', base_dir=config.RAW_DATA_DIR)

    print('Fetching commodities...')
    comm_prices = fetch_yahoo(commodities, start=start, end=end)
    write_series_parquet(comm_prices, 'commodities_yahoo', base_dir=config.RAW_DATA_DIR)

    # 2) Macroeconomic series from FRED
    fred_series = ['CPIAUCSL', 'UNRATE', 'INDPRO']  # CPI, Unemployment rate, Industrial production
    print('Fetching FRED series...')
    macro = fetch_fred_series(fred_series, api_key=config.FRED_API_KEY, start=start, end=end)
    write_series_parquet(macro, 'macro_fred', base_dir=config.RAW_DATA_DIR)

    # 3) Example: Nasdaq Data Link for continuous futures (replace with actual dataset codes you have access to)
    # Common Quandl dataset examples (subject to availability): 'CHRIS/CME_CL1' (Crude Oil front-month continuous)
    try:
        ndl = fetch_nasdaq_dataset('CHRIS/CME_CL1', start_date=start, end_date=end, api_key=config.NASDAQ_DATALINK_KEY)
        if not ndl.empty:
            write_series_parquet(ndl, 'quandl_crude_cl1', base_dir=config.RAW_DATA_DIR)
    except Exception as e:
        print('Nasdaq Data Link fetch skipped or failed:', e)

    # 4) Preprocess: align and compute returns
    prices = align_and_forward_fill({'sp500': eq_prices['^GSPC'] if '^GSPC' in eq_prices.columns else eq_prices.iloc[:,0],
                                    'crude': comm_prices['CL=F'] if 'CL=F' in comm_prices.columns else comm_prices.iloc[:,0]})
    returns = compute_log_returns(prices)
    write_series_parquet(returns, 'daily_returns', base_dir=config.PROCESSED_DIR)


if __name__ == '__main__':
    example_run()
