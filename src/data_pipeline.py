# This script handles all data ingestion, preprocessing, and feature engineering in one place with modular functions

'''
load_data(source: str) -> pd.DataFrame
clean_data(df: pd.DataFrame) -> pd.DataFrame
merge_equity_commodity_macro(eq_df, com_df, macro_df) -> pd.DataFrame
create_features(df) -> pd.DataFrame  # e.g., volatility, spreads, returns
save_processed(df, path: str)
'''