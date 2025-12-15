import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import VECM
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:

    def __init__(self, data, lookback_windows=[63, 126, 252]):

        self.prices = data['prices']
        self.macro = data['macro']
        self.returns = data['returns']
        self.windows = lookback_windows

        self.features = pd.DataFrame(index=self.prices.index)
    
    def compute_cointegration_features(self, asset1: str, asset2: str):
        """
        Compute cointegration features between two assets.
        """
        print(f"Computing cointegration features for {asset1} and {asset2}")

        p1 = self.prices[asset1].dropna()
        p2 = self.prices[asset2].dropna()

        common_id = p1.index.intersection(p2.index)
        p1 = p1.loc[common_id]
        p2 = p2.loc[common_id]

        features = {}

        for window in self.windows:

            coint_pvalues = []
            adf_stats = []
            ect_vals = []
            hedge_ratios = []

            for end in range(window, len(common_id)):
                start = end - window
                sub_p1 = p1.iloc[start:end]
                sub_p2 = p2.iloc[start:end]

                try:
                    coint_t, pvalue, _ = coint(sub_p1, sub_p2)
                    coint_pvalues.append(pvalue)

                    beta = np.cov(sub_p1, sub_p2)[0,1] / np.var(sub_p2)
                    hedge_ratios.append(beta)

                    ect = sub_p1[-1] - beta * sub_p2[-1]
                    ect_vals.append(ect)
                    adf_result = adfuller(sub_p1 - beta * sub_p2)
                    adf_stats.append(adf_result[0])
                
                except Exception as e:
                    coint_pvalues.append(np.nan)
                    hedge_ratios.append(np.nan)
                    ect_vals.append(np.nan)
                    adf_stats.append(np.nan)
            
            feat_label = f"{asset1}_{asset2}_window{window}"
            features[f"{feat_label}_coint_pvalue"] = coint_pvalues
            features[f"{feat_label}_hedge_ratio"] = hedge_ratios
            features[f"{feat_label}_ect"] = ect_vals
            features[f"{feat_label}_adf_stat"] = adf_stats

            ect_series = pd.Series(ect_vals, index=common_id[window:])
            ect_zscore = (ect_series - ect_series.rolling(window=window).mean()) / ect_series.rolling(window=window).std()
            features[f"{feat_label}_ect_zscore"] = ect_zscore.reindex(common_id).values
        
        feat_df = pd.DataFrame(features, index=p1.index[self.windows[-1]:])
        return feat_df
    
    def compute_volatility_regime_features(self, asset: str):
        """
        Compute volatility regime features for a single asset.
        """
        print(f"Computing volatility regime features for {asset}")

        returns = self.returns[asset].dropna()
        features = {}

        for window in self.windows:
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility
            rolling_mean_vol = rolling_vol.rolling(window=window).mean()
            rolling_std_vol = rolling_vol.rolling(window=window).std()

            zscore_vol = (rolling_vol - rolling_mean_vol) / rolling_std_vol

            features[f"{asset}_vol_window{window}"] = rolling_vol
            features[f"{asset}_vol_zscore_window{window}"] = zscore_vol
        
        feat_df = pd.DataFrame(features, index=returns.index)
        return feat_df