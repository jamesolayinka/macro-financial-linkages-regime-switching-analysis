import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import VECM
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:

    def __init__(self, data, lookback_windows=[63, 126, 252]):

        self.prices = data['prices']
        self.macro = data['macro']
        self.returns = data['returns']
        self.windows = lookback_windows if isinstance(lookback_windows, list) else [lookback_windows]

        self.features = pd.DataFrame(index=self.prices.index)
    
    def compute_cointegration_features(self, asset1: str, asset2: str):
        """
        Compute cointegration features between two assets.
        """
        logger.info(f"Computing cointegration features: {asset1} vs {asset2}")

        p1 = self.prices[asset1].dropna()
        p2 = self.prices[asset2].dropna()

        common_id = p1.index.intersection(p2.index)
        p1 = p1.loc[common_id]
        p2 = p2.loc[common_id]

        features = {}
        
        for window in self.windows:

            n_obs = len(common_id)

            coint_pvalues = np.full(n_obs, np.nan)
            adf_stats = np.full(n_obs, np.nan)
            ect_vals = np.full(n_obs, np.nan)
            hedge_ratios = np.full(n_obs, np.nan)

            for end in range(window, len(common_id)):
                start = end - window
                sub_p1 = p1.iloc[start:end].values
                sub_p2 = p2.iloc[start:end].values

                try:
                    coint_t, pvalue, _ = coint(sub_p1, sub_p2)
                    coint_pvalues[end-1] = pvalue

                    beta = np.cov(sub_p1, sub_p2)[0,1] / np.var(sub_p2)
                    hedge_ratios[end-1] = beta

                    ect = sub_p1[-1] - beta * sub_p2[-1]
                    ect_vals[end-1] = ect
                    adf_result = adfuller(sub_p1 - beta * sub_p2)
                    adf_stats[end-1] = adf_result[0]
                
                except Exception as e:
                    pass
            
            feat_label = f"{asset1}_{asset2}_{window}d"
            features[f"coint_pvalue_{feat_label}"] = coint_pvalues
            features[f"hedge_ratio_{feat_label}"] = hedge_ratios
            features[f"ect_{feat_label}"] = ect_vals
            features[f"adf_stat_{feat_label}"] = adf_stats

            ect_series = pd.Series(ect_vals, index=common_id)
            ect_mean = ect_series.rolling(window=window, min_periods=window).mean()
            ect_std = ect_series.rolling(window=window, min_periods=window).std()
            ect_zscore = (ect_series - ect_mean) / ect_std
            features[f"ect_zscore_{feat_label}"] = ect_zscore.values
        
        feat_df = pd.DataFrame(features, index=common_id)
        logger.info(f"  Generated {feat_df.shape[1]} cointegration features")
        return feat_df
    
    # ============================= REGIME FEATURES =============================
    def compute_volatility_regime_features(self, asset: str):
        """
        Compute volatility regime features for a single asset.
        """
        logger.info(f"Computing volatility regime features: {asset}")

        returns = self.returns[asset].dropna()
        features = {}

        for window in self.windows:
            rolling_vol = returns.rolling(window, min_periods=window).std() * np.sqrt(252)  # Annualized volatility
            features[f"{asset}_vol_{window}"] = rolling_vol

            # Expanding window quantiles for volatility regimes
            vol_expanding = rolling_vol.expanding(min_periods=window*2)
            q33 = vol_expanding.quantile(0.33)
            q67 = vol_expanding.quantile(0.67)

            vol_regime = pd.Series(1, index=rolling_vol.index)
            vol_regime[rolling_vol <= q33] = 0  # Low volatility regime
            vol_regime[rolling_vol >= q67] = 2  # High volatility regime
            features[f"{asset}_vol_regime_{window}d"] = vol_regime

            vol_of_vol = rolling_vol.rolling(window, min_periods=window).std()
            features[f"{asset}_vol_of_vol_{window}d"] = vol_of_vol

        if 'VIX' in self.macro.columns:
            vix = self.macro['VIX'].reindex(returns.index, method='ffill')

            vix_clean= vix.dropna()
            if len(vix_clean) > 0:
                q33 = vix_clean.quantile(0.33)
                q67 = vix_clean.quantile(0.67)

                features["vix_level"] = vix
                features['vix_regime_low'] = (vix < q33).astype(int)
                features['vix_regime_high'] = (vix > q67).astype(int)
                features['vix_change'] = vix.pct_change()
        feat_df = pd.DataFrame(features, index=returns.index)
        logger.info(f"Generated {feat_df.shape[1]} volatility features")
        return feat_df
    
    def compute_correlation_regime_features(self, asset1: str, asset2: str):
        """
        Compute correlation regime features between two assets.
        """
        logger.info(f"Computing correlation regime features: {asset1} vs {asset2}")

        r1 = self.returns[asset1].dropna()
        r2 = self.returns[asset2].dropna()

        common_id = r1.index.intersection(r2.index)
        r1 = r1.loc[common_id]
        r2 = r2.loc[common_id]

        features = {}

        for window in self.windows:
            rolling_corr = r1.rolling(window, min_periods=window).corr(r2)
            features[f"corr_{asset1}_{asset2}_{window}d"] = rolling_corr

            corr_median = rolling_corr.expanding(min_periods=window*2).median()
            features[f"corr_median_{asset1}_{asset2}_{window}d"] = corr_median

            corr_regime = (rolling_corr > corr_median).astype(int)
            features[f"corr_regime_{asset1}_{asset2}_{window}d"] = corr_regime

            corr_vol = rolling_corr.rolling(window, min_periods=window).std()
            features[f"corr_vol_{asset1}_{asset2}_{window}d"] = corr_vol

        feat_df = pd.DataFrame(features, index=common_id)
        logger.info(f"Generated {feat_df.shape[1]} correlation features")
        return feat_df
    
    def compute_momentum_regime(self, asset: str):
        """
        Compute momentum regime features for a single asset.
        """
        logger.info(f"Computing momentum regime features: {asset}")

        prices = self.prices[asset].dropna()
        returns = self.returns[asset].dropna()
        features = {}

        for window in self.windows:
            ma = prices.rolling(window).mean()
            features[f"price_above_ma_{asset}_{window}d"] = (prices > ma).astype(int)

            cum_return = returns.rolling(window).sum()
            features[f"momentum_{asset}_{window}d"] = cum_return

            features[f"momentum_regime_{asset}_{window}d"] = (cum_return > 0).astype(int)
            trend_strength = prices.rolling(window).apply(
                lambda x: self._compute_trend_strength(x) if len(x.dropna()) == window else np.nan, raw=False
            )
            features[f"trend_strength_{asset}_{window}d"] = trend_strength
        feat_df = pd.DataFrame(features, index=returns.index)
        logger.info(f"Generated {feat_df.shape[1]} momentum features")
        return feat_df
    
    @staticmethod
    def _compute_trend_strength(price_series):
        try:
            x = np.arange(len(price_series))
            y = price_series.values
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            return r_value ** 2  # R-squared as trend strength
        except:
            return np.nan

    # ============================= COMBINE FEATURES =============================
    def generate_all_features(self, asset_pairs=[("SPY", "GLD"), ("SPY", "TLT")]):
        """
        Generate all features for given asset pairs.
        """
        logger.info("=" * 60)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 60)

        all_features = []

        for asset1, asset2 in asset_pairs:
            coint_feats = self.compute_cointegration_features(asset1, asset2)
            corr_feats = self.compute_correlation_regime_features(asset1, asset2)
            all_features.extend([coint_feats, corr_feats])

        unique_assets = set([a for pair in asset_pairs for a in pair])
        for asset in unique_assets:
            vol_feats = self.compute_volatility_regime_features(asset)
            mom_feats = self.compute_momentum_regime(asset)
            all_features.append(vol_feats)
            all_features.append(mom_feats)

        self.features = pd.concat(all_features, axis=1, join='outer').sort_index()
        
        # Log summary
        logger.info("=" * 60)
        logger.info("Feature Engineering Complete")
        logger.info(f"  Total features: {self.features.shape[1]}")
        logger.info(f"  Date range: {self.features.index[0]} to {self.features.index[-1]}")
        logger.info(f"  Observations: {len(self.features)}")
        logger.info(f"  Missing values: {self.features.isna().sum().sum():,} ({self.features.isna().sum().sum() / self.features.size * 100:.1f}%)")
        logger.info("=" * 60)

        return self.features
    
    def get_feature_categories(self):
        """Categorize features by type for easier analysis"""
        categories = {
            'cointegration': [],
            'correlation': [],
            'volatility': [],
            'momentum': [],
            'vix': [],
        }
        
        for col in self.features.columns:
            if 'coint' in col or 'ect' in col or 'hedge' in col or 'adf' in col:
                categories['cointegration'].append(col)
            elif 'corr' in col:
                categories['correlation'].append(col)
            elif 'vol' in col and 'vix' not in col.lower():
                categories['volatility'].append(col)
            elif 'momentum' in col or 'trend' in col or 'ma' in col:
                categories['momentum'].append(col)
            elif 'vix' in col.lower():
                categories['vix'].append(col)
        
        return categories
    
    def feature_summary(self):

        summary = self.features.describe().T
        summary['missing_values'] = self.features.isna().sum()
        summary['missing_percentage'] = (summary['missing_values'] / len(self.features)) * 100
        return summary.sort_values(by='missing_percentage', ascending=False)
    
if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Load your data here
    data = {
        'prices': pd.read_csv('data/processed/asset_prices.csv', index_col=0, parse_dates=True),
        'returns': pd.read_csv('data/processed/asset_returns.csv', index_col=0, parse_dates=True),
        'macro': pd.read_csv('data/processed/macro_data.csv', index_col=0, parse_dates=True)
    }
    fe = FeatureEngineer(data)
    features = fe.generate_all_features(asset_pairs=[("SPY", "GLD"), ("SPY", "USO")])
    summary = fe.feature_summary()
    print(summary)