import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import cwt, ricker
import pymannkendall as mk
from tqdm import tqdm

from scipy import signal
import numpy as np
import pycatch22 
from sktime.transformations.panel import catch22
import tsfresh
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

from cesium import featurize

from tqdm import tqdm
import sys, os
import pandas as pd 

import gc

from collections import defaultdict

from time import sleep

    
# add Cesium features
DEFAULT_CESIUM_FEATURES = ["amplitude", "percent_beyond_1_std", 
                          "median_absolute_deviation", "percent_close_to_median",
                          "weighted_average", "all_times_nhist_numpeaks", 
                          "all_times_nhist_peak_1_to_2", "all_times_nhist_peak_val",
                          "avg_double_to_single_step", "avg_err", "avgt",
                          "anderson_darling",  "shapiro_wilk"]

def get_crossectional(tsdf: pd.DataFrame,
                      id_col='ID',
                      val_col='eGFR_CKDEpi2012', 
                      time_col='Time_col', 
                      catch22=False, 
                      cesium=False):

    CustomExtractor = Extractor()
    CustomExtractor.fit(tsdf, 
                        id_col=id_col, 
                        val_col=val_col, 
                        time_col=time_col)

    ts_data_agg = CustomExtractor.transform()

    FreshExtractor = TsFreshExtractor()
    FreshExtractor.fit(tsdf, 
                        id_col=id_col, 
                        val_col=val_col,
                        time_col=time_col)
    ts_data_agg_fresh = FreshExtractor.transform()


    if catch22==True:
        Catch22Extract  = Catch22Extractor()
        Catch22Extract.fit(tsdf, 
                    id_col=id_col, 
                    val_col=val_col,
                    time_col=time_col)
        ts_data_agg_catch22 = Catch22Extract.transform()

    if cesium==True:
        CesiumExtract = CesiumExtractor()
        CesiumExtract.fit(tsdf, 
                            id_col=id_col, 
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_cesium = CesiumExtract.transform()

    ts_data_agg = ts_data_agg.set_index('id')

    FINAL_FEATURES = ts_data_agg.merge(ts_data_agg_fresh,
                                       left_index=True,
                                       right_index=True)
    if catch22==True:
        FINAL_FEATURES = FINAL_FEATURES.merge(ts_data_agg_catch22,
                                              left_index=True,
                                              right_index=True)
    
    if cesium==True:
        FINAL_FEATURES = FINAL_FEATURES.merge(ts_data_agg_cesium,
                                              left_index=True,
                                              right_index=True)    
    
    return FINAL_FEATURES

class Extractor:
    def __init__(self):
        self.features = {}
    
    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        df = df.sort_values(by=[id_col, time_col])
        # Iterate over all unique IDs in the DataFrame
        for _id in tqdm(df[id_col].unique()):
            # Extract the time series data for the current ID
            ts_data = df[df[id_col] == _id][val_col].values
            
            # Calculate features
            mean_ = np.mean(ts_data)
            min_ = np.min(ts_data)
            max_ = np.max(ts_data)
            variance = np.var(ts_data)
            skewness = skew(ts_data)
            kurtosis_value = kurtosis(ts_data)
            mk_s, mk_z, mk_Tau, mk_ss, mk_var_s, mk_slope, mk_intercept, mk_trend = \
                        self._mann_kendall_test(ts_data)
            wavelet_transform_feature = self._wavelet_transform_feature(ts_data)
            psd_int = _psd_int(ts_data, integrator='trapezoidal')
            
            # Store the features for the current ID
            self.features[_id] = {
                'mean': mean_,
                'min': min_,
                'max': max_,
                'variance': variance,
                'skewness': skewness,
                'kurtosis': kurtosis_value,
                'mann_kendall_s': mk_s,
                'mann_kendall_z': mk_z,
                'mann_kendall_Tau': mk_Tau,
                'mann_kendall_ss': mk_ss,
                'mann_kendall_var_s': mk_var_s,
                'mann_kendall_slope': mk_slope,
                'mann_kendall_intercept': mk_intercept,
                'mann_kendall_trend': mk_trend,
                'wavelet_transform_feature': wavelet_transform_feature,
                'psd_int': psd_int
            }
    
    def _mann_kendall_test(self, data):
        n = len(data)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(data[j] - data[i])
                
        trend, _, _, z, Tau, s, var_s, slope, intercept = mk.original_test(data)
        trend = 1 if trend == 'increasing' else -1 if trend == 'decreasing' else 0        
        return s/n, z, Tau, s, var_s, slope, intercept, trend

    def _wavelet_transform_feature(self, data):
        # This is a placeholder for a real wavelet transform feature extraction.
        # For simplicity, we return the mean of the wavelet coefficients here.
        # Replace this with your actual wavelet feature extraction logic.
        widths = np.arange(1, 31)
        cwtmatr = cwt(data, ricker, widths)
        return np.mean(cwtmatr)
    
    def transform(self):
        # Convert the features dictionary to a DataFrame
        features_df = pd.DataFrame.from_dict(self.features, orient='index').reset_index()
        features_df.rename(columns={'index': 'id'}, inplace=True)
        return features_df
    


class TsFreshExtractor:
    def __init__(self):
        self.features = {}
    
    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        # Extract features
        df = df.sort_values(by=[id_col, time_col])
        extracted_features = \
                extract_features(df, 
                                 column_id=id_col, 
                                 column_sort=time_col,
                                 impute_function=impute,
                                 default_fc_parameters=ComprehensiveFCParameters())
        self.features = extracted_features
    
    def _filter_correlated_features(self, df, threshold=0.8):
        # use scipy.stats.spearmanr to calculate the correlation between features
        # and filter out features that are highly correlated
        tsc = ts_data_agg_fresh.corr(method='spearman')
        cols = tsc.columns
        rows = tsc.index
        
        tsc = tsc.values
        tsc = np.tril(tsc, k=-1)      
                
        return df    
    
    def transform(self):
        return self.features

from pycatch22 import catch22_all

class Catch22Extractor:
    def __init__(self):
        self.features = {}
    
    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        # Extract features
        df = df.sort_values(by=[id_col, time_col])
        _features = {}
        for _id in tqdm(df[id_col].unique()):
            ts_data = df[df[id_col] == _id][val_col].values
            extracted_features = pycatch22.catch22_all(ts_data)            
            _features[_id] = dict(zip(extracted_features['names'], extracted_features['values']))
        self.features = _features
    
    def transform(self):
        # Convert the features dictionary to a DataFrame
        features_df = pd.DataFrame.from_dict(self.features, 
                                             orient='index').reset_index()
        features_df = features_df.rename(columns={'index': 'id'})
        features_df = features_df.set_index('id')
        return features_df
    

class CesiumExtractor:
    def __init__(self, 
                 features_to_use=None, 
                 extract_periodic_features=True,
                 extract_cad_features=False):
        self.features = {}
        self.features_to_use = features_to_use if features_to_use \
                                    else DEFAULT_CESIUM_FEATURES

        if extract_periodic_features:
            self.features_to_use += ["period_fast", "freq1_freq", "freq2_freq", 
                                     "freq3_freq", "linear_trend","freq1_rel_phase2", 
                                     "freq2_rel_phase2", "freq3_rel_phase2"]
            
        if extract_cad_features:
            self.features_to_use += ["cad_probs_10", "cad_probs_30", "cad_probs_100", 
                                     "cad_probs_500", "cads_avg"]

    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        df = df.sort_values(by=[id_col, time_col])
        _features = {}

        for _id in tqdm(df[id_col].unique()):
            times = df[df[id_col] == _id][time_col].values
            vals = df[df[id_col] == _id][val_col].values
            feats = featurize.featurize_time_series(
                                            times=times,
                                            values=vals,
                                            errors=None,
                                            features_to_use=self.features_to_use,
                                            )
            feats.columns = feats.columns.droplevel(-1)
            feats_dict = feats.to_dict()
            _features[_id] = {k:v[0] for k,v in feats_dict.items()}
        self.features = _features

    def transform(self):
        # Convert the features dictionary to a DataFrame
        features_df = pd.DataFrame.from_dict(self.features, 
                                             orient='index').reset_index()
        features_df = features_df.rename(columns={'index': 'id'})
        features_df = features_df.set_index('id')
        return features_df

# TODO: Implement the KatzExtractor class
class KatzExtractor:
    # https://github.com/facebookresearch/Kats/blob/main/tutorials/kats_203_tsfeatures.ipynb
    # https://github.com/facebookresearch/Kats/blob/main/kats/tsfeatures/tsfeatures.py
    pass


def get_smoothNsmooth_diffStatistics(ts_raw: pd.DataFrame, 
                                     ts_smooth: pd.DataFrame,
                                     id_col: str='ID',
                                     val_col: str='value',
                                     time_col: str='dt')->pd.DataFrame:
    """
    Get the difference statistics between raw time series and its smoothed version.
    
    Args:   ts_raw: np.array: raw time series
            ts_smooth: np.array: smoothed time series
            
    Output: dict: difference statistics
    """

    tsM = ts_raw.merge(ts_smooth,
                        left_on=[id_col, time_col],
                        right_on=[id_col, time_col],
                        suffixes=('_raw', '_smoothed'))
    
    tsM = tsM.assign(diff = tsM[val_col+"_raw"] - tsM[val_col+"_smoothed"])
    diffg = tsM.groupby(id_col)['diff']         
    diff_mean = diffg.aggregate(lambda x: np.mean(x))
    diff_abs_mean = diffg.aggregate(lambda x: np.mean(np.abs(x)))
    diff_abs_median = diffg.aggregate(lambda x: np.median(np.abs(x)))
    diff_std = diffg.aggregate(lambda x: np.std(x))
    diff_skew = diffg.aggregate(lambda x: skew(x))
    diff_kurtosis = diffg.aggregate(lambda x: kurtosis(x))
    
    res = pd.concat([diff_mean, diff_abs_mean, diff_abs_median, 
                     diff_std, diff_skew, diff_kurtosis], axis=1)
    
    res.columns = [f'diff{c}' for c in ['_mean', '_abs_mean', '_abs_median', 
                                        '_std', '_skew', '_kurtosis']]
    return res

def _psd_int(ts_data, integrator='trapezoidal'):
    """
    Get the integral of the power spectral density of the time series.
    
    Args:   ts_data: np.array: time series data
            integrator: str: integration technique ['trapezoidal' or 'weighted']
            
    Output: float: integral of the power spectral density
    """
    n_samples = len(ts_data)
    base_num = min(n_samples, 512)
    
    f, Pxx = signal.welch(ts_data, fs=1.0, nperseg=base_num, noverlap=int(0.5*base_num), nfft=2*base_num)
    if integrator == 'trapezoidal':
        psd_int = np.trapz(Pxx, f)
    elif integrator == 'weighted':
        psd_int = sum(f[1:]*np.diff(f)*(Pxx[:-1]+Pxx[1:]))
    else:
        raise ValueError('Invalid integrator. Use "trapezoidal" or "weighted".')
    return psd_int

def psd_int(df: pd.DataFrame,
            id_col: str='ID', 
            val_col: str='value',
            time_col: str='dt',
            integrator: str='trapezoidal')->pd.DataFrame:
    """
    Get the integral of the power spectral density of the time series.
    
    Args:   df: pd.DataFrame: time series data frame
            id_col: str: id column name
            val_col: str: value column name
            time_col: str: time column name
            integrator: str: integration technique ['trapezoidal' or 'weighted']
            
    Output: pd.DataFrame: integral of the power spectral density
    """
    df = df.sort_values(by=[id_col, time_col])
    res = {id_col: [], 'psd_int': []}
    for _id in tqdm(df[id_col].unique()):
        ts_data = df[df[id_col] == _id][val_col].values
        
        psd_int = _psd_int(ts_data, integrator=integrator)
        res[id_col].append(_id)
        res['psd_int'].append(psd_int)
    
    return pd.DataFrame(res)