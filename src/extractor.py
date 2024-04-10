import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis, entropy as _entropy, linregress
from scipy.fft import rfft, rfftfreq
from scipy.signal import cwt, ricker, periodogram
from scipy import signal

import pymannkendall as mk
from tqdm import tqdm

import pycatch22 
from sktime.transformations.panel import catch22
import tsfresh
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, kpss, pacf

import antropy as ant
from cesium import featurize

from tqdm import tqdm
import sys, os
import pandas as pd 

from numba import jit
import gc

from collections import defaultdict

from time import sleep

import antropy as ant
import nolds

#TODO: add interpretable feature mappings 
# e.g. {'slopes':{}, 'periodicity':{}, 'entropy':{}, 'amplitude':{}, 'trend':{}, 'nonlinearity':{}, 'spikes':{}, 'crossings':{}, 'energy':{}, 'statistics':{}, 'distribution':{}, 'autocorrelation':{}, 'stability':{}, 'linearity':{}, 'complexity':{}, 'nonlinear':{}, 'chaos':{}, 'misc':{}} 
    
## Ideas for extracts
# 'complexity': how many fourier components are needed to describe the signal with a certain accuracy
# 'complexity': spline complexity, how many splines are needed to describe the signal with a certain accuracy

# add Cesium features
DEFAULT_CESIUM_FEATURES = ["amplitude", "percent_beyond_1_std", 
                          "median_absolute_deviation", "percent_close_to_median",
                          "weighted_average", "all_times_nhist_numpeaks", 
                          "all_times_nhist_peak_1_to_2", "all_times_nhist_peak_val",
                          "avg_double_to_single_step", "avg_err", "avgt",
                          "anderson_darling",  "shapiro_wilk"]

ANTROPY_FEATURES = [
                    "perm_entropy",
                    "spectral_entropy",
                    "svd_entropy",
                    "app_entropy",
                    "sample_entropy",
                    "lziv_complexity",
                    "num_zerocross",
                    "hjorth_params",
                    "petrosian_fd", 
                    "katz_fd", 
                    "higuchi_fd", 
                    "detrended_fluctuation"
                ]


def get_crossectional(tsdf: pd.DataFrame,
                      id_col='ID',
                      val_col='eGFR_CKDEpi2012', 
                      time_col='Time_col', 
                      tsfresh_features=False,
                      catch22_features=False, 
                      cesium_features=False,
                      antropy_features=False,
                      nolds_features=False):

    CustomExtractor = Extractor()
    CustomExtractor.fit(tsdf, 
                        id_col=id_col, 
                        val_col=val_col, 
                        time_col=time_col)

    ts_data_agg = CustomExtractor.transform()

    if tsfresh_features ==True:
        FreshExtractor = TsFreshExtractor()
        FreshExtractor.fit(tsdf, 
                            id_col=id_col, 
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_fresh = FreshExtractor.transform()


    if catch22_features==True:
        Catch22Extract  = Catch22Extractor()
        Catch22Extract.fit(tsdf, 
                    id_col=id_col, 
                    val_col=val_col,
                    time_col=time_col)
        ts_data_agg_catch22 = Catch22Extract.transform()

    if cesium_features==True:
        CesiumExtract = CesiumExtractor()
        CesiumExtract.fit(tsdf, 
                            id_col=id_col, 
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_cesium = CesiumExtract.transform()
    
    if antropy_features==True:
        AntropyExtract = AntropyExtractor()
        AntropyExtract.fit(tsdf, 
                            id_col=id_col, 
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_antropy = AntropyExtract.transform()
    
    if nolds_features==True:
        NoldsExtract = NoldsExtractor()
        NoldsExtract.fit(tsdf, 
                            id_col=id_col, 
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_nolds = NoldsExtract.transform()

    ts_data_agg = ts_data_agg.set_index('id')

    _cstring = '_custom'
    if tsfresh_features==True:        
        FINAL_FEATURES = FINAL_FEATURES.merge(ts_data_agg_fresh,
                                        left_index=True,
                                        right_index=True,
                                        suffixes=('_custom', '_fresh'))
        _cstring = ''
        
    if catch22_features==True:
        FINAL_FEATURES = FINAL_FEATURES.merge(ts_data_agg_catch22,
                                              left_index=True,
                                              right_index=True,
                                              suffixes=(_cstring,'_catch22'))
    
    if cesium_features==True:
        FINAL_FEATURES = FINAL_FEATURES.merge(ts_data_agg_cesium,
                                              left_index=True,
                                              right_index=True,
                                              suffixes=(_cstring,'_cesium'))
    
    if antropy_features==True:
        FINAL_FEATURES = FINAL_FEATURES.merge(ts_data_agg_antropy,
                                              left_index=True,
                                              right_index=True,
                                              suffixes=(_cstring,'_antropy'))
    if nolds_features==True:
        FINAL_FEATURES = FINAL_FEATURES.merge(ts_data_agg_nolds,
                                              left_index=True,
                                              right_index=True,
                                              suffixes=(_cstring, '_nolds'))
           
    return FINAL_FEATURES

class Extractor:
    def __init__(self):
        self.features = {}
    
    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        #TODO: streamline kwargs for internal functions...
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
            
            entropy_per_rel = get_spectral_entropy(ts_data, freq=1)
            entropy_rel = get_relative_entropy(ts_data)
            lumpiness = get_lumpiness(ts_data, window_size=8)
            lump_stability = get_stability(ts_data, window_size=8)
            hurst_exponent = get_hurst(ts_data, lag_size=8)            
            rel_slope_sign_switch_sum = get_slope_sign_switch_sum(ts_data)
                        
            mk_s, mk_z, mk_Tau, mk_ss, mk_var_s, mk_slope, mk_intercept, mk_trend = \
                        _mann_kendall_test(ts_data)
            wavelet_transform_feature = _wavelet_transform_feature(ts_data)
            psd_int = _psd_int(ts_data, integrator='trapezoidal')
            
            # Store the features for the current ID
            self.features[_id] = {
                'mean': mean_,
                'min': min_,
                'max': max_,
                'variance': variance,
                'skewness': skewness,
                'kurtosis': kurtosis_value,
                'entropy_per_rel': entropy_per_rel,
                'entropy_rel': entropy_rel,
                'lumpiness': lumpiness,
                'lump_stability': lump_stability,
                'hurst_exponent': hurst_exponent,
                'mann_kendall_s': mk_s,
                'mann_kendall_z': mk_z,
                'mann_kendall_Tau': mk_Tau,
                'mann_kendall_ss': mk_ss,
                'mann_kendall_var_s': mk_var_s,
                'mann_kendall_slope': mk_slope,
                'mann_kendall_intercept': mk_intercept,
                'mann_kendall_trend': mk_trend,
                'wavelet_transform_feature': wavelet_transform_feature,
                'psd_int': psd_int,
                'rel_slope_sign_switch_sum': rel_slope_sign_switch_sum
            }
        
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

        if extract_periodic_features==True:
            self.features_to_use += ["period_fast", "freq1_freq", "freq2_freq", 
                                     "freq3_freq", "linear_trend","freq1_rel_phase2", 
                                     "freq2_rel_phase2", "freq3_rel_phase2"]
            
        if extract_cad_features==True:
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
    
    
class AntropyExtractor:
    def __init__(self,  
                 features_to_use: list=None):
                self.features_to_use = features_to_use if features_to_use \
                                            else ANTROPY_FEATURES
    @staticmethod
    def tsfeatures(ts_data, features_to_use: list=None):
        res = {}
        for f in features_to_use:
            if f == 'perm_entropy':
                res[f] = ant.perm_entropy(ts_data)
            elif f == 'spectral_entropy':
                res[f] = ant.spectral_entropy(ts_data)
            elif f == 'svd_entropy':
                res[f] = ant.svd_entropy(ts_data)
            elif f == 'app_entropy':
                res[f] = ant.app_entropy(ts_data)
            elif f == 'sample_entropy':
                res[f] = ant.sample_entropy(ts_data)
            elif f == 'lziv_complexity':
                res[f] = ant.lziv_complexity(ts_data)
            elif f == 'num_zerocross':
                res[f] = ant.num_zerocross(ts_data)
            elif f == 'hjorth_params':
                hjorth = ant.hjorth(ts_data)
                res['activity'] = hjorth[0]
                res['mobility'] = hjorth[1]
                res['complexity'] = hjorth[2]
            elif f == 'petrosian_fd':
                res[f] = ant.petrosian_fd(ts_data)
            elif f == 'katz_fd':
                res[f] = ant.katz_fd(ts_data)
            elif f == 'higuchi_fd':
                res[f] = ant.higuchi_fd(ts_data)
            elif f == 'detrended_fluctuation':
                res[f] = ant.detrended_fluctuation(ts_data)
        
        return res 
     
    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        assert(self.features_to_use is not None), "Please provide a list of features to use"
        df = df.sort_values(by=[id_col, time_col])
        _features = {}
        for _id in tqdm(df[id_col].unique()):
            ts_data = df[df[id_col] == _id][val_col].values
            feats = self.tsfeatures(ts_data, features_to_use=self.features_to_use)
            _features[_id] = feats
        self.features = _features

    def transform(self):
        # Convert the features dictionary to a DataFrame
        features_df = pd.DataFrame.from_dict(self.features, 
                                             orient='index').reset_index()
        features_df = features_df.rename(columns={'index': 'id'})
        features_df = features_df.set_index('id')
        return features_df


class NoldsExtractor:
    def __init__(self, 
                 features_to_use: list=None,
                 emb_dims=[1,2,3,4],
                 min_ts_len=100):
        self.emb_dims = [1,2,3,4] if emb_dims is None else emb_dims
        self.features_to_use = features_to_use if features_to_use \
                                else ['lyap_e', 'corr_dim']
    
    @staticmethod
    def tsfeatures(ts_data, features_to_use: list=None, min_len=50):
        res = {}
        for f in features_to_use:
            if f == 'lyap_e':
                if ts_data.shape[0] < min_len:
                    v_ = np.tile(ts_data, min_len//ts_data.shape[0] + 1)
                else:
                    v_ = ts_data       
                _res = nolds.lyap_e(v_)
                for nd, res_ in enumerate(_res):
                    res[f'lyap_e_{nd}'] = res_
            elif f == 'corr_dim':
                for edim in self.emb_dims:
                    res[f'corr_dim_{edim}'] = nolds.corr_dim(ts_data, emb_dim=edim)
        return res
    
    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        assert(self.features_to_use is not None), "Please provide a list of features to use"
        df = df.sort_values(by=[id_col, time_col])
        _features = {}
        for _id in tqdm(df[id_col].unique()):
            ts_data = df[df[id_col] == _id][val_col].values
            feats = self.tsfeatures(ts_data, 
                                    features_to_use=self.features_to_use,
                                    min_len=self.min_ts_len)
            _features[_id] = feats
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
    

    
    # get_linearity
    # get_het_arch
    # get_stl_features
    # TsFeatures.get_acf_features
    # TsFeatures.get_flat_spots
    # TsFeatures.get_pacf_features
    # TsFeatures.get_acfpacf_features
    # TsFeatures.get_unitroot_kpss
    # TsFeatures.get_holt_params
    # TsFeatures.get_cusum_detector
    # TsFeatures.get_trend_detector
    
    
    
    pass

class TSFelExtractor:
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

@jit(forceobj=True)
def get_spectral_entropy(x: np.ndarray, freq: int = 1) -> float:
    """
    source: https://github.com/facebookresearch/Kats/blob/main/kats/tsfeatures/tsfeatures.py
    Getting normalized Shannon entropy of power spectral density.
    PSD is calculated using scipy's periodogram.

    Args:
        x: The univariate time series array in the form of 1d numpy array.
        freq: int; Frequency for calculating the PSD via scipy periodogram.

    Returns:
        Normalized Shannon entropy.
    """

    # calculate periodogram
    _, psd = periodogram(x, freq)

    # calculate shannon entropy of normalized psd
    psd_norm = psd / np.sum(psd)
    entropy = np.nansum(psd_norm * np.log2(psd_norm))

    return -(entropy / np.log2(psd_norm.size))

@jit(forceobj=True)
def get_relative_entropy(x: np.ndarray) -> float:
    """
    source: entropy of continuous values relative to maximum entropy
    """
    # calculate the entropy of the time series
    entropy = _entropy(x)
    # calculate the maximum entropy
    max_entropy = np.log2(len(x))
    # calculate the relative entropy
    rel_entropy = entropy / max_entropy
    return rel_entropy

@jit(forceobj=True)
def get_lumpiness(x: np.ndarray, window_size: int = 8) -> float:
    """
    source: https://github.com/facebookresearch/Kats/blob/main/kats/tsfeatures/tsfeatures.py
        
    Calculating the lumpiness of time series.
    Lumpiness is defined as the variance of the chunk-wise variances.

    Args:
        x: The univariate time series array in the form of 1d numpy array.
        window_size: int; Window size to split the data into chunks for getting
            variances. Default value is 8.

    Returns:
        Lumpiness of the time series array.
    """

    v = [np.var(x_w) for x_w in np.array_split(x, len(x) // window_size + 1)]
    return np.var(v)

# stability
@jit(forceobj=True)
def get_stability(x: np.ndarray, window_size: int = 8) -> float:
    """
    source: https://github.com/facebookresearch/Kats/blob/main/kats/tsfeatures/tsfeatures.py

    Calculate the stability of time series.
    Stability is defined as the variance of chunk-wise means.

    Args:
        x: The univariate time series array in the form of 1d numpy array.
        window_size: int; Window size to split the data into chunks for getting
            variances. Default value is 8.

    Returns:
        Stability of the time series array.
    """

    v = [np.mean(x_w) for x_w in np.array_split(x, len(x) // window_size + 1)]
    return np.var(v)


@jit(forceobj=True)
def get_hurst(x: np.ndarray, lag_size: int = 30) -> float:
    """
    Getting: Hurst Exponent wiki: https://en.wikipedia.org/wiki/Hurst_exponent

    Args:
        x: The univariate time series array in the form of 1d numpy array.
        lag_size: int; Size for getting lagged time series data.

    Returns:
        The Hurst Exponent of the time series array
    """

    # Create the range of lag values
    lags = range(2, min(lag_size, len(x) - 1))

    # Calculate the array of the variances of the lagged differences
    tau = [np.std(np.asarray(x)[lag:] - np.asarray(x)[:-lag]) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] if not np.isnan(poly[0]) else 0

@jit(forceobj=True)
def get_slope_sign_switch_sum(ts_data: np.ndarray) -> float:
    """
    Get the sum of the number of times the slope of the time series changes sign.
    
    Args:   ts_data: np.array: time series data
            
    Output: float: sum of the number of times the slope of the time series changes sign
    """
    return np.sum(np.diff(np.sign(np.diff(ts_data))) != 0)/len(ts_data)

@jit(forceobj=True)
def get_linearity(x: np.ndarray) -> float:
    """
    Compute linearity feature: R square from a fitted linear regression.

    Args:
        x: The univariate time series array in the form of 1d numpy array.

    Returns:
        R square from a fitted linear regression.
    """

    _, _, r_value, _, _ = stats.linregress(np.arange(len(x)), x)
    return r_value**2


def _mann_kendall_test(data: np.ndarray) -> float:
    n = len(data)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data[j] - data[i])
            
    trend, _, _, z, Tau, s, var_s, slope, intercept = mk.original_test(data)
    trend = 1 if trend == 'increasing' else -1 if trend == 'decreasing' else 0        
    return s/n, z, Tau, s, var_s, slope, intercept, trend

@jit(forceobj=True)
def _wavelet_transform_feature(data: np.ndarray) -> float:
    # This is a placeholder for a real wavelet transform feature extraction.
    # For simplicity, we return the mean of the wavelet coefficients here.
    # Replace this with your actual wavelet feature extraction logic.
    widths = np.arange(1, 31)
    cwtmatr = cwt(data, ricker, widths)
    return np.mean(cwtmatr)

@jit(forceobj=True)
def get_het_arch(x: np.ndarray) -> float:
    """
    Compute Engle's test for autogregressive Conditional Heteroscedasticity (ARCH).

    reference: https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_arch.html
    Engle’s Test for Autoregressive Conditional Heteroscedasticity (ARCH)

    Args:
        x: The univariate time series array in the form of 1d numpy array.

    Returns:
        Lagrange multiplier test statistic
    """

    return het_arch(x, nlags=min(10, len(x) // 5))[0]

@jit(forceobj=True)
def get_stl_features(
    x: np.ndarray,
    period: int = 7,
    extra_args: Optional[Dict[str, bool]] = None,
    default_status: bool = True,
) -> Dict[str, float]:
    """
    Calculate STL based features for a time series.

    Args:
        x: The univariate time series array in the form of 1d numpy array.
        period: int; Period parameter for performing seasonality trend
            decomposition using LOESS with statsmodels.
        extra_args: A dictionary containing information for disabling
            calculation of a certain feature. If None, no feature is
            disabled.
        default_status: Default status of the switch for calculate the
            features or not.

    Returns:
        Seasonality features including strength of trend, seasonality,
        spikiness, peak/trough.
    """

    stl_features = {}

    # STL decomposition
    res = STL(x, period=period).fit()

    # strength of trend
    if extra_args is not None and extra_args.get("trend_strength", default_status):
        stl_features["trend_strength"] = 1 - np.var(res.resid) / np.var(
            res.trend + res.resid
        )

    # strength of seasonality
    if extra_args is not None and extra_args.get(
        "seasonality_strength", default_status
    ):
        stl_features["seasonality_strength"] = 1 - np.var(res.resid) / np.var(
            res.seasonal + res.resid
        )

    # spikiness: variance of the leave-one-out variances of the remainder component
    if extra_args is not None and extra_args.get("spikiness", default_status):
        resid_array = np.repeat(
            np.array(res.resid)[:, np.newaxis], len(res.resid), axis=1
        )
        resid_array[np.diag_indices(len(resid_array))] = np.NaN
        stl_features["spikiness"] = np.var(np.nanvar(resid_array, axis=0))

    # location of peak
    if extra_args is not None and extra_args.get("peak", default_status):
        stl_features["peak"] = np.argmax(res.seasonal[:period])

    # location of trough
    if extra_args is not None and extra_args.get("trough", default_status):
        stl_features["trough"] = np.argmin(res.seasonal[:period])

    return stl_features