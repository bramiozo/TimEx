import numpy as np
import pandas as pd
import logging
import joblib

from scipy.stats import skew, kurtosis, entropy as _entropy, linregress, median_abs_deviation
from scipy.fft import rfft, rfftfreq, dct
from scipy.signal import cwt, ricker, periodogram, welch, find_peaks
from scipy.linalg import toeplitz
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
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import antropy as ant
from cesium import featurize

from tqdm import tqdm
import sys, os
import pandas as pd 

from numba import jit
import gc

from collections import defaultdict
from itertools import groupby
from typing import Optional, Dict, List, Tuple

from time import sleep
from time import time as timer
from datetime import time
from datetime import datetime
import antropy as ant
import nolds

from fastdtw import fastdtw

import wavelets
#TODO: add interpretable feature mappings 
# e.g. {'slopes':{}, 'periodicity':{}, 'entropy':{}, 'amplitude':{}, 'trend':{}, 'nonlinearity':{}, 'spikes':{}, 'crossings':{}, 'energy':{}, 'statistics':{}, 'distribution':{}, 'autocorrelation':{}, 'stability':{}, 'linearity':{}, 'complexity':{}, 'nonlinear':{}, 'chaos':{}, 'misc':{}} 
    
## Ideas for extracts
# 'complexity': how many fourier components are needed to describe the signal with a certain accuracy
# 'complexity': (1) spline complexity, how many splines are needed to describe the signal with a certain accuracy, (2) spline-series of knots etc.
# 'periodicity': peak_counter

# TODO: extract shapelets
# https://tslearn.readthedocs.io/en/stable/user_guide/shapelets.html


# add Cesium features
DEFAULT_CESIUM_FEATURES = ["amplitude", "percent_beyond_1_std", 
                          "median_absolute_deviation", "percent_close_to_median",
                          "weighted_average", "all_times_nhist_numpeaks", 
                          "all_times_nhist_peak_1_to_2", "all_times_nhist_peak_val",
                          "avg_double_to_single_step", "avg_err", "avgt",
                          "anderson_darling",  "shapiro_wilk"]

ANTROPY_FEATURES = ["perm_entropy", "spectral_entropy",
                    "svd_entropy", "app_entropy", "sample_entropy",
                    "lziv_complexity", "num_zerocross",
                    "hjorth_params", "petrosian_fd",  "katz_fd",
                    "higuchi_fd", "detrended_fluctuation"
                ]
KATZ_FEATURES = ['linearity','het_arch', 'stl_features', 'acfpacf_features',
                 'flat_spots', 'unitroot_kpss', 'holt_params']

TSFEL_FEATURES = ['positive_turning', 'negative_turning',
                'travelled_distance', 'auc',
                'lempel_ziv', 'median_abs_deviation',
                'fundamental_frequency', 'spectral_roll_on',
                'spectral_roll_off', 'spectral_positive_turning',
                'spectral_variation', 'spectral_slope',
                'spectral_spread', 'spectral_skewness',
                'spectral_kurtosis', 'spectral_decrease',
                'spectral_centroid','spectral_distance',
                'spectral_entropy','median_frequency',
                'max_frequency', 'max_power_spectrum',
                'wavelet_entropy','wavelet_abs_mean',
                'wavelet_std','wavelet_var',
                'wavelet_energy', 'mfcc',
                'lpcc']

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = timer()
        result = func(*args, **kwargs)
        end_time = timer()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    return wrapper


def get_crossectional(tsdf: pd.DataFrame,
                      id_col='ID',
                      val_col='eGFR_CKDEpi2012', 
                      time_col='Time_col',
                      custom_features=True,
                      tsfresh_features=False,
                      catch22_features=False, 
                      cesium_features=False,
                      antropy_features=False,
                      nolds_features=False,
                      katz_features=False,
                      tsfel_features=False):
    DURATIONS = defaultdict(dict)

    ResDict = {}
    if custom_features == True:
        CustomExtractor = Extractor()
        CustomExtractor.fit(tsdf,
                            id_col=id_col,
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_custom = CustomExtractor.transform()
        DURATIONS['CustomExtractor'] = CustomExtractor.duration_dict
        ResDict['custom'] = ts_data_agg_custom.set_index('id')

    if tsfresh_features ==True:
        FreshExtractor = TsFreshExtractor()
        FreshExtractor.fit(tsdf, 
                            id_col=id_col, 
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_fresh = FreshExtractor.transform()
        DURATIONS['FreshExtractor'] = FreshExtractor.duration_dict
        ResDict['tsfresh'] = ts_data_agg_fresh

    if catch22_features==True:
        Catch22Extract  = Catch22Extractor()
        Catch22Extract.fit(tsdf, 
                    id_col=id_col, 
                    val_col=val_col,
                    time_col=time_col)
        ts_data_agg_catch22 = Catch22Extract.transform()
        DURATIONS['Catch22Extractor'] = Catch22Extract.duration_dict
        ResDict['catch22'] = ts_data_agg_catch22

    if cesium_features==True:
        CesiumExtract = CesiumExtractor()
        CesiumExtract.fit(tsdf, 
                            id_col=id_col, 
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_cesium = CesiumExtract.transform()
        DURATIONS['CesiumExtract'] = CesiumExtract.duration_dict
        ResDict['cesium'] = ts_data_agg_cesium
    
    if antropy_features==True:
        AntropyExtract = AntropyExtractor()
        AntropyExtract.fit(tsdf, 
                            id_col=id_col, 
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_antropy = AntropyExtract.transform()
        DURATIONS['AntropyExtract'] = AntropyExtract.duration_dict
        ResDict['antropy'] = ts_data_agg_antropy
    
    if nolds_features==True:
        NoldsExtract = NoldsExtractor()
        NoldsExtract.fit(tsdf, 
                            id_col=id_col, 
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_nolds = NoldsExtract.transform()
        DURATIONS['NoldsExtract'] = NoldsExtract.duration_dict
        ResDict['nolds'] = ts_data_agg_nolds

    if katz_features==True:
        KatzExtract = KatzExtractor()
        KatzExtract.fit(tsdf,
                            id_col=id_col,
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_katz = KatzExtract.transform()
        DURATIONS['KatzExtract'] = KatzExtract.duration_dict
        ResDict['katz'] = ts_data_agg_katz

    if tsfel_features==True:
        TSfelExtract = TSFelExtractor()
        TSfelExtract.fit(tsdf,
                            id_col=id_col,
                            val_col=val_col,
                            time_col=time_col)
        ts_data_agg_tsfel = TSfelExtract.transform()
        DURATIONS['TSfelExtract'] = TSfelExtract.duration_dict
        ResDict['tsfel'] = ts_data_agg_tsfel

    # merge
    iterator = iter(ResDict.items())
    k, final = next(iterator)
    final.columns = [f'{c}_{k}' for c in final.columns]
    for k, df in iterator:
        final = final.join(df, how='inner', rsuffix='_'+k)

    return final, DURATIONS

class Extractor:
    def __init__(self):
        self.features = {}
        self.duration_dict = defaultdict(float)
    
    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        #TODO: streamline kwargs for internal functions...
        df = df.sort_values(by=[id_col, time_col])
        num_series = df[id_col].nunique()
        # Iterate over all unique IDs in the DataFrame
        for _id in tqdm(df[id_col].unique()):
            # Extract the time series data for the current ID
            ts_data = df[df[id_col] == _id][val_col].values
            
            # Calculate features

            mean_, time_mean = time_function(np.mean)(ts_data)
            min_, time_min = time_function(np.min)(ts_data)
            max_, time_max = time_function(np.max)(ts_data)
            variance, time_var = time_function(np.var)(ts_data)
            skewness, time_skewness = time_function(skew)(ts_data)
            kurtosis_value, time_kurtosis = time_function(kurtosis)(ts_data)

            entropy_per_rel, time_entropy_per_rel = time_function(get_spectral_entropy)(ts_data, freq=1)
            entropy_rel, time_entropy_rel = time_function(get_relative_entropy)(ts_data)
            lumpiness, time_lumpiness = time_function(get_lumpiness)(ts_data, window_size=8)
            lump_stability, time_lump_stability = time_function(get_stability)(ts_data, window_size=8)
            hurst_exponent, time_hurst_exponent = time_function(get_hurst)(ts_data, lag_size=8)
            rel_slope_sign_switch_sum, time_rel_ssss = time_function(get_slope_sign_switch_sum)(ts_data)

            (mk_s, mk_z, mk_Tau, mk_ss, mk_var_s, mk_slope, mk_intercept, mk_trend), time_mankendall = \
                        time_function(_mann_kendall_test)(ts_data)

            wavelet_transform_feature, time_wavelet_transform = time_function(_wavelet_transform_feature)(ts_data)
            psd_int, time_psdint = time_function(_psd_int)(ts_data, integrator='trapezoidal')

            avg_3rd_diff, time_avg3rdDiff = time_function(avg_3rd_order)(ts_data)
            avg_2nd_diff, time_avg2ndDiff = time_function(avg_2nd_order)(ts_data)
            avg_1st_diff, time_avg1stDiff = time_function(avg_1st_order)(ts_data)
            entr_1st_diff, time_entr1stDiff = time_function(diff_entropy_1st)(ts_data)
            entr_2nd_diff, time_entr2ndDiff = time_function(diff_entropy_2nd)(ts_data)
            entr_3rd_diff, time_entr3rdDiff = time_function(diff_entropy_3rd)(ts_data)

            shape_comparisons, time_shape = time_function(shape_compare)(ts_data)

            _peak_over_mean, time_peak_o_mean = time_function(peak_over_mean)(ts_data)
            _peak_over_median, time_peak_o_median = time_function(peak_over_median)(ts_data)
            _first_gradient, time_first_gradient = time_function(first_gradient)(ts_data)
            _second_gradient, time_second_gradient = time_function(second_gradient)(ts_data)
            _last_gradient, time_last_gradient = time_function(last_gradient)(ts_data)

            cumuvalues, time_cumuvalues = time_function(cumuvals)(ts_data)

            self.duration_dict['time_mean'] += time_mean
            self.duration_dict['time_min'] += time_min
            self.duration_dict['time_max'] += time_max
            self.duration_dict['time_var'] += time_var
            self.duration_dict['time_skewness'] += time_skewness
            self.duration_dict['time_kurtosis'] += time_kurtosis
            self.duration_dict['time_entropy_per_rel'] += time_entropy_per_rel
            self.duration_dict['time_entropy_rel'] += time_entropy_rel
            self.duration_dict['time_lumpiness'] += time_lumpiness
            self.duration_dict['time_lump_stability'] += time_lump_stability
            self.duration_dict['time_hurst_exponent'] += time_hurst_exponent
            self.duration_dict['time_rel_ssss'] += time_rel_ssss
            self.duration_dict['time_mankendall'] += time_mankendall
            self.duration_dict['time_wavelet_transform'] += time_wavelet_transform
            self.duration_dict['time_psdint'] += time_psdint
            self.duration_dict['time_avg3rdDiff'] += time_avg3rdDiff
            self.duration_dict['time_avg2ndDiff'] += time_avg2ndDiff
            self.duration_dict['time_avg1stDiff'] += time_avg1stDiff
            self.duration_dict['time_entr1stDiff'] += time_entr1stDiff
            self.duration_dict['time_entr2ndDiff'] += time_entr2ndDiff
            self.duration_dict['time_entr3rdDiff'] += time_entr3rdDiff
            self.duration_dict['time_shape'] += time_shape
            self.duration_dict['time_peak_o_mean'] += time_peak_o_mean
            self.duration_dict['time_peak_o_median'] += time_peak_o_median
            self.duration_dict['time_first_gradient'] += time_first_gradient
            self.duration_dict['time_second_gradient'] += time_second_gradient
            self.duration_dict['time_last_gradient'] += time_last_gradient
            self.duration_dict['time_cumuvalues'] += time_cumuvalues

            # Store the features for the current ID
            res_dict = {
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
                'rel_slope_sign_switch_sum': rel_slope_sign_switch_sum,
                'avg_3rd_diff': avg_3rd_diff,
                'avg_2nd_diff': avg_2nd_diff,
                'avg_1st_diff': avg_1st_diff,
                'entr_1st_diff': entr_1st_diff,
                'entr_2nd_diff': entr_2nd_diff,
                'entr_3rd_diff': entr_3rd_diff,
                'peak_over_mean': _peak_over_mean,
                'peak_over_median': _peak_over_median,
                'first_gradient': _first_gradient,
                'second_gradient': _second_gradient,
                'last_gradient': _last_gradient,
            }
            
            res_dict.update(shape_comparisons)
            res_dict.update(cumuvalues)

            self.features[_id] = res_dict
        self.duration_dict = {k:v/num_series for k,v in self.duration_dict.items()}
    def transform(self):
        # Convert the features dictionary to a DataFrame
        features_df = pd.DataFrame.from_dict(self.features, orient='index').reset_index()
        features_df.rename(columns={'index': 'id'}, inplace=True)
        return features_df
    


class TsFreshExtractor:
    def __init__(self):
        self.features = {}
        self.duration_dict = defaultdict(float)

    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        # Extract features
        df = df.sort_values(by=[id_col, time_col])
        extracted_features, time_tsfresh = \
                time_function(extract_features)(df,
                                 column_id=id_col, 
                                 column_sort=time_col,
                                 impute_function=impute,
                                 default_fc_parameters=ComprehensiveFCParameters())
        self.features = extracted_features
        self.duration_dict['total'] = time_tsfresh
    
    def transform(self):
        return self.features

#from pycatch22 import catch22_all
class Catch22Extractor:
    def __init__(self):
        self.features = {}
        self.duration_dict = defaultdict(float)

    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        # Extract features
        df = df.sort_values(by=[id_col, time_col])
        _features = {}
        for _id in tqdm(df[id_col].unique()):
            ts_data = df[df[id_col] == _id][val_col].values
            extracted_features, time_total = time_function(pycatch22.catch22_all)(ts_data)
            _features[_id] = dict(zip(extracted_features['names'], extracted_features['values']))
            self.duration_dict['total'] += time_total
        self.features = _features
        self.duration_dict['total'] /= df.shape[0]
    
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
        self.duration_dict = defaultdict(float)

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
            feats, time_total = time_function(featurize.featurize_time_series)(
                                            times=times,
                                            values=vals,
                                            errors=None,
                                            features_to_use=self.features_to_use,
                                            )
            feats.columns = feats.columns.droplevel(-1)
            feats_dict = feats.to_dict()
            _features[_id] = {k:v[0] for k,v in feats_dict.items()}
            self.duration_dict['total'] += time_total
        self.features = _features
        self.duration_dict['total'] /= df.shape[0]

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
        self.duration_dict = defaultdict(float)

    def tsfeatures(self, ts_data, features_to_use: list = None):
        feature_functions = {
            'perm_entropy': ant.perm_entropy,
            'spectral_entropy': ant.spectral_entropy,
            'svd_entropy': ant.svd_entropy,
            'app_entropy': ant.app_entropy,
            'sample_entropy': ant.sample_entropy,
            'lziv_complexity': ant.lziv_complexity,
            'num_zerocross': ant.num_zerocross,
            'hjorth_params': ant.hjorth,
            'petrosian_fd': ant.petrosian_fd,
            'katz_fd': ant.katz_fd,
            'higuchi_fd': ant.higuchi_fd,
            'detrended_fluctuation': ant.detrended_fluctuation
        }

        res = {}
        for f in features_to_use:
            if f in feature_functions:
                fun = feature_functions[f]
                if f == 'hjorth_params':
                    hjorth, time_elapsed = time_function(fun)(ts_data)
                    res['activity'] = hjorth[0]
                    res['mobility'] = hjorth[1]
                    res['complexity'] = hjorth[2]
                else:
                    result, time_elapsed = time_function(fun)(ts_data)
                    res[f] = result
                self.duration_dict[f] += time_elapsed
        return res
     
    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        assert(self.features_to_use is not None), "Please provide a list of features to use"
        df = df.sort_values(by=[id_col, time_col])
        _features = {}
        for _id in tqdm(df[id_col].unique()):
            ts_data = df[df[id_col] == _id][val_col].values
            feats = self.tsfeatures(ts_data, features_to_use=self.features_to_use)
            _features[_id] = feats
        self.duration_dict = {k: v/df.shape[0] for k,v in self.duration_dict.items()}
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
        self.emb_dims = [1, 2, 3, 4] if emb_dims is None else emb_dims
        self.features_to_use = features_to_use if features_to_use \
                                else ['lyap_e', 'corr_dim']
        self.min_ts_len = min_ts_len
        self.duration_dict = defaultdict(float)
    
    def tsfeatures(self,ts_data, features_to_use: list=None, min_len=50):
        res = {}
        for f in features_to_use:
            if f == 'lyap_e':
                if ts_data.shape[0] < min_len:
                    v_ = np.tile(ts_data, min_len//ts_data.shape[0] + 1)
                else:
                    v_ = ts_data       
                _res, time_lyap = time_function(nolds.lyap_e(v_))
                for nd, res_ in enumerate(_res):
                    res[f'lyap_e_{nd}'] = res_
                self.duration_dict['lyap'] += time_lyap
            elif f == 'corr_dim':
                for edim in self.emb_dims:
                    res[f'corr_dim_{edim}'], time_corr = (
                        time_function(nolds.corr_dim(ts_data, emb_dim=edim)))
                    self.duration_dict[f'corr_dim_{edim}'] += time_corr
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
        self.duration_dict = {k:v/df.shape[0] for k,v in self.duration_dict.items()}
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

    # groupby in functiontools ?

    # TsFeatures.get_holt_params
    # TsFeatures.get_cusum_detector
    # TsFeatures.get_trend_detector

    def __init__(self,
                 features_to_use: list = None):
        self.features_to_use = features_to_use if features_to_use \
            else KATZ_FEATURES
        self.duration_dict = defaultdict(float)

    def katzfeatures(self, ts_data, features_to_use: list = None):
        feature_functions = {
            'linearity': get_linearity,
            'het_arch': get_het_arch,
            'stl_features': get_stl_features,
            'acfpacf_features': get_acfpacf_features,
            'flat_spots': get_flat_spots,
            'unitroot_kpss': get_unitroot_kpss,
            'holt_params': get_holt_params,
        }

        res = {}
        for f in features_to_use:
            if f in feature_functions:
                fun = feature_functions[f]
                result, time_elapsed = time_function(fun)(ts_data)
                if f in ['stl_features', 'acfpacf_features', 'holt_params']:
                    res.update(result)
                else:
                    res[f] = result
                self.duration_dict[f] += time_elapsed
        return res

    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        assert (self.features_to_use is not None), "Please provide a list of features to use"
        df = df.sort_values(by=[id_col, time_col])
        _features = {}
        for _id in tqdm(df[id_col].unique()):
            ts_data = df[df[id_col] == _id][val_col].values
            feats = self.katzfeatures(ts_data, features_to_use=self.features_to_use)
            _features[_id] = feats
        self.duration_dict = {k: v/df.shape[0] for k, v in self.duration_dict.items()}
        self.features = _features

    def transform(self):
        # Convert the features dictionary to a DataFrame
        features_df = pd.DataFrame.from_dict(self.features,
                                             orient='index').reset_index()
        features_df = features_df.rename(columns={'index': 'id'})
        features_df = features_df.set_index('id')
        return features_df

class TSFelExtractor:
    def __init__(self,
                 features_to_use: list = None):
        self.features_to_use = features_to_use if features_to_use \
            else TSFEL_FEATURES
        self.features = None
        self.duration_dict = defaultdict(float)
    @staticmethod
    def tsfelfeatures(self, ts_data, features_to_use: list = None):
        feature_functions = {
            'positive_turning': positive_turning,
            'negative_turning': negative_turning,
            'travelled_distance': travelled_distance,
            'auc': auc,
            'lempel_ziv': lempel_ziv,
            'median_abs_deviation': median_abs_deviation,
            'fundamental_frequency': fundamental_frequency,
            'spectral_roll_on': spectral_roll_on,
            'spectral_roll_off': spectral_roll_off,
            'spectral_positive_turning': spectral_positive_turning,
            'spectral_variation': spectral_variation,
            'spectral_slope': spectral_slope,
            'spectral_spread': spectral_spread,
            'spectral_skewness': spectral_skewness,
            'spectral_kurtosis': spectral_kurtosis,
            'spectral_decrease': spectral_decrease,
            'spectral_centroid': spectral_centroid,
            'spectral_distance': spectral_distance,
            'spectral_entropy': spectral_entropy,
            'median_frequency': median_frequency,
            'max_frequency': max_frequency,
            'max_power_spectrum': max_power_spectrum,
            'wavelet_entropy': wavelets.wavelet_entropy,
            'wavelet_abs_mean': wavelets.wavelet_abs_mean,
            'wavelet_std': wavelets.wavelet_std,
            'wavelet_var': wavelets.wavelet_var,
            'wavelet_energy': wavelets.wavelet_energy,
            'mfcc': mfcc,
            'lpcc': lpcc
        }
        res = {}
        for f in features_to_use:
            if f in feature_functions:
                fun = feature_functions[f]
                result, time_elapsed = time_function(fun)(ts_data)
                if f in ['wavelet_abs_mean', 'wavelet_std', 'wavelet_var',
                         'wavelet_energy', 'mfcc', 'lpcc']:
                    res.update(result)
                else:
                    res[f] = result
                self.duration_dict[f] += time_elapsed
        return res

    def fit(self, df, id_col='id', val_col='value', time_col='dt'):
        assert (self.features_to_use is not None), "Please provide a list of features to use"
        df = df.sort_values(by=[id_col, time_col])
        _features = {}
        for _id in tqdm(df[id_col].unique()):
            ts_data = df[df[id_col] == _id][val_col].values
            feats = self.tsfelfeatures(ts_data, features_to_use=self.features_to_use)
            _features[_id] = feats
        self.features = _features
        self.duration_dict = {k: v/df.shape[0] for k,v in self.duration_dict.items()}

    def transform(self):
        # Convert the features dictionary to a DataFrame
        features_df = pd.DataFrame.from_dict(self.features,
                                             orient='index').reset_index()
        features_df = features_df.rename(columns={'index': 'id'})
        features_df = features_df.set_index('id')
        return features_df


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

    _, _, r_value, _, _ = linregress(np.arange(len(x)), x)
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
    period: int = 3,
    window: int = 4,
    robust: bool = False,
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
    
    assert(window>=period), "The smoothing window should be larger than or equal to the period"

    stl_features = {}

    # STL decomposition
    res = STL(x, period=period, seasonal=(window+abs(window%2-1)), robust=robust).fit()
    
    stl_features["var_trend"] = np.var(res.trend)
    stl_features["var_res"] = np.var(res.resid)

    # strength of trend
    stl_features["trend_strength"] = 1 - np.var(res.resid) / np.var(
            res.trend + res.resid
        )

    stl_features["seasonality_strength"] = 1 - np.var(res.resid) / np.var(
            res.seasonal + res.resid
        )

    # spikiness: variance of the leave-one-out variances of the remainder component
    resid_array = np.repeat(
        np.array(res.resid)[:, np.newaxis], len(res.resid), axis=1
    )
    resid_array[np.diag_indices(len(resid_array))] = np.NaN
    stl_features["spikiness"] = np.var(np.nanvar(resid_array, axis=0))

    # location of peak
    stl_features["peak"] = np.argmax(res.seasonal[:period])

    # location of trough
    stl_features["trough"] = np.argmin(res.seasonal[:period])

    return stl_features

@jit(forceobj=True)
def get_acf_features(
    extra_args: Dict[str, bool],
    default_status: bool,
    y_acf_list: List[float],
    diff1y_acf_list: List[float],
    diff2y_acf_list: List[float],
) -> Dict[str,float]:
    """
    Aggregating extracted ACF features from get_acfpacf_features function.

    Args:
        extra_args: A dictionary containing information for disabling calculation
            of a certain feature. If None, no feature is disabled.
        default_status: Default status of the switch for calculate the
            features or not.
        y_acf_list: List of ACF values acquired from original time series.
        diff1y_acf_list: List of ACF values acquired from differenced time series.
        diff2y_acf_list: List of ACF values acquired from twice differenced
            time series.

    Returns:
        Auto-correlation function (ACF) features.
    """

    y_acf1 = y_acf5 = diff1y_acf1 = diff1y_acf5 = diff2y_acf1 = np.nan
    diff2y_acf5 = seas_acf1 = np.nan

    # y_acf1: first ACF value of the original series
    if extra_args.get("y_acf1", default_status):
        y_acf1 = y_acf_list[0]

    # y_acf5: sum of squares of first 5 ACF values of original series
    if extra_args.get("y_acf5", default_status):
        y_acf5 = np.sum(np.asarray(y_acf_list)[:5] ** 2)

    # diff1y_acf1: first ACF value of the differenced series
    if extra_args.get("diff1y_acf1", default_status):
        diff1y_acf1 = diff1y_acf_list[0]

    # diff1y_acf5: sum of squares of first 5 ACF values of differenced series
    if extra_args.get("diff1y_acf5", default_status):
        diff1y_acf5 = np.sum(np.asarray(diff1y_acf_list)[:5] ** 2)

    # diff2y_acf1: first ACF value of the twice-differenced series
    if extra_args.get("diff2y_acf1", default_status):
        diff2y_acf1 = diff2y_acf_list[0]

    # diff2y_acf5: sum of squares of first 5 ACF values of twice-differenced series
    if extra_args.get("diff2y_acf5", default_status):
        diff2y_acf5 = np.sum(np.asarray(diff2y_acf_list)[:5] ** 2)

    # Autocorrelation coefficient at the first seasonal lag.
    if extra_args.get("seas_acf1", default_status):
        seas_acf1 = y_acf_list[-1]

    return {
        'acf1': y_acf1,
        'acf5': y_acf5,
        'diff1y_acf1': diff1y_acf1,
        'diff1y_acf5': diff1y_acf5,
        'diff2y_acf1': diff2y_acf1,
        'diff2y_acf5': diff2y_acf5,
        'seas_acf1': seas_acf1,
    }
        
def get_pacf_features(
    extra_args: Dict[str, bool],
    default_status: bool,
    y_pacf_list: List[float],
    diff1y_pacf_list: List[float],
    diff2y_pacf_list: List[float],
) -> Dict[str,float]:
    """
    Aggregating extracted PACF features from get_acfpacf_features function.

    Args:
        extra_args: A dictionary containing information for disabling calculation
            of a certain feature. If None, no feature is disabled.
        default_status: Default status of the switch for calculate the
            features or not.
        y_pacf_list: List of PACF values acquired from original time series.
        diff1y_pacf_list: List of PACF values acquired from differenced time series.
        diff2y_pacf_list: List of PACF values acquired from twice differenced
            time series.

    Returns:
        Partial auto-correlation function (PACF) features.
    """

    y_pacf5 = diff1y_pacf5 = diff2y_pacf5 = seas_pacf1 = np.nan

    # y_pacf5: sum of squares of first 5 PACF values of original series
    if extra_args.get("y_pacf5", default_status):
        y_pacf5 = np.nansum(np.asarray(y_pacf_list)[:5] ** 2)

    # diff1y_pacf5: sum of squares of first 5 PACF values of differenced series
    if extra_args.get("diff1y_pacf5", default_status):
        diff1y_pacf5 = np.nansum(np.asarray(diff1y_pacf_list)[:5] ** 2)

    # diff2y_pacf5: sum of squares of first 5 PACF values of twice-differenced series
    if extra_args.get("diff2y_pacf5", default_status):
        diff2y_pacf5 = np.nansum(np.asarray(diff2y_pacf_list)[:5] ** 2)

    # Patial Autocorrelation coefficient at the first seasonal lag.
    if extra_args.get("seas_pacf1", default_status):
        seas_pacf1 = y_pacf_list[-1]

    return {
        'pacf5': y_pacf5,
        'diff1y_pacf5': diff1y_pacf5,
        'diff2y_pacf5': diff2y_pacf5,
        'seas_pacf1': seas_pacf1,
    }

@jit(forceobj=True)
def get_acfpacf_features(
        x: np.ndarray,
        acfpacf_lag: int = 6,
        period: int = 7,
        extra_args: Optional[Dict[str, bool]] = None,
        default_status: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate ACF and PACF based features. Calculate seasonal ACF, PACF based features.

        Reference: https://stackoverflow.com/questions/36038927/whats-the-difference-between-pandas-acf-and-statsmodel-acf
        R code: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html
        Paper: Meta-learning how to forecast time series

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            acfpacf_lag: int; Largest lag number for returning ACF/PACF features
                via statsmodels.
            period: int; Seasonal period.
            extra_args: A dictionary containing information for disabling
                calculation of a certain feature. If None, no feature is disabled.
            default_status: Default status of the switch for calculate the
                features or not.

        Returns:
            Aggregated ACF, PACF features.
        """

        acfpacf_features = {
            "y_acf1": np.nan,
            "y_acf5": np.nan,
            "diff1y_acf1": np.nan,
            "diff1y_acf5": np.nan,
            "diff2y_acf1": np.nan,
            "diff2y_acf5": np.nan,
            "y_pacf5": np.nan,
            "diff1y_pacf5": np.nan,
            "diff2y_pacf5": np.nan,
            "seas_acf1": np.nan,
            "seas_pacf1": np.nan,
        }
        if len(x) < 10 or len(x) < period or len(np.unique(x)) == 1:
            msg = (
                "Length is shorter than period, or constant time series, "
                "unable to calculate acf/pacf features"
            )
            logging.error(msg)
            return acfpacf_features

        nlag = min(acfpacf_lag, len(x) - 2)

        diff1x = [x[i] - x[i - 1] for i in range(1, len(x))]
        diff2x = [diff1x[i] - diff1x[i - 1] for i in range(1, len(diff1x))]

        y_acf_list = acf(x, fft=True, nlags=period)[1:]
        diff1y_acf_list = acf(diff1x, fft=True, nlags=nlag)[1:]
        diff2y_acf_list = acf(diff2x, fft=True, nlags=nlag)[1:]

        y_pacf_list = pacf(x, nlags=period)[1:]

        if (
            _yule_walker_determinant(diff1x) == 0
            or _yule_walker_determinant(diff2x) == 0
        ):
            logging.warning(
                "Could not generate acfpacf features because input matrix is singular."
            )
            return acfpacf_features

        diff1y_pacf_list = pacf(diff1x, nlags=nlag)[1:]
        diff2y_pacf_list = pacf(diff2x, nlags=nlag)[1:]

        (
            acfpacf_features["y_acf1"],
            acfpacf_features["y_acf5"],
            acfpacf_features["diff1y_acf1"],
            acfpacf_features["diff1y_acf5"],
            acfpacf_features["diff2y_acf1"],
            acfpacf_features["diff2y_acf5"],
            acfpacf_features["seas_acf1"],
        ) = get_acf_features(
            extra_args,
            default_status,
            y_acf_list,
            diff1y_acf_list,
            diff2y_acf_list,
        )

        # getting PACF features
        (
            acfpacf_features["y_pacf5"],
            acfpacf_features["diff1y_pacf5"],
            acfpacf_features["diff2y_pacf5"],
            acfpacf_features["seas_pacf1"],
        ) = get_pacf_features(
            extra_args,
            default_status,
            y_pacf_list,
            diff1y_pacf_list,
            diff2y_pacf_list,
        )
        return acfpacf_features

def _yule_walker_determinant(x_list: List[float]) -> float:
        x = np.array(x_list, dtype=np.float64)

        if x.ndim > 1 and x.shape[1] != 1:
            raise ValueError("expecting a vector to estimate AR parameters")

        x -= x.mean()
        r = np.zeros(2, np.float64)
        r[0] = (x**2).mean()
        r[1] = (x[0:-1] * x[1:]).mean()
        R = toeplitz(r[:-1])
        return np.linalg.det(R)

@jit(forceobj=True)
def get_flat_spots(x: np.ndarray, nbins: int = 10) -> int:
    """
    Getting flat spots: Maximum run-lengths across equally-sized segments of time series

    Args:
        x: The univariate time series array in the form of 1d numpy array.
        nbins: int; Number of bins to segment time series data into.

    Returns:
        Maximum run-lengths across segmented time series array.
    """

    if len(x) <= nbins:
        msg = (
            "Length of time series is shorter than nbins, unable to "
            "calculate flat spots feature"
        )
        logging.error(msg)
        return np.nan

    max_run_length = 0
    window_size = int(len(x) / nbins)
    for i in range(0, len(x), window_size):
        run_length = np.max(
            [len(list(v)) for k, v in groupby(x[i : i + window_size])]
        )
        if run_length > max_run_length:
            max_run_length = run_length
    return max_run_length

@jit(forceobj=True)
def avg_3rd_order(x: np.ndarray)-> float:
    """
        Requires a minimum of 4 points    
    """
    assert(x.shape[0]>4), "We would like a minimum of 5 points for averaging the 2nd diff"
    xd3 = np.diff(x, n=3)
    return np.nanmean(xd3)

@jit(forceobj=True)
def avg_2nd_order(x: np.ndarray)-> float:
    """
        Requires a minimum of 4 points    
    """
    assert(x.shape[0]>3), "We would like a minimum of 4 points for averaging the 2nd diff"
    xd2 = np.diff(x, n=2)
    return np.nanmean(xd2)

@jit(forceobj=True)
def avg_1st_order(x: np.ndarray)-> float:
    """
        Requires a minimum of 3 points    
    """
    assert(x.shape[0]>2), "We would like a minimum of 3 points for averaging the 1nd diff"
    xd1 = np.diff(x, n=1)
    return np.nanmean(xd1)

@jit(forceobj=True)
def diff_entropy_1st(x:np.ndarray) -> float:
    assert(x.shape[0]>1), "We would like a minimum of 3 points for averaging the 1nd diff"
    xd1 = np.diff(x, n=1)
    return get_relative_entropy(xd1)

@jit(forceobj=True)
def diff_entropy_2nd(x:np.ndarray) -> float:
    assert(x.shape[0]>2), "We would like a minimum of 3 points for averaging the 1nd diff"
    xd2 = np.diff(x, n=2)
    return get_relative_entropy(xd2)

@jit(forceobj=True)
def diff_entropy_3rd(x:np.ndarray) -> float:
    assert(x.shape[0]>3), "We would like a minimum of 3 points for averaging the 1nd diff"
    xd3 = np.diff(x, n=3)
    return get_relative_entropy(xd3)

@jit(forceobj=True)
def peak_over_mean(x: np.ndarray) -> float:
    max_v = x.max()
    mean_v = x.mean()
    return max_v/mean_v

@jit(forceobj=True)
def peak_over_median(x: np.ndarray) -> float:
    max_v = x.max()
    median_v = np.median(x)
    return max_v/median_v

@jit(forceobj=True)
def first_gradient(x: np.ndarray) -> float:
    return np.gradient(x)[0]

@jit(forceobj=True)
def second_gradient(x: np.ndarray) -> float:
    return np.gradient(x)[1]
@jit(forceobj=True)
def last_gradient(x: np.ndarray) -> float:
    return np.gradient(x)[-1]

@jit(forceobj=True)
def get_unitroot_kpss(x: np.ndarray) -> float:
    """
    Get the test statistic based on KPSS test.

    Test a null hypothesis that an observable time series is stationary
    around a deterministic trend. A vector comprising the statistic for the
    KPSS unit root test with linear trend and lag one
    Wiki: https://en.wikipedia.org/wiki/KPSS_test

    Args:
        x: The univariate time series array in the form of 1d numpy array.

    Returns:
        Test statistics acquired using KPSS test.
    """
    return kpss(x, regression="ct", nlags=1)[0]

def get_holt_params(
    x: np.ndarray,
    extra_args: Optional[Dict[str, bool]] = None,
    default_status: bool = True,
) -> Dict[str, float]:
    """
    Estimates the smoothing parameters for Holt's linear trend model.

    * 'alpha': Level parameter of the Holt model.
    * 'beta': Trend parameter of the Hold model.

    Args:
        x: The univariate time series array in the form of 1d numpy array.
        extra_args: A dictionary containing information for disabling
            calculation of a certain feature. If None, no feature is disabled.
        default_status: Default status of the switch for calculate the
            features or not.

    Returns:
        Level and trend parameter of a fitted Holt model.
    """

    holt_params_features = {"holt_alpha": np.nan, "holt_beta": np.nan}
    try:
        m = ExponentialSmoothing(x, trend="add", seasonal=None).fit()
        if extra_args is not None and extra_args.get("holt_alpha", default_status):
            holt_params_features["holt_alpha"] = m.params["smoothing_level"]
        if extra_args is not None and extra_args.get("holt_beta", default_status):
            holt_params_features["holt_beta"] = m.params["smoothing_trend"]
    except Exception as e:
        logging.warning(f"Holt Linear failed {e}")
    return holt_params_features

def get_hw_params(
    x: np.ndarray,
    period: int = 7,
    extra_args: Optional[Dict[str, bool]] = None,
    default_status: bool = True,
) -> Dict[str, float]:
    """
    Estimates the smoothing parameters for HW linear trend.

    Args:
        x: The univariate time series array in the form of 1d numpy array.
        period: int; Seaonal period for fitting exponential smoothing model.
        extra_args: A dictionary containing information for disabling calculation
            of a certain feature. If None, no feature is disabled.
        default_status: Default status of the switch for calculate the
            features or not.

    Returns:
        Level, trend and seasonal parameter of a fitted Holt-Winter's model.
    """

    hw_params_features = {"hw_alpha": np.nan, "hw_beta": np.nan, "hw_gamma": np.nan}
    try:
        m = ExponentialSmoothing(
            x,
            initialization_method="estimated",
            seasonal="add",
            seasonal_periods=period,
            trend="add",
            use_boxcox=True,
        ).fit()
        if extra_args is not None:
            if extra_args.get("hw_alpha", default_status):
                hw_params_features["hw_alpha"] = m.params["smoothing_level"]
            if extra_args.get("hw_beta", default_status):
                hw_params_features["hw_beta"] = m.params["smoothing_trend"]
            if extra_args.get("hw_gamma", default_status):
                hw_params_features["hw_gamma"] = m.params["smoothing_seasonal"]
    except Exception as e:
        logging.warning(f"Holt-Winters failed {e}")
    return hw_params_features

@jit(nopython=True)
def negative_turning(signal):
    """Computes number of negative turning points of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which minimum number of negative turning points are counted
    Returns
    -------
    float
        Number of negative turning points
    """
    diff_sig = np.diff(signal)
    array_signal = np.arange(len(diff_sig[:-1]))
    negative_turning_pts = np.where((diff_sig[array_signal] < 0) & (diff_sig[array_signal + 1] > 0))[0]

    return len(negative_turning_pts)
@jit(nopython=True)
def positive_turning(signal):
    """Computes number of positive turning points of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which  positive turning points are counted

    Returns
    -------
    float
        Number of positive turning points
    """
    diff_sig = np.diff(signal)

    array_signal = np.arange(len(diff_sig[:-1]))

    positive_turning_pts = np.where((diff_sig[array_signal + 1] < 0) & (diff_sig[array_signal] > 0))[0]

    return len(positive_turning_pts)

@jit(nopython=True)
def travelled_distance(signal):
    """Computes signal traveled distance.

    Calculates the total distance traveled by the signal
    using the hypotenuse between 2 datapoints.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which distance is computed

    Returns
    -------
    float
        Signal distance
    """
    diff_sig = np.diff(signal).astype(float)
    return np.sum([np.sqrt(1 + diff_sig**2)])

@jit(nopython=True)
def auc(signal, fs=1):
    """Computes the area under the curve of the signal computed with trapezoid
    rule.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed
    fs : float
        Sampling Frequency
    Returns
    -------
    float
        The area under the curve value
    """
    t = _compute_time(signal, fs)

    return np.sum(0.5 * np.diff(t) * np.abs(np.array(signal[:-1]) + np.array(signal[1:])))

def lempel_ziv(signal, threshold=None):
    """Computes the Lempel-Ziv's (LZ) complexity index, normalized by the
    signal's length.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    amp_thres : float, optional
        Amplitude Threshold for the binarisation. If None, the mean of the signal is used.

    Returns
    -------
    lz_index : float
        Lempel-Ziv complexity index
    """
    if threshold is None:
        threshold = np.mean(signal)

    binary_signal = (signal > threshold).astype(int)
    string_binary_signal = "".join(map(str, binary_signal))
    lz_index = _calc_lempel_ziv_complexity(string_binary_signal)
    return lz_index

def median_abs_deviation(signal):
    """Computes median absolute deviation of the signal.

    Feature computational cost: 1
    Parameters
    ----------
    signal : nd-array
        Input from which median absolute deviation is computed

    Returns
    -------
    float
        Mean absolute deviation result
    """
    return median_abs_deviation(signal, scale=1)

def ecdf(signal, d=10):
    """Computes the values of ECDF (empirical cumulative distribution function)
    along the time axis.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    d: integer
        Number of ECDF values to return

    Returns
    -------
    float
        The values of the ECDF along the time axis
    """
    _, y = _calc_cumfun(signal)
    if len(signal) <= d:
        return tuple(y)
    else:
        return tuple(y[:d])

@jit(nopython=True)
def _compute_time(signal, fs):
    """Creates the signal correspondent time array.

    Parameters
    ----------
    signal: nd-array
        Input from which the time is computed.
    fs: int
        Sampling Frequency

    Returns
    -------
    time : float list
        Signal time
    """

    return np.arange(0, len(signal)) / fs

@jit(nopython=True)
def _calc_lempel_ziv_complexity(sequence):
    """Manual implementation of the Lempel-Ziv complexity.

    It is defined as the number of different substrings encountered as
    the stream is viewed from begining to the end.

    Reference:
    https://github.com/Naereen/Lempel-Ziv_Complexity/blob/master/src/lempel_ziv_complexity.py

    Parameters
    ----------
    sequence : string
        Binarised signal, as a string of characters

    Returns
    -------
        LZ index
    """

    sub_strings = set()

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind : ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1

    return len(sub_strings) / len(sequence)

def spectral_distance(signal, fs=1):
    """Computes the signal spectral distance.

    Distance of the signal's cumulative sum of the FFT elements to
    the respective linear regression.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral distance is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        spectral distance
    """
    f, fmag = _calc_fft(signal, fs)

    cum_fmag = np.cumsum(fmag)

    # Computing the linear regression
    points_y = np.linspace(0, cum_fmag[-1], len(cum_fmag))

    return np.sum(points_y - cum_fmag)

def fundamental_frequency(signal, fs=1):
    """Computes fundamental frequency of the signal.

    The fundamental frequency integer multiple best explain
    the content of the signal spectrum.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which fundamental frequency is computed
    fs : float
        Sampling frequency

    Returns
    -------
    f0: float
       Predominant frequency of the signal
    """
    signal = signal - np.mean(signal)
    f, fmag = _calc_fft(signal, fs)

    # Finding big peaks, not considering noise peaks with low amplitude
    bp = find_peaks(fmag, height=max(fmag) * 0.3)[0]

    # # Condition for offset removal, since the offset generates a peak at frequency zero
    bp = bp[bp != 0]
    if not list(bp):
        f0 = 0
    else:
        # f0 is the minimum big peak frequency
        f0 = f[min(bp)]
    return f0

def max_power_spectrum(signal, fs=1):
    """Computes maximum power spectrum density of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which maximum power spectrum is computed
    fs : float
        Sampling frequency

    Returns
    -------
    nd-array
        Max value of the power spectrum density
    """
    if np.std(signal) == 0:
        return float(max(welch(signal, fs, nperseg=len(signal))[1]))
    else:
        return float(max(welch(signal / np.std(signal), fs, nperseg=len(signal))[1]))


def max_frequency(signal, fs=1):
    """Computes maximum frequency of the signal.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which maximum frequency is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        0.95 of maximum frequency using cumsum
    """
    f, fmag = _calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag)

    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.95)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)

    return f[ind_mag]


def median_frequency(signal, fs=1):
    """Computes median frequency of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which median frequency is computed
    fs: int
        Sampling frequency

    Returns
    -------
    f_median : int
       0.50 of maximum frequency using cumsum.
    """
    f, fmag = _calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag)
    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.50)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)
    f_median = f[ind_mag]

    return f_median


def spectral_centroid(signal, fs=1):
    """Barycenter of the spectrum.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral centroid is computed
    fs: int
        Sampling frequency

    Returns
    -------
    float
        Centroid
    """
    f, fmag = _calc_fft(signal, fs)
    if not np.sum(fmag):
        return 0
    else:
        return np.dot(f, fmag / np.sum(fmag))


def spectral_decrease(signal, fs=1):
    """Represents the amount of decreasing of the spectra amplitude.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral decrease is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral decrease
    """
    f, fmag = _calc_fft(signal, fs)

    fmag_band = fmag[1:]
    len_fmag_band = np.arange(2, len(fmag) + 1)

    # Sum of numerator
    soma_num = np.sum((fmag_band - fmag[0]) / (len_fmag_band - 1), axis=0)

    if not np.sum(fmag_band):
        return 0
    else:
        # Sum of denominator
        soma_den = 1 / np.sum(fmag_band)

        # Spectral decrease computing
        return soma_den * soma_num


def spectral_kurtosis(signal, fs=1):
    """Measures the flatness of a distribution around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral kurtosis is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral Kurtosis
    """
    f, fmag = _calc_fft(signal, fs)
    if not spectral_spread(signal, fs):
        return 0
    else:
        spect_kurt = ((f - spectral_centroid(signal, fs)) ** 4) * (fmag / np.sum(fmag))
        return np.sum(spect_kurt) / (spectral_spread(signal, fs) ** 4)


def spectral_skewness(signal, fs=1):
    """Measures the asymmetry of a distribution around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral skewness is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral Skewness
    """
    f, fmag = _calc_fft(signal, fs)
    spect_centr = spectral_centroid(signal, fs)

    if not spectral_spread(signal, fs):
        return 0
    else:
        skew = ((f - spect_centr) ** 3) * (fmag / np.sum(fmag))
        return np.sum(skew) / (spectral_spread(signal, fs) ** 3)


def spectral_spread(signal, fs=1):
    """Measures the spread of the spectrum around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral spread is computed.
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral Spread
    """
    f, fmag = _calc_fft(signal, fs)
    spect_centroid = spectral_centroid(signal, fs)

    if not np.sum(fmag):
        return 0
    else:
        return np.dot(((f - spect_centroid) ** 2), (fmag / np.sum(fmag))) ** 0.5


def spectral_slope(signal, fs=1):
    """Computes the spectral slope.

    Spectral slope is computed by finding constants m and b of the function aFFT = mf + b, obtained by linear regression
    of the spectral amplitude.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral slope is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral Slope
    """
    f, fmag = _calc_fft(signal, fs)
    sum_fmag = fmag.sum()
    dot_ff = (f * f).sum()
    sum_f = f.sum()
    len_f = len(f)

    if not ([f]) or (sum_fmag == 0):
        return 0
    else:
        if not (len_f * dot_ff - sum_f**2):
            return 0
        else:
            num_ = (1 / sum_fmag) * (len_f * np.sum(f * fmag) - sum_f * sum_fmag)
            denom_ = len_f * dot_ff - sum_f**2
            return num_ / denom_


def spectral_variation(signal, fs=1):
    """Computes the amount of variation of the spectrum along time.

    Spectral variation is computed from the normalized cross-correlation between two consecutive amplitude spectra.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral variation is computed.
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral Variation
    """
    f, fmag = _calc_fft(signal, fs)

    sum1 = np.sum(np.array(fmag)[:-1] * np.array(fmag)[1:])
    sum2 = np.sum(np.array(fmag)[1:] ** 2)
    sum3 = np.sum(np.array(fmag)[:-1] ** 2)

    if not sum2 or not sum3:
        variation = 1
    else:
        variation = 1 - (sum1 / ((sum2**0.5) * (sum3**0.5)))

    return variation


def spectral_positive_turning(signal, fs=1):
    """Computes number of positive turning points of the fft magnitude signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the number of positive turning points of the fft magnitude are computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Number of positive turning points
    """
    f, fmag = _calc_fft(signal, fs)
    diff_sig = np.diff(fmag)

    array_signal = np.arange(len(diff_sig[:-1]))

    positive_turning_pts = np.where((diff_sig[array_signal + 1] < 0) & (diff_sig[array_signal] > 0))[0]

    return len(positive_turning_pts)

def spectral_roll_off(signal, fs=1):
    """Computes the spectral roll-off of the signal.

    The spectral roll-off corresponds to the frequency where 95% of the signal magnitude is contained
    below of this value.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral roll-off is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral roll-off
    """
    f, fmag = _calc_fft(signal, fs)
    cum_ff = np.cumsum(fmag)
    value = 0.95 * (np.sum(fmag))

    return f[np.where(cum_ff >= value)[0][0]]


def spectral_roll_on(signal, fs=1):
    """Computes the spectral roll-on of the signal.

    The spectral roll-on corresponds to the frequency where 5% of the signal magnitude is contained
    below of this value.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral roll-on is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral roll-on
    """
    f, fmag = _calc_fft(signal, fs)
    cum_ff = np.cumsum(fmag)
    value = 0.05 * (np.sum(fmag))

def spectral_entropy(signal, fs=1):
    """Computes the spectral entropy of the signal based on Fourier transform.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which spectral entropy is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        The normalized spectral entropy value
    """
    # Removing DC component
    sig = signal - np.mean(signal)

    f, fmag = _calc_fft(sig, fs)

    power = fmag**2

    if power.sum() == 0:
        return 0.0

    prob = np.divide(power, power.sum())

    prob = prob[prob != 0]

    # If probability all in one value, there is no entropy
    if prob.size == 1:
        return 0.0

    return -np.multiply(prob, np.log2(prob)).sum() / np.log2(prob.size)

def mfcc(signal, fs, pre_emphasis=0.97, nfft=512, nfilt=40, num_ceps=12, cep_lifter=22):
    """Computes the MEL cepstral coefficients.

    It provides the information about the power in each frequency band.

    Implementation details and description on:
    https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which MEL coefficients is computed
    fs : float
        Sampling frequency
    pre_emphasis : float
        Pre-emphasis coefficient for pre-emphasis filter application
    nfft : int
        Number of points of fft
    nfilt : int
        Number of filters
    num_ceps: int
        Number of cepstral coefficients
    cep_lifter: int
        Filter length

    Returns
    -------
    nd-array
        MEL cepstral coefficients
    """
    filter_banks = _filterbank(signal, fs, pre_emphasis, nfft, nfilt)

    mel_coeff = dct(filter_banks, type=2, axis=0, norm="ortho")[1 : (num_ceps + 1)]  # Keep 2-13

    mel_coeff -= np.mean(mel_coeff, axis=0) + 1e-8

    # liftering
    ncoeff = len(mel_coeff)
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)  # cep_lifter = 22 from python_speech_features library

    mel_coeff *= lift

    return {f'MFCC_{k}': v for k,v in enumerate(tuple(mel_coeff))}

def lpcc(signal, n_coeff=12):
    """Computes the linear prediction cepstral coefficients.

    Implementation details and description in:
    http://www.practicalcryptography.com/miscellaneous/machine-learning/tutorial-cepstrum-and-lpccs/

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction cepstral coefficients are computed
    n_coeff : int
        Number of coefficients

    Returns
    -------
    nd-array
        Linear prediction cepstral coefficients
    """
    # 12-20 cepstral coefficients are sufficient for speech recognition
    lpc_coeffs = _lpc(signal, n_coeff)

    if np.sum(lpc_coeffs) == 0:
        return tuple(np.zeros(len(lpc_coeffs)))

    # Power spectrum
    powerspectrum = np.abs(np.fft.fft(lpc_coeffs)) ** 2
    lpcc_coeff = np.fft.ifft(np.log(powerspectrum))

    return {f'LPCC_{k}': v for k, v in enumerate(tuple(np.abs(lpcc_coeff)))}



def cumuvals(signal, d=5):
    """Computes the values of cumulative values
    along the time axis.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    d: integer
        Number of ECDF values to return

    Returns
    -------
    float
        The cumulative values along the time axis
    """

    _, y = _calc_cumfun(signal)
    length = len(signal)
    if len(signal) > d:
        y = y[np.arange(0, length, length // d)]
        return {f'cumfunval_{k}': y[k] for k in range(0, d)}
    else:
        return {f'cumfunval_{k}': np.nan for k in range(0, d)}

def _filterbank(signal, fs=1, pre_emphasis=0.97, nfft=512, nfilt=40):
    """Computes the MEL-spaced filterbank.

    It provides the information about the power in each frequency band.

    Implementation details and description on:
    https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1

    Parameters
    ----------
    signal : nd-array
        Input from which filterbank is computed
    fs : float
        Sampling frequency
    pre_emphasis : float
        Pre-emphasis coefficient for pre-emphasis filter application
    nfft : int
        Number of points of fft
    nfilt : int
        Number of filters

    Returns
    -------
    nd-array
        MEL-spaced filterbank
    """

    # Signal is already a window from the original signal, so no frame is needed.
    # According to the references it is needed the application of a window function such as
    # hann window. However if the signal windows don't have overlap, we will lose information,
    # as the application of a hann window will overshadow the windows signal edges.

    # pre-emphasis filter to amplify the high frequencies

    emphasized_signal = np.append(
        np.array(signal)[0],
        np.array(signal[1:]) - pre_emphasis * np.array(signal[:-1]),
    )

    # Fourier transform and Power spectrum
    mag_frames = np.absolute(
        np.fft.rfft(emphasized_signal, nfft),
    )  # Magnitude of the FFT

    pow_frames = (1.0 / nfft) * (mag_frames**2)  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (fs / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(
        low_freq_mel,
        high_freq_mel,
        nfilt + 2,
    )  # Equally spaced in Mel scale
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    filter_bin = np.floor((nfft + 1) * hz_points / fs)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in np.arange(1, nfilt + 1):

        f_m_minus = int(filter_bin[m - 1])  # left
        f_m = int(filter_bin[m])  # center
        f_m_plus = int(filter_bin[m + 1])  # right

        for k in np.arange(f_m_minus, f_m):
            fbank[m - 1, k] = (k - filter_bin[m - 1]) / (filter_bin[m] - filter_bin[m - 1])
        for k in np.arange(f_m, f_m_plus):
            fbank[m - 1, k] = (filter_bin[m + 1] - k) / (filter_bin[m + 1] - filter_bin[m])

    # Area Normalization
    # If we don't normalize the noise will increase with frequency because of the filter width.
    enorm = 2.0 / (hz_points[2 : nfilt + 2] - hz_points[:nfilt])
    fbank *= enorm[:, np.newaxis]

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(
        filter_banks == 0,
        np.finfo(float).eps,
        filter_banks,
    )  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    return filter_banks


def _lpc(signal, n_coeff=12):
    """Computes the linear prediction coefficients.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction coefficients are computed
    n_coeff : int
        Number of coefficients

    Returns
    -------
    nd-array
        Linear prediction coefficients
    """

    if signal.ndim > 1:
        raise ValueError("Only 1 dimensional arrays are valid")
    if n_coeff > signal.size:
        raise ValueError("Input signal must have a length >= n_coeff")

    # Calculate the order based on the number of coefficients
    order = n_coeff - 1

    # Calculate LPC with Yule-Walker
    acf = np.correlate(signal, signal, "full")

    r = np.zeros(order + 1, "float32")
    # Assuring that works for all type of input lengths
    nx = np.min([order + 1, len(signal)])
    r[:nx] = acf[len(signal) - 1 : len(signal) + order]

    smatrix = _create_symmetric_matrix(r[:-1], order)

    if np.sum(smatrix) == 0:
        return tuple(np.zeros(order + 1))

    lpc_coeffs = np.dot(np.linalg.inv(smatrix), -r[1:])

    return tuple(np.concatenate(([1.0], lpc_coeffs)))

def _create_symmetric_matrix(acf, order=11):
    """Computes a symmetric matrix.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    acf : nd-array
        Input from which a symmetric matrix is computed
    order : int
        Order

    Returns
    -------
    nd-array
        Symmetric Matrix
    """

    smatrix = np.empty((order, order))
    xx = np.arange(order)
    j = np.tile(xx, order)
    i = np.repeat(xx, order)
    smatrix[i, j] = acf[np.abs(i - j)]

    return smatrix

@jit(nopython=True)
def _calc_cumfun(signal):
    """Computes the ECDF of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    Returns
    -------
    nd-array
      Sorted signal and computed ECDF.
    """
    return np.sort(signal), np.cumsum(signal)/(np.arange(1,signal.shape[0]+1))

def _calc_fft(signal, fs=1):
    """This functions computes the fft of a signal.

    Parameters
    ----------
    signal : nd-array
        The input signal from which fft is computed
    fs : float
        Sampling frequency

    Returns
    -------
    f: nd-array
        Frequency values (xx axis)
    fmag: nd-array
        Amplitude of the frequency values (yy axis)
    """

    fmag = np.abs(np.fft.rfft(signal))
    f = np.fft.rfftfreq(len(signal), d=1 / fs)
    return f.copy(), fmag.copy()


# similarity with pre-defined shapes
#@jit(forceobj=True)
def compute_dtw(x, y):
    return fastdtw(x, y)[0]

def shape_compare(x: np.ndarray) -> dict:
    lenX = x.shape[0]
    tDummy = np.arange(0, lenX, dtype='float64')

    try:
        lenX_inv = 1 / lenX
        tDummy_sq = tDummy ** 2
        lenX_sq_inv = lenX_inv ** 2

        xLinearUp = tDummy * lenX_inv
        xLinearDown = -xLinearUp
        xQup = tDummy_sq * lenX_sq_inv
        xQdown = -xQup
        xSinFU = np.sin(2 * np.pi * tDummy * lenX_inv)
        xSinFD = -xSinFU
        xCos = np.cos(2 * np.pi * tDummy * lenX_inv)

        half_lenX = lenX // 2
        half_lenX_array = half_lenX * np.ones((half_lenX + lenX % 2), dtype='float64')

        # first-up-then-down
        ts_fUtD = np.hstack([tDummy[:half_lenX], lenX - tDummy[half_lenX:]])

        # first-down-then-up
        ts_fDtU = np.hstack([-tDummy[:half_lenX], -lenX + tDummy[half_lenX:]])

        # first-up-then-straight
        ts_fUtS = np.hstack([tDummy[:half_lenX], half_lenX_array])

        # first-down-then-straight
        ts_fDtS = np.hstack([-tDummy[:half_lenX], -half_lenX_array])

        # first-straight-then-down
        ts_fStD = np.hstack([half_lenX_array, lenX - tDummy[half_lenX:]])

        # first-straight-then-up
        ts_fStU = np.hstack([half_lenX_array, tDummy[half_lenX:]])

        # List of (name, series) tuples
        series_list = [
            ('sim_linup', xLinearUp),
            ('sim_lindown', xLinearDown),
            ('sim_qup', xQup),
            ('sim_qdown', xQdown),
            ('sim_sinfu', xSinFU),
            ('sim_sinfd', xSinFD),
            ('sim_cos', xCos),
            ('sim_fUtD', ts_fUtD[:lenX]),
            ('sim_fDtU', ts_fDtU[:lenX]),
            ('sim_fUtS', ts_fUtS[:lenX]),
            ('sim_fDtS', ts_fDtS[:lenX]),
            ('sim_fStD', ts_fStD[:lenX]),
            ('sim_fStU', ts_fStU[:lenX])
        ]

        # Parallel DTW computation
        results = joblib.Parallel(n_jobs=-1)(joblib.delayed(compute_dtw)(x, series) for _, series in series_list)

        # Combine results into dictionary
        out_dict = {name: result for (name, _), result in zip(series_list, results)}

    except Exception as e:
        print(f"Shape Sim error: {e}")
        print(f"Shapes: "
              f"{ts_fUtD.mean()}, "
              f"{ts_fDtU.mean()}, "
              f"{ts_fUtS.mean()}, "
              f"{ts_fDtS.mean()},"
              f"{ts_fStD.mean()},"
              f"{ts_fStU.mean()}")
        print(f"X shape: {x.shape}")
        return {
            'sim_linup': np.nan,
            'sim_lindown': np.nan,
            'sim_qup': np.nan,
            'sim_qdown': np.nan,
            'sim_sinfu': np.nan,
            'sim_sinfd': np.nan,
            'sim_cos': np.nan,
            'sim_fUtD': np.nan,
            'sim_fDtU': np.nan,
            'sim_fUtS': np.nan,
            'sim_fDtS': np.nan,
            'sim_fStD': np.nan,
            'sim_fStU': np.nan,
        }

    return out_dict



