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
from tqdm import tqdm
import sys, os
import pandas as pd 

import gc

from collections import defaultdict

from time import sleep

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
                'wavelet_transform_feature': wavelet_transform_feature
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
    

from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

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
        for _id in tqdm(df[id_col].unique()):
            ts_data = df[df[id_col] == _id][val_col].values
            extracted_features = pycatch22.catch22_all(ts_data)            
            self.features[_id] = dict(zip(extracted_features['names'], extracted_features['values']))
    
    def transform(self):
        # Convert the features dictionary to a DataFrame
        features_df = pd.DataFrame.from_dict(self.features, orient='index').reset_index()
        features_df.rename(columns={'index': 'id'}, inplace=True)
        return features_df