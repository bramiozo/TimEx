####################
## Pre-processing ##
####################

from scipy import signal
from scipy import interpolate
from scipy import ndimage
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import cwt, ricker

import numpy as np
import tsfresh

from tqdm import tqdm
import sys, os
import pandas as pd 
import gc
from collections import defaultdict
from time import sleep
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from sklego.meta.grouped_transformer import GroupedTransformer

from typing import Optional, List, Dict, Literal

from tslearn import metrics

def normalise_ts(ts_df, id_col='ID', 
                 time_col='Time_days', 
                 val_col='eGFR_CKDEpi2012',
                 df_out=False,
                 scaler: Literal['standard', 'minmax'] = 'standard'):
    if scaler=='standard':
        scaler = TimeSeriesScalerMeanVariance()
    elif scaler=='minmax':
        scaler = TimeSeriesScalerMinMax()
    
    #group_scaler = GroupedTransformer(scaler, groups=id_col)
    #ts_df[val_col] = group_scaler.fit_transform(ts_df)    
    
    res = {id_col: [], time_col: [], val_col: []}
    for _id in tqdm(ts_df[id_col].unique()):
        _df = ts_df.loc[ts_df[id_col]==_id][[id_col, time_col, val_col]]
        ts = _df[val_col].values
        ts = ts.reshape(1, -1, 1)
        ts = scaler.fit_transform(ts)
        ts = ts.reshape(-1)
        _df[val_col] = ts
        
        res[id_col].extend(_df[id_col].values)
        res[time_col].extend(_df[time_col].values)
        res[val_col].extend(_df[val_col].values)
    if df_out:
        return pd.DataFrame(res)
    else:
        return res

def get_filtered_df(ts_df, 
                    id_col='ID',
                    time_col='Time_days', 
                    min_days=365, 
                    min_measurements=3):
    maxts = ts_df.groupby(id_col)[time_col].max()
    ltids = maxts[maxts>min_days].index
    
    cnts = ts_df[(ts_df[id_col].isin(ltids)) &
                 (ts_df[time_col]<min_days)].groupby(id_col).size()
    filtered_ids = cnts[cnts>min_measurements].index
    
    ts = ts_df.loc[ts_df[id_col].isin(filtered_ids)]
    return ts

def get_interpolated(ts_df, id_col='ID', time_col='Time_days',
                     val_col='eGFR_CKDEpi2012',
                     days_before=0,
                     max_days=365,
                     time_res=7,
                     keep_t0_value=False,
                     df_out=False):
    '''
    Extract interpolated time series data.

    :param ts_df: pd.DataFrame, time series data
    :param id_col: str, default 'ID'
    :param time_col: str, time column
    :param val_col: str, default 'eGFR_CKDEpi2012'
    :param days_before: int, how many days before t0 to include
    :param max_days: int, maximum number of days to include
    :param time_res: int, time resolution of time series
    :param keep_t0_value: boolean, whether to keep t0 value
    :param df_out: boolean, whether to output dataframe
    '''
    trange = np.arange(-days_before, max_days, time_res)
    if (keep_t0_value) & (days_before>0):
        trange = np.insert(trange, 1, 0)

    res = {id_col: [], time_col: [], val_col: []}
    for _id in tqdm(ts_df[id_col].unique()):
        _df = ts_df.loc[ts_df[id_col]==_id][[time_col, val_col]]
        x = _df[time_col]
        y = _df[val_col]
        interpolator = interpolate.PchipInterpolator(x, y, extrapolate=False)
        interpolated = interpolator(trange)
        
        ids = len(trange)*[_id]

        res[id_col].extend(ids)
        res[time_col].extend(trange)
        res[val_col].extend(interpolated)
    
    if df_out:
        return pd.DataFrame.from_dict(res, orient='columns')
    else:
        return res

# TODO: ensure that first N-points are excluded from smoothing
def get_smoothed_rolling_mean(ts_dict: dict, 
                              id_col='ID', 
                              time_col='Time_days',
                              val_col='eGFR_CKDEpi2012', 
                              window=5,
                              Nskip=3,
                              df_out=False):
    res = {id_col: [], time_col: [], val_col: []}
    
    ts_df = pd.DataFrame.from_dict(ts_dict)
    for _id in tqdm(ts_df[id_col].unique()):
        _df = ts_df.loc[ts_df[id_col]==_id][[id_col, time_col, val_col]]
        _df = _df.sort_values(by=time_col)
        if Nskip > 0:
            r = range(Nskip, _df.shape[0]-Nskip)
        else:
            r = range(_df.shape[0])
        _df.iloc[r, 2] = _df[val_col].rolling(window=window,
                                            min_periods=1).mean().values[r]
        
        res[id_col].extend(_df[id_col].values)
        res[time_col].extend(_df[time_col].values)
        res[val_col].extend(_df[val_col].values)
    if df_out:
        return pd.DataFrame.from_dict(res, orient='columns')
    else:
        return res

# TODO: ensure that first N-points are excluded from smoothing
def get_smoothed_gaussian_kernel(ts_dict: dict, 
                                 id_col='ID', 
                                 time_col='Time_days',
                                 val_col='eGFR_CKDEpi2012', 
                                 window=5,
                                 Nskip=3,
                                 df_out=False):
    res = {
            id_col: [], 
            time_col: [], 
            val_col: []
        }
    
    ts_df = pd.DataFrame.from_dict(ts_dict)
    for _id in tqdm(ts_df[id_col].unique()):
        _df = ts_df.loc[ts_df[id_col]==_id][[id_col, time_col, val_col]]
        _df = _df.sort_values(by=time_col)
        smoothed_values = ndimage.gaussian_filter1d(_df[val_col], window)

        if Nskip > 0:
            r = range(Nskip, len(_df)-Nskip)
        else:
            r = range(len(_df))

        _df.iloc[r, 2] = smoothed_values[r]
        
        res[id_col].extend(_df[id_col].values)
        res[time_col].extend(_df[time_col].values)
        res[val_col].extend(_df[val_col].values)
    if df_out:
        return pd.DataFrame.from_dict(res, orient='columns')
    else:
        return res


# perform low pass filtering
def get_low_pass_filtered(ts_dict: dict, id_col='ID', time_col='Time_days',
                        val_col='eGFR_CKDEpi2012', cutoff=0.25, stationary=True,
                        order=1, btype='low', window=3, df_out=False):
    res = {id_col: [], time_col: [], val_col: []}
    '''
    Should normaly only be applied for uniformly sampled time series that 
    are stationary and have a constant sampling rate.
    If stationary is set to False
    '''
    
    if stationary:
        for _id in tqdm(ts_dict[id_col].unique()):
            _df = ts_dict.loc[ts_dict[id_col]==_id][[id_col, time_col, val_col]]
            _df = _df.sort_values(by=time_col)
            vals = _df[val_col].values
            filtered = signal.butter(order, cutoff, 
                                         btype='low', 
                                         analog=False, 
                                         output='sos')
            filtered = signal.sosfilt(filtered, vals)
            
            res[id_col].extend(_df[id_col].values)
            res[time_col].extend(_df[time_col].values)
            res[val_col].extend(filtered)
    else:       
        for _id in tqdm(ts_dict[id_col].unique()):
            _df = ts_dict.loc[ts_dict[id_col]==_id][[id_col, time_col, val_col]]
            _df = _df.sort_values(by=time_col)
            
            vals = _df[val_col].values
            trend = ndimage.gaussian_filter1d(vals, window)   
            trend[-window:] = vals[-window:]
            
            detrended = vals - trend        
            passfilter = signal.butter(order, cutoff, 
                                         btype='low', 
                                         analog=False, 
                                         output='sos')
            filtered = signal.sosfilt(passfilter, detrended)             
            retrended = filtered + trend
            
            res[id_col].extend(_df[id_col].values)
            res[time_col].extend(_df[time_col].values)
            res[val_col].extend(retrended)       
    
    if df_out:
        return pd.DataFrame.from_dict(res, orient='columns')
    else:
        return res
    