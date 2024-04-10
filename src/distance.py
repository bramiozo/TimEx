import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import cwt, ricker

from tqdm import tqdm
from scipy import signal
import numpy as np

from tqdm import tqdm
import sys, os
import pandas as pd 

import gc

from collections import defaultdict
from time import sleep

from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.svm import SVC, SVR
from collections import defaultdict
import networkx as nx
from pytwed import twed
import dcor

#from tslearn.clustering import GlobalAlignmentKernelKMeans, TimeSeriesKMeans, KernelKMeans
from tslearn.metrics import gak, dtw, lcss, dtw_path,  soft_dtw, ctw
from tslearn.metrics import cdist_soft_dtw, cdist_soft_dtw_normalized
from tslearn.metrics import cdist_dtw, cdist_gak, cdist_lcss, cdist_ctw
from tslearn.generators import random_walks
from tslearn.utils import to_time_series_dataset

from dtaidistance import dtw as dtw_fast, ed
from dtaidistance.clustering import HierarchicalTree, Hierarchical, LinkageTree
from dtaidistance import dtw_ndim

from sktime import distances as skdist

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

import collections.abc

def granger_causality(x,y, maxlag=5):
    # Computes the Granger causality measure between two numeric time series.
    # x is the cause, y is the effect
    #
    # Input: x
    #        y
    # Output: Granger causality measure
    #
    return grangercausalitytests(np.vstack([x,y]).T,
                                 maxlag=maxlag, 
                                 verbose=False)[maxlag][0]['ssr_ftest'][0]
# cross-correlation distance
@jit(nopython=True)
def cross_correlation(x, y, lag):
    y_shifted = np.roll(y, lag)
    if lag >= 0:        
        y_shifted[:lag] = 0
    else:
        y_shifted[lag:] = 0
    return np.correlate(x, y_shifted)[0]**2

@jit(nopython=True)
def cross_corr_dist(x, y):
    # Computes the distance measure based on the cross-correlation between a pair of numeric time series.
    diffLen = min(len(x),len(y))-1
    numerator = 1 - cross_correlation(x, y, 0)
    denominator = sum(1-cross_correlation(x,y,1))
    quot = numerator/denominator
    return np.sqrt(quot)

def dist_corr_direct(s1, s2):
    # s1 and s2 are arrays of shape (T1,D) or (T2,D)
    # returns the distance correlation between s1 and s2
    # https://en.wikipedia.org/wiki/Distance_correlation
    #
    # take the minimum of T1 and T2
    T = min(s1.shape[0], s2.shape[0])
    # calculate the distance matrices
    return dcor.distance_correlation_sqr(s1[:T], s2[:T])

@jit(nopython=True)
def cust_dist(x, y):
    alpha=2
    eps=1e-4 
    # extend shortest series with linear extrapolation of first and last value
    if len(x)<len(y):
        #extra_steps_arr = np.arange(len(x), len(y), 1)
        x = np.hstack((x, np.ones(len(y)-len(x))*x[-1])) #extra_steps_arr * (x[-1] - x[0]) + x[0]))
    elif len(y)<len(x):
        #extra_steps_arr = np.arange(len(y), len(x), 1)
        y = np.hstack((y, np.ones(len(x)-len(y))*y[-1])) #extra_steps_arr * (y[-1] - y[0]) + y[0]))
    # calculate distance
    slopeX = np.hstack((np.array([1]), x[1:] -x[:-1]))
    xang = (x/(x.argsort()+1)) # np.atan, x/(x.argsort()+1)
    slopeY = np.hstack((np.array([1]), y[1:] -y[:-1]))
    yang = (y/(y.argsort()+1)) # np.atan, y/(y.argsort()+1)
    xsign = np.sign(xang)
    ysign = np.sign(yang)
    signDiff = alpha*np.abs(xsign - ysign)+1
    d1 = np.linalg.norm((xang - yang)*signDiff)/(np.linalg.norm(xang)+np.linalg.norm(yang)+eps)
    d2 = np.linalg.norm(slopeX-slopeY)/(np.linalg.norm(slopeX)+np.linalg.norm(slopeY)+eps)
    d3 = np.linalg.norm(x-y)/(np.linalg.norm(x)+np.linalg.norm(y)+eps)
    return d1+2*d2+d3

@jit(nopython=True)
def cust_twed(s1, s2, nu, lambda_):
    n, m = len(s1), len(s2)
    dp = np.full((n + 1, m + 1), np.inf)

    # Initialization
    dp[0, 0] = 0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + (abs(s1[i - 1]) ** 2) + lambda_
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + (abs(0 - s2[j - 1]) ** 2) + lambda_

    # Main loop
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Deletion
            d1 = dp[i - 1, j] + (abs(s1[i - 1]) ** 2) + lambda_
            # Insertion
            d2 = dp[i, j - 1] + (abs(0 - s2[j - 1]) ** 2) + lambda_
            # Match
            d3 = dp[i - 1, j - 1] + (abs(s1[i - 1] - s2[j - 1]) ** 2) + (nu * abs((i - 1) - (j - 1)))
            dp[i, j] = min(d1, d2, d3)

    return dp[n, m]


# shape-based distance
@jit(nopython=True)
def shape_based_distance(x,y):
    return True

# SVM distance


# DTW
def dtw_distance(x, y):
    return dtw_fast.distance(x, y)

# soft DTW
def soft_dtw_distance(x, y, normalized=False):
    if normalized:
        return cdist_soft_dtw_normalized(x, y,
                                         gamma=1.0)
    else:
        return cdist_soft_dtw(x, y, 
                              gamma=1.0)
    

# Canonical TW
def canonical_tw_distance(x, y):
    return cdist_ctw(x, y, n_components=50,
                     sakoe_chiba_radius=None,
                     itakura_max_slope=None)

# Global Alignment Kernel
def gak_distance(x, y):
    return cdist_gak(x, y, sigma=1.0)

# WDTW
def wdtw_distance(x, y):
    return sktime.wdtw_distance.distance(x, y, window=None, 
                                         itakura_max_slope=None)

# DDTW
def ddtw_distance(x, y):
    return sktime.ddtw_distance(x, y, window=None, 
                                itakura_max_slope=None)

# WDDTW
def ddtw_distance(x, y):
    return sktime.wddtw_distance(x, y, window=None, 
                                 itakura_max_slope=None)

# TWED
# https://github.com/pfmarteau/TWED
# https://github.com/jzumer/pytwed

def twed_distance(x, y):
    return sktime.twed_distance(x, y, window=None, 
                                itakura_max_slope=None)
    
def msm_distance(x, y):
    return sktime.msm_distance(x, y, window=None, 
                                itakura_max_slope=None,
                                c=1.0)

def erp_distance(x, y):
    return sktime.erp_distance(x, y, window=None, 
                                itakura_max_slope=None)

def edr_distance(x, y):
    return sktime.edr_distance(x, y, window=None, 
                                itakura_max_slope=None,
                                epsilon=None)

# longest common subsequence
def lcs_distance(x, y):
    return skdist.lcss_distance(x, y)

# cross-distance correlation

# cross-multiscale graph correlation

# compression-based dissimilarity

# TICC

# MASS
    
# Kendall-Tau

# Spearmans-Rho

# Pearson


    
