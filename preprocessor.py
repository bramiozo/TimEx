#!/usr/bin/env python
# coding: utf-8

# In[6]:


import wfdb
#import tslearn
import pandas as pd
import numpy as np
import scipy as sc
import urllib.request as ur
import csv
from pathlib import Path
import urllib.parse
from scipy import signal
import neurokit2 as nk

import os
import sys
import re

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, butter, filtfilt, detrend, savgol_filter, sosfilt


class ECGSignalProcessor:
    def __init__(self, filepath, fs=500):
        self.filepath = filepath
        self.fs = fs
        self.record = wfdb.rdrecord(filepath)
        self.n_sig = self.record.n_sig
        self.sig_len = self.record.sig_len
        self.sig_name = self.record.sig_name
        self.p_signal = np.transpose(self.record.p_signal)
        self.x = np.linspace(0, 10, self.sig_len)
        self.filtered_signal = None
        self.smoothed = None
        self.detrended = None
        self.trimmed_signal = None
        self.trimmed_time = None
        self.peaks = None
        self.dips = None

    def plot_signals(self):
        fig, axes = plt.subplots(nrows=self.n_sig, ncols=1, figsize=(18, 35))
        for i in range(self.n_sig):
            axes[i].plot(self.x, self.p_signal[i])
            axes[i].set_title(self.sig_name[i])
        plt.show()

    def detect_peaks_and_dips(self):
        self.peaks, _ = find_peaks(self.p_signal[0], distance=30, prominence=(0.1))
        self.dips, _ = find_peaks(-self.p_signal[0], distance=250)
        
    def apply_bandpass_filter(self, lowcut=0.5, highcut=40.0, order=4):
        nyq = 0.5 * self.fs
        sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
        self.filtered_signal = sosfilt(sos, self.p_signal[0])
       

    def apply_notch_filter(self, freq=60.0, bandwidth=1.0, order=4):
        nyq = 0.5 * self.fs
        b, a = butter(order, [(freq - bandwidth) / nyq, (freq + bandwidth) / nyq], btype='bandstop')
        self.filtered_signal = filtfilt(b, a, self.filtered_signal)
       

    def apply_savgol_filter(self, window_length=31, polyorder=3):
        self.smoothed = savgol_filter(self.filtered_signal, window_length, polyorder)
        t = np.linspace(0, 10, len(self.smoothed))
        

    def apply_detrend(self):
        self.detrended = detrend(self.smoothed, type='linear')
        t = np.linspace(0, 10, len(self.detrended))
        

    def trim_signal(self):
        peaks, _ = find_peaks(self.detrended, distance=30, prominence=(0.1))
        dips, _ = find_peaks(-self.detrended, distance=250)
        self.trimmed_signal = self.detrended[peaks[0]:dips[-1]]
        self.trimmed_time = np.linspace(self.x[peaks[0]], self.x[dips[-1]], len(self.trimmed_signal))
        plt.figure(figsize=(15, 5))
        plt.plot(self.x[peaks], self.detrended[peaks], 'o', color='red')
        plt.plot(self.x[dips], self.detrended[dips], 'o', color='green')
        plt.plot(self.x, self.detrended)
        plt.plot(self.trimmed_time, self.trimmed_signal)
        plt.title('Trimmed Signal')
        plt.show()
        
        
    def quality_check(self):
        quality = nk.ecg_quality(self.trimmed_signal, method= 'zhao2018', approach= 'fuzzy', sampling_rate=self.fs)
        print(quality)

processor = ECGSignalProcessor('Downloads\\data\\mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1\\files\\p10000032\\s102511170\\102511170')
processor.plot_signals()
processor.detect_peaks_and_dips()
processor.apply_bandpass_filter()
processor.apply_notch_filter()
processor.apply_savgol_filter()
processor.apply_detrend()
processor.trim_signal()
processor.quality_check()


# In[ ]:





# In[ ]:




