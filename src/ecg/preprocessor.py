import wfdb
import pandas as pd
import numpy as np
import torch
import scipy as sc
import urllib.request as ur
import csv
from pathlib import Path
import urllib.parse
from scipy import signal
import neurokit2 as nk
from wfdb.processing import resample_sig

import os
import sys
import re

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, butter, filtfilt, detrend, savgol_filter, sosfilt

"""
Standard filter:
Resample to 500 Hz
Bandpass filter between 0.5 and 40 Hz
50/60 Hz notch filter to remove powerline interference
trim to 10 seconds -> truncate
pad with zeros to 10 seconds if needed
smooth with Savitzky-Golay filter
detrend the signal
replace NaNs with zeros
Peak limiter: remove peaks above 3mV and below -3mV
Wavelet denoising (db6)
"""

# in wfdb.processing there is function xqrs_detect that detects QRS complexes, this can be seen as sentences 
# if we consider the ECG signal as a text and the leads are paragraphs. I.e. we can use this idea to develop an hierarchical model

class ECGSignalProcessor:
    def __init__(self, data, num_channels, channel_length, channel_names, x, fs=500):

        self.fs = fs
        self.n_sig = num_channels
        self.sig_len = channel_length
        self.sig_name = channel_names
        self.p_signal = data
        self.x = x
        self.filtered_signal = None
        self.filtered_signal2 = None
        self.smoothed = None
        self.detrend = None
        self.trim_arr = None
        self.peaks = None
        self.dips = None

    # add augmentation
    #

    def plot_signals(self):
        fig, axes = plt.subplots(nrows=self.n_sig, ncols=1, figsize=(18, 35))
        for i in range(self.n_sig):
            axes[i].plot(self.x, self.p_signal[i])
            axes[i].set_title(self.sig_name[i])
        plt.show()

    def detect_peaks_and_dips(self):

        for i in range(self.n_sig):
          self.peaks, _ = find_peaks(self.p_signal[i], distance=30, prominence=(0.1))
          self.dips, _ = find_peaks(-self.p_signal[i], distance=250)

    def apply_bandpass_filter(self, lowcut=0.5, highcut=40.0, order=4):
        nyq = 0.5 * self.fs
        sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
        self.filtered_signal= np.array([sosfilt(sos, sig) for sig in self.p_signal], dtype='float32')
        return self.filtered_signal


    def apply_notch_filter(self, filtered, freq=60.0, bandwidth=1.0, order=4):
        nyq = 0.5 * self.fs
        b, a = butter(order, [(freq - bandwidth) / nyq, (freq + bandwidth) / nyq], btype='bandstop')
        self.filtered_signal2= np.array([filtfilt(b, a, sig) for sig in filtered], dtype='float32')
        return self.filtered_signal2


    def apply_savgol_filter(self, filtered, window_length=31, polyorder=3):
        for i in range(self.n_sig):
          self.smoothed= np.array([savgol_filter(sig, window_length, polyorder) for sig in filtered], dtype='float32')
          t = np.linspace(0, 10, len(self.smoothed))
          return self.smoothed


    def apply_detrend(self, filtered):
        for i in range(self.n_sig):
          self.detrend=np.array([detrend(sig, type='linear') for sig in filtered], dtype='float32')
          t = np.linspace(0, 10, len(self.detrend))

    def trim_signal(self, filtered):
        trim_signals = []
        trim_times = []
        for i in range(self.n_sig):
          peaks, _ = find_peaks(filtered[i], distance=30, prominence=(0.1))
          dips, _ =  find_peaks(-filtered[i], distance=250)
          start_point = min(peaks[0], dips[0])
          end_point = max(peaks[-1], dips[-1])
          trimmed = filtered[i][start_point:end_point]
          time_trimmed = np.linspace(self.x[start_point], self.x[end_point], len(trimmed))
          trim_signals.append(trimmed)
          trim_times.append(time_trimmed)

          self.trim_arr=np.array(trim_signals, dtype='object')
          self.trim_time_arr= np.array(trim_times, dtype='object')
        return self.trim_arr

    # TODO: EXTERNAL, needs refactoring
    def augment_signal(self, signal):
        """Apply enhanced augmentations to the signal"""
        if not self.is_train or not self.config.USE_AUGMENTATION:
            return signal

        # Time shift augmentation (increased probability and range)
        if 'shift' in self.augmentations:
            if np.random.random() < self.config.TIME_SHIFT_PROB:
                shift_factor = np.random.uniform(-self.config.TIME_SHIFT_MAX, self.config.TIME_SHIFT_MAX)
                shift_amount = int(shift_factor * signal.shape[1])
                if shift_amount > 0:
                    signal = torch.cat([signal[:, shift_amount:], signal[:, :shift_amount]], dim=1)
                elif shift_amount < 0:
                    shift_amount = abs(shift_amount)
                    signal = torch.cat([signal[:, -shift_amount:], signal[:, :-shift_amount]], dim=1)

        # TODO: scale should only be applied to the peaks/valleys
        if 'scale' in self.augmentations:
            # Amplitude scaling augmentation
            if np.random.random() < self.config.AMPLITUDE_SCALE_PROB:
                scale_factor = np.random.uniform(*self.config.AMPLITUDE_SCALE_RANGE)
                signal = signal * scale_factor

        if 'gauss' in self.augmentations:
            # Enhanced Gaussian noise
            if np.random.random() < self.config.GAUSSIAN_NOISE_PROB:
                noise = torch.randn_like(signal) * self.config.GAUSSIAN_NOISE_SCALE
                signal = signal + noise

        if 'dropout' in self.augmentations:
            # Lead dropout (randomly zero out leads to improve robustness)
            if np.random.random() < 0.8:  # 10% chance
                lead_idx = np.random.randint(0, int(0.05*signal.shape[0]))
                signal[lead_idx] = signal[lead_idx] * 0.0

        return signal

    # resample

    def quality_check(self, trimmed):
        #TODO: nk is FF-ing slow, find alternative..
        for sig in trimmed:
          quality = nk.ecg_quality(sig, method= 'zhao2018', approach= 'fuzzy', sampling_rate=self.fs)
          print(quality)
