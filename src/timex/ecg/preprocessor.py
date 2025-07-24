import wfdb
import pandas as pd
import numpy as np
from numpy import ndarray
import torch
import scipy as sc
import urllib.request as ur
import csv
from pathlib import Path
import urllib.parse
from scipy import signal
import neurokit2 as nk
from wfdb.processing import resample_sig
from neurokit2.signal import signal_resample

from meegkit.detrend import detrend as meeg_detrend, regress
from statsmodels.tsa import tsatools, seasonal
import warnings

import os
import sys
import re

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, butter, filtfilt, detrend, savgol_filter, sosfilt

from typing import Literal
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
    def __init__(self, data, num_channels, fs=500):

        self.fs = fs
        self.n_sig = num_channels
        self.p_signal = np.array(data)
        self.smoothed = None
        self.detrend = None
        self.trim_arr = None
        self.peaks = None
        self.dips = None

        if self.p_signal.shape[0] != num_channels:
            warnings.warn(f"Number of channels {self.p_signal.shape[0]} is not as expected: {num_channels}")

    def detect_peaks_and_dips(self):

        for i in range(self.p_signal.shape[0]):
          self.peaks, _ = find_peaks(self.p_signal[i], distance=30, prominence=(0.1))
          self.dips, _ = find_peaks(-self.p_signal[i], distance=250)

    def apply_bandpass_filter(self, lowcut=0.5, highcut=40.0, order=4):
        nyq = 0.5 * self.fs
        sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')

        # Convert if needed
        if isinstance(self.p_signal, torch.Tensor):
            signal_np = self.p_signal.numpy()
        else:
            signal_np = self.p_signal

        self.p_signal= np.array([sosfilt(sos, sig) for sig in self.p_signal], dtype='float32')


    def apply_notch_filter(self, freq=60.0, bandwidth=1.0, order=4):
        nyq = 0.5 * self.fs
        b, a = butter(order, [(freq - bandwidth) / nyq, (freq + bandwidth) / nyq], btype='bandstop')

        # Convert if needed
        if isinstance(self.p_signal, torch.Tensor):
            signal_np = self.p_signal.numpy()
        else:
            signal_np = self.p_signal

        self.p_signal = np.array([filtfilt(b, a, sig) for sig in self.p_signal], dtype='float32')


    def apply_savgol_filter(self, window_length=31, polyorder=3):
        """
        Apply Savitzky-Golay filter for smoothing the ECG signal.

        Args:
            signal: np.array or torch.Tensor with shape [leads, samples]
            window_length: Size of the filter window (must be odd and >= polyorder + 2)
            polyorder: Polynomial order for fitting (must be less than window_length)

        Returns:
            Smoothed signal of the same shape.
        """
        # Validate window size
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length")

        # Convert if needed
        if isinstance(self.p_signal, torch.Tensor):
            signal_np = self.p_signal.numpy()
        else:
            signal_np = self.p_signal

        # Apply Savitzky-Golay filter across each lead
        self.p_signal = savgol_filter(signal_np, window_length=window_length, polyorder=polyorder, axis=1)


    def apply_detrend(self, 
                        method: Literal['linear', 'poly', 'poly_np', 'MSTL', 'meeg', 'sliding_mean', 'sliding_median'] ='linear',
                        order: int=3,
                        period: int=100,
                        windows: int=101,
                        cutsize: int=250,
                        median_window: int=30,
                        iterate: int=5
                        ):
        if isinstance(self.p_signal, torch.Tensor):
            signal_np = self.p_signal.numpy()
        else:
            signal_np = self.p_signal

        if method == 'linear':
            self.p_signal = detrend(signal_np, axis=1, type='linear')
        elif method == 'poly':
            for channel in range(self.p_signal.shape[0]):
                self.p_signal[channel, :] = tsatools.detrend(signal_np[channel, :], order=order)
        elif method == 'poly_np':
            for channel in range(self.p_signal.shape[0]):
                z = np.polyfit(np.arange(0, signal_np.shape[1], 1), signal_np[channel, :], order)
                y_poly = np.poly1d(z)
                self.p_signal[channel, :] = signal_n[channel, :] - y_poly(np.arange(0, signal_np.shape[1], 1))
        elif method == 'MSTL':
            for channel in range(self.p_signal.shape[0]):
                detrender= seasonal.MSTL(signal_np[channel, :], periods=period, windows=windows, iterate=iterate).fit()
                self.p_signal[channel, :] = detrender.seasonal
        elif method == 'meeg':
            '''
             Use sinusoids as basis functions:
              This will suffer from the Gibbs-phenomenon. 
              To mitigate this, we have to cut the head/tail
            '''
            new_signal = np.zeros((self.p_signal.shape[0], self.p_signal.shape[1] - cutsize*2), np.float32)
            for channel in range(self.p_signal.shape[0]):
                new_signal[channel, :] = meeg_detrend(signal_np[channel, :], order=order, basis="sinusoids")[0][cutsize:-cutsize]
            self.p_signal = new_signal
        elif method == 'sliding_median':
            '''
             Make base plot using sliding mean, subtract from raw 
            '''
            new_signal = np.zeros((self.p_signal.shape[0], self.p_signal.shape[1] - median_window+1), np.float32)
            for channel in range(self.p_signal.shape[0]): 
                sl= np.median(np.lib.stride_tricks.sliding_window_view(signal_np[channel, :], (median_window,)), axis=1)
                new_signal[channel, :] = signal_np[channel, median_window-1:] - sl
            self.p_signal = new_signal
        elif method == 'sliding_mean':
            '''
             Make base plot using sliding mean, subtract from raw 
            '''
            new_signal = np.zeros((self.p_signal.shape[0], self.p_signal.shape[1] - median_window+1), np.float32)
            for channel in range(self.p_signal.shape[0]): 
                sl= np.mean(np.lib.stride_tricks.sliding_window_view(signal_np[channel, :], (median_window,)), axis=1)
                new_signal[channel, :] = signal_np[channel, median_window-1:] - sl
            self.p_signal = new_signal
        else:
            raise ValueError(f"For now we only accept ['linear',' poly', 'poly_np', 'MSTL', 'meeg', 'sliding_median', 'sliding_mean']")

    def  standardize_sampling_rate(self,
                                   backend: Literal['neurokit2', 'wfdb']='neurokit2', 
                                   method: Literal['interpolation', 'pandas', 'numpy', 'poly', 'fft']='interpolation',
                                   fs_target: int=500):
        '''
        Apply the wfdb resampler to the ECG signal

        Args: 
            signal: np.array [channels, samples]
        '''
        assert(backend in ['neurokit2', 'wfdb']), f"backend should be one of {['neurokit2', 'wfdb']}"
        assert(method in ['interpolation', 'numpy', 'poly', 'fft']), f"method should be on of {['interpolation', 'numpy', 'poly', 'fft']}"

        if isinstance(self.p_signal, torch.Tensor):
            signal_np = self.p_signal.numpy()
        else:
            signal_np = self.p_signal


        if backend == 'wfdb':
            s = np.transpose(signal_np)
            end_index = s.shape[0] - s.shape[0] % 2 # needs even indices?
            s = s[:end_index,: ]

            try:
                signal_resampled, _ = resample_sig(s, 
                                                self.fs, 
                                                fs_target)
            except Exception as e:
                print(e)
                raise ValueError(f" resampling failed for {s.shape} with fs {self.fs} and fs_target {fs_target}")
            self.p_signal = np.transpose(signal_resampled)
        else:
            try:
                self.p_signal = signal_resample(signal_np, 
                                        sampling_rate=self.fs, 
                                        desired_sampling_rate=fs_target,
                                        method=method)
            except Exception as e:
                print(e)
                raise ValueError(f" resampling failed for {self.p_signal.shape} with fs {self.fs} and fs_target {fs_target}")

    def trim_signal(self, peak_distance=30, dip_distance=250, prominence=0.1):
        trim_signals = []
        trim_times = []

        # Convert if needed
        if isinstance(self.p_signal, torch.Tensor):
            signal_np = self.p_signal.numpy()
        else:
            signal_np = self.p_signal

        new_signal = np.zeros((self.p_signal.shape[0], self.p_signal.shape[1]))
        for i in range(self.p_signal.shape[0]):
          peaks, _ = find_peaks(signal_np[i], distance=peak_distance, prominence=(prominence))
          dips, _ =  find_peaks(-signal_np[i], distance=dip_distance)
          start_point = min(peaks[0], dips[0])
          end_point = max(peaks[-1], dips[-1])
          new_signal[i, start_point:end_point] = signal_np[i, start_point:end_point]

        self.p_signal = new_signal

    # TODO: EXTERNAL, needs refactoring
    def augment_signal(self):
        """Apply enhanced augmentations to the signal"""
        if not self.is_train or not self.config.USE_AUGMENTATION:
            return p_signal

        # Time shift augmentation (increased probability and range)
        if 'shift' in self.augmentations:
            if np.random.random() < self.config.TIME_SHIFT_PROB:
                shift_factor = np.random.uniform(-self.config.TIME_SHIFT_MAX, self.config.TIME_SHIFT_MAX)
                shift_amount = int(shift_factor * p_signal.shape[1])
                if shift_amount > 0:
                    p_signal = torch.cat([p_signal[:, shift_amount:], p_signal[:, :shift_amount]], dim=1)
                elif shift_amount < 0:
                    shift_amount = abs(shift_amount)
                    p_signal = torch.cat([p_signal[:, -shift_amount:], p_signal[:, :-shift_amount]], dim=1)

        # TODO: scale should only be applied to the peaks/valleys
        if 'scale' in self.augmentations:
            # Amplitude scaling augmentation
            if np.random.random() < self.config.AMPLITUDE_SCALE_PROB:
                scale_factor = np.random.uniform(*self.config.AMPLITUDE_SCALE_RANGE)
                p_signal = p_signal * scale_factor

        if 'gauss' in self.augmentations:
            # Enhanced Gaussian noise
            if np.random.random() < self.config.GAUSSIAN_NOISE_PROB:
                noise = torch.randn_like(p_signal) * self.config.GAUSSIAN_NOISE_SCALE
                p_signal = p_signal + noise

        if 'dropout' in self.augmentations:
            # Lead dropout (randomly zero out leads to improve robustness)
            if np.random.random() < 0.8:  # 10% chance
                lead_idx = np.random.randint(0, int(0.05*p_signal.shape[0]))
                p_signal[lead_idx] = p_signal[lead_idx] * 0.0

        return p_signal

    # resample

    def quality_check(self, trimmed):
        #TODO: nk is FF-ing slow, find alternative..
        for sig in trimmed:
          quality = nk.ecg_quality(sig, method= 'zhao2018', approach= 'fuzzy', sampling_rate=self.fs)
          print(quality)

    def get(self)-> torch.tensor:
        # Convert back if needed
        if ~isinstance(self.p_signal, torch.Tensor):
            return torch.tensor(self.p_signal, dtype=torch.float16)
        else:
            return self.p_signal
