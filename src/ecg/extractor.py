# Complete implementation using neurokit2, scipy, catch22, tsfresh
from typing import Literal, Dict, List, Annotated, Optional, Sequence
from numpy import ndarray, concatenate
import numpy as np

import neurokit2 as nk
from pycatch22 import catch22_all
from tsfresh.feature_extraction import extract_features
from tsfeatures import pacf_features, acf_features, stl_features, hurst
import tsfel
import pandas as pd
import torch
import os
import sys
import gc
import time
import wfdb
from wfdb.processing import resample_sig # should be default to normalize fs
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
# PyTorch imports
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import re
import h5py
import numba
from scipy.signal import butter, filtfilt, detrend, savgol_filter

sys.path.append(os.path.join(os.getcwd(), '..', 'src'))
import extractor
import wavelets

NDArray2D = Annotated[ndarray, "2-dimensional ndarray"]


@numba.njit
def numba_sanity_check_1d(signal: ndarray) -> bool:
    """Numba-accelerated implementation of signal sanity check for 1D arrays."""
    # Check if empty (length check)
    if len(signal) == 0:
        return False

    # Check for NaN or Inf
    for val in signal:
        if np.isnan(val) or np.isinf(val):
            return False

    # Check if all zeros
    all_zeros = True
    for val in signal:
        if val != 0:
            all_zeros = False
            break
    if all_zeros:
        return False

    # Check std == 0 (constant signal)
    mean = 0.0
    for val in signal:
        mean += val
    mean /= len(signal)

    var = 0.0
    for val in signal:
        var += (val - mean) ** 2
    var /= len(signal)

    if var == 0:
        return False

    return True

def numba_sanity_check(signal: ndarray, num_check: int=10_000) -> bool:
    """
    Wrapper function to handle different array types and shapes.
    """
    # Make sure we have a numpy array
    signal = np.asarray(signal)

    # Handle different dimensionality
    if signal.ndim == 1:
        return numba_sanity_check_1d(signal[:num_check])
    elif signal.ndim == 2:
        # Check each channel (row or column)
        for i in range(signal.shape[0]):
            if numba_sanity_check_1d(signal[i][:num_check]):
                return True
        return False
    else:
        raise ValueError(f"Unsupported array dimension: {signal.ndim}. \
                         Only 1D and 2D arrays are supported.")


@numba.njit
def spectral_edge_freq_impl(freqs, power, percent):
    # NOTE: in no-Python mode, can't raise normal Python exceptions
    # We'll return NaN if 'percent' is invalid
    if percent < 0.0 or percent > 100.0:
        return np.nan

    n = freqs.size
    if n < 2:
        return freqs[0] if n == 1 else np.nan

    total_power = 0.0
    for i in range(n-1):
        total_power += 0.5 * (power[i] + power[i+1]) * (freqs[i+1] - freqs[i])

    target_power = total_power * (percent / 100.0)

    cum_power = 0.0
    for i in range(n-1):
        prev_cum = cum_power
        trap = 0.5 * (power[i] + power[i+1]) * (freqs[i+1] - freqs[i])
        cum_power += trap

        if cum_power >= target_power:
            segment_power = cum_power - prev_cum
            if abs(segment_power) < 1e-14:
                return freqs[i+1]

            needed = target_power - prev_cum
            frac = needed / segment_power
            return freqs[i] + (freqs[i+1] - freqs[i]) * frac

    return freqs[-1]

@numba.njit
def spectral_entropy_impl(power):
    eps = 1e-10
    denom = 0.0
    for p in power:
        denom += p
    denom = denom + eps  # ensure nonzero

    # Calculate normalized power
    # and accumulate the sum of p_i*log2(p_i)
    entropy_sum = 0.0
    for p in power:
        p_norm = p / denom
        # Avoid log(0), so add eps inside the log as well
        entropy_sum += p_norm * np.log2(p_norm + eps)

    # spectral entropy = - sum(p_norm * log2(p_norm)) ...
    return -entropy_sum


@numba.njit
def spectral_flatness_impl(power):
    eps = 1e-10
    # Compute geometric mean
    # geometric_mean = exp(mean(log(power_positive)))
    log_sum = 0.0
    n = 0
    for p in power:
        p_pos = p if p > eps else eps  # clamp to avoid log(0)
        log_sum += np.log(p_pos)
        n += 1
    geometric_mean = np.exp(log_sum / n)

    # Compute arithmetic mean
    a_sum = 0.0
    for p in power:
        a_sum += p if p > 0.0 else 0.0
    arithmetic_mean = a_sum / n if n > 0 else eps

    if arithmetic_mean < 1e-14:
        return 0.0
    return geometric_mean / arithmetic_mean


@numba.njit
def median_power_impl(freqs, power):
    n = freqs.size
    if n < 1:
        return np.nan

    df = freqs[1] - freqs[0] if n > 1 else 1.0

    total_power = 0.0
    for p in power:
        total_power += p
    total_power *= df

    half_power = total_power / 2.0

    cum = 0.0
    for i in range(n):
        cum += power[i]
        if (cum * df) >= half_power:
            return freqs[i]

    # Should not happen unless power is extremely small
    return freqs[-1]



class ECGxtract():
    p_wave_band = 0.5, 3      # P-wave components
    qrs_complex_band = (4, 20)   # QRS complex components
    t_wave_band = (0.5, 7)      # T-wave components
    baseline_band = (0, 0.5)     # Baseline wander
    mains_noise_band = (49, 51)  # 50Hz power line interference
    low_freq_band = (0.04, 0.15) # low frequency
    high_freq_band = (0.15, 0.4) # high frequency
    noise_band = (40, 100)     # noise

    def __init__(self,
                 sampling_rate: int = 500,
                 smoothing: bool = True,
                 trimming: bool = True,
                 sanity_check: bool = True,
                 min_window_size: int = 25_000,
                 extractor_type: Literal['catch22', 'tsfresh', 'both'] = 'catch22',
                 aggregation: Literal['concatenate'] = 'concatenate',
                 extractor_groups: List[Literal['tsfel', 'extractor', 'wavelets', 'ecgspecific', 'acf']] = ['wavelets'],
                 trimming_kwargs: Dict = {},
                 smoothing_kwargs: Dict = {'lowcut': 0.5, 'highcut': 40.0, 'order': 3}):
        """ECGExtract

        This class is meant for feature extraction from ECG signals.
        The class contains some pre-processing steps such as trimming, smoothing and down/up sampling to 500Hz.
        """

        self.sampling_rate = sampling_rate
        self.smoothing = smoothing
        self.trimming = trimming
        self.sanity_check = sanity_check
        self.extractor_type = extractor_type
        self.aggregation = aggregation
        self.trimming_kwargs = trimming_kwargs
        self.smoothing_kwargs = smoothing_kwargs
        self.extractor_groups = extractor_groups
        self.min_window_size = min_window_size


    @staticmethod
    def _sanity_check(signal: ndarray) -> bool:
        """
        Perform a sanity check on the ECG signal.
        Check if the signal is not empty and contains valid values.
        """
        if signal is None:
            return False
        return numba_sanity_check(signal)

    @staticmethod
    def _band_power(freqs, power, low, high):
        # Calculate the band power between low and high frequencies
        indices = np.logical_and(freqs >= low, freqs <= high)
        return np.trapz(power[indices], freqs[indices])

    @staticmethod
    def _spectral_edge_freq(freqs, power, percent=95.0):
        """
        Calculate the spectral edge frequency (frequency below which percent% of total power is contained).
        """
        # You can still do Python checks or raise exceptions here
        if not (0 <= percent <= 100):
            raise ValueError("Percent must be between 0 and 100")
        if len(freqs) == 0:
            raise ValueError("freqs must not be empty")
        if len(freqs) != len(power):
            raise ValueError("freqs and power must have the same length")

        return spectral_edge_freq_impl(freqs, power, percent)

    @staticmethod
    def _spectral_entropy(power):
        """
        Calculate the spectral entropy of a power spectrum.

        Parameters:
        -----------
        power : array-like
            Power spectrum values

        Returns:
        --------
        float
            Spectral entropy value
        """
        if len(power) == 0:
            return 0.0
        return spectral_entropy_impl(power)

    @staticmethod
    def _spectral_flatness(power):
        """
        Calculate the spectral flatness (Wiener entropy) of a power spectrum.

        Parameters:
        -----------
        power : array-like
            Power spectrum values

        Returns:
        --------
        float
            Spectral flatness value between 0 and 1
        """
        if len(power) == 0:
            return 0.0
        return spectral_flatness_impl(power)


    @staticmethod
    def _median_power(freqs, power):
        """
        Calculate the median power frequency (frequency that divides the power spectrum in half).

        Parameters:
        -----------
        freqs : array-like
            Frequency values
        power : array-like
            Power spectrum values corresponding to freqs

        Returns:
        --------
        float
            Median power frequency
        """
        if len(freqs) == 0:
            return np.nan
        if len(freqs) != len(power):
            raise ValueError("freqs and power must have the same length")

        return median_power_impl(freqs, power)

    def _smoothing(self, TimeSerie: ndarray) -> ndarray:
        # Default Neurokit smoothing (low-pass filtering)
        return nk.signal_filter(TimeSerie, sampling_rate=self.sampling_rate,
                                **self.smoothing_kwargs)


    def _standardize_sampling_rate(self, signal):
        signal_resampled, _ = resample_sig(signal,
                                           self.current_sampling_rate,
                                           self.sampling_rate)
        return signal_resampled

    def _trimming(self, TimeSerie: ndarray, max_trim_factor=0.2) -> ndarray:
        # Trim ECG between the first and last R-peaks
        _, peaks = nk.ecg_peaks(TimeSerie, sampling_rate=self.sampling_rate)
        peak_indices = peaks['ECG_R_Peaks']

        if peak_indices[-1]-peak_indices[0] < len(TimeSerie) * max_trim_factor:
            # If the distance is too small, return the original signal
            print("The distance between the first and last R-peaks is too small." \
            " Returning the original signal.")
            return TimeSerie

        return TimeSerie[peak_indices[0]:peak_indices[-1]]

    def _tsfel_features(self, signal: ndarray, channel: int):
        # 'statistical', 'temporal', 'spectral', 'fractal'
        cfg = tsfel.get_features_by_domain(['temporal'])
        ws = min(len(signal) // 2, self.min_window_size)
        if len(signal) < ws:
            print(f"Signal length {len(signal)} is smaller than the window size {ws}.")
            return None
        res = tsfel.time_series_features_extractor(cfg,
                                         signal,
                                         fs=self.sampling_rate,
                                         window_size=ws,
                                         verbose=0,
                                         overlap=0.5,
                                         n_jobs=1).mean().to_dict()
        # update key names with the channel number
        res = {f'CHANNEL_{channel}_{k}': v for k, v in res.items()}
        return res

    def _peak_features(self, signal: ndarray, channel: int):
        pass


    def _model_features(self, signal: ndarray, channel: int):
        # Extract features using a pre-trained autoencoder
        pass

    def _reg_features(self, signal: ndarray, channel: int):
        _acf_features = acf_features(signal, freq=self.sampling_rate)
        _pacf_features = pacf_features(signal, freq=self.sampling_rate)
        #_stl_features = stl_features(signal, freq=self.sampling_rate)
        #_hurst = hurst(signal, freq=self.sampling_rate)
        # MERGE dicts
        _features = {**_acf_features, **_pacf_features}
        # update key names with the  channel number
        _features = {f'CHANNEL_{channel}_{k}': v for k, v in _features.items()}
        return _features


    def _wavelet_features(self, signal: ndarray, channel: int):
        # Continuous Wavelet Transform (CWT) based features
        psd_chars = nk.signal_psd(signal, sampling_rate=self.sampling_rate, method='welch').values
        FREQS = psd_chars[:,0]
        POWER = psd_chars[:,1]

        # frequency of maximum power
        dominant_frequency = FREQS[np.argmax(POWER)]
        # Median frequency, frequency that divides the cumulative power spectrum in two equal parts
        median_freq = self._median_power(FREQS, POWER)

        p_wave_band_power = self._band_power(FREQS, POWER, *self.p_wave_band) # delta
        qrs_complex_band_power = self._band_power(FREQS, POWER, *self.qrs_complex_band) # theta
        t_wave_band_power = self._band_power(FREQS, POWER, *self.t_wave_band) # alpha
        baseline_band_power = self._band_power(FREQS, POWER, *self.baseline_band) # beta
        mains_noise_band_power = self._band_power(FREQS, POWER, *self.mains_noise_band) # gamma
        noise_band_power = self._band_power(FREQS, POWER, *self.noise_band) # noise
        lf_power = self._band_power(FREQS, POWER, *self.low_freq_band) # low frequency
        hf_power = self._band_power(FREQS, POWER, *self.high_freq_band) # high frequency
        total_power = np.trapz(POWER, FREQS) # total power

        # ratio of low frequency to high frequency power
        lf_hf_ratio = lf_power / hf_power
        # ratio of low frequency to total power
        lf_total_ratio = lf_power / total_power
        qrs_noise_ratio = qrs_complex_band_power / noise_band_power

        #######
        _spectral_edge_freq = self._spectral_edge_freq(FREQS, POWER)

        #######
        _spectral_entropy = self._spectral_entropy(POWER)

        #######
        _spectral_flatness = self._spectral_flatness(POWER)

        #######
        spectral_centroid = np.sum(FREQS * POWER) / np.sum(POWER)

        psd_features = {
            'dominant_frequency': dominant_frequency,
            'median_freq': median_freq,
            'p_wave_band_power': p_wave_band_power,
            'qrs_complex_band_power': qrs_complex_band_power,
            't_wave_band_power': t_wave_band_power,
            'baseline_band_power': baseline_band_power,
            'main_noise_band_power': mains_noise_band_power,
            'noise_band_power': noise_band_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'lf_hf_ratio': lf_hf_ratio,
            'lf_total_ratio': lf_total_ratio,
            'qrs_noise_ratio': qrs_noise_ratio,
            'spectral_edge_freq': _spectral_edge_freq,
            'spectral_entropy': _spectral_entropy,
            'spectral_flatness': _spectral_flatness,
            'spectral_centroid': spectral_centroid
        }
        ################################
        wvc_features = wavelets.extract_wavelet_features(signal, wavelet='db4',
                                                level=3, num_features=6)
        fcs_features = extractor.extract_fft_features(signal, num_features=8,
                                             max_frequency=40)

        # merge features dicts
        _features = {**psd_features, **wvc_features, **fcs_features}
        # update key names with the channel number
        _features = {f'CHANNEL_{channel}_{k}': v for k, v in _features.items()}
        return _features

    def _extractor_features(self, signal: ndarray, channel: int):
        _features = {}
        if self.extractor_type in ['catch22', 'both']:
            c22_features_df = catch22_all(signal, catch24=True)
            c22_features = dict(zip(c22_features_df['names'],
                                    c22_features_df['values']))
            _features.update(c22_features)

        if self.extractor_type in ['tsfresh', 'both']:
            tsfresh_df = pd.DataFrame({
                                       'id': np.zeros(len(signal)),
                                       'time': np.arange(len(signal)),
                                       'signal': signal})
            tsfresh_features_df = extract_features(tsfresh_df,
                                                column_value='signal',
                                                column_sort='time',
                                                column_id='id',
                                                disable_progressbar=True)
            tsfresh_features = dict(zip(tsfresh_features_df.columns,
                                        tsfresh_features_df.values.flatten()))
            # Merge features
            _features.update(tsfresh_features)

        _features = {f'CHANNEL_{channel}_{k}': v for k, v in _features.items()}
        return _features

    def _ecg_specific_features(self, signal: ndarray, channel: int):
        # Extract ECG-specific features: RR intervals, QRS features
        try:
            signals, _ = nk.ecg_process(signal, sampling_rate=self.sampling_rate)
            hr_features = nk.ecg_intervalrelated(signals)
            _features = dict(zip(hr_features.columns, hr_features.values.flatten()))
            _features = {f'CHANNEL_{channel}_{k}': v for k, v in _features.items()}
            return _features
        except Exception:
            return None

    def _extract_features_for_single_channel(self,
                                             TimeSerie: ndarray,
                                             channel: int) -> ndarray:
        # check if the signal is not empty
        # check if any of the features are empty

        if self.current_sampling_rate!=self.sampling_rate:
            #print(f"Resampling signal...{self.current_sampling_rate} to {self.sampling_rate}")
            try:
                TimeSerie = self._standardize_sampling_rate(TimeSerie)
            except Exception as e:
                print(f"Error resampling: {e}\n TimeSerie: {TimeSerie}")
                return {}

        if self.smoothing:
            try:
                TimeSerie = self._smoothing(TimeSerie)
            except Exception as e:
                print(f"Error applying smoothing: {e}\n TimeSerie: {TimeSerie}")
                return {}

        if len(TimeSerie) == 0:
            return {}

        wavelet_feats = {}
        extractor_feats = {}
        ecg_feats = {}
        acf_feats = {}
        tsfel_feats = {}
        # Extract features based on the selected methods
        if 'wavelets' in self.extractor_groups:
            try:
                wavelet_feats = self._wavelet_features(TimeSerie, channel)
            except Exception as e:
                print(f"Error extracting wavelet features: {e}\n TimeSerie: {TimeSerie}")
                raise ValueError("Wavelet feature extraction failed.")
        if 'extractor' in self.extractor_groups:
            try:
                extractor_feats = self._extractor_features(TimeSerie, channel)
            except Exception as e:
                print(f"Error applying extractor: {e}\n TimeSerie: {TimeSerie}")
                raise ValueError("Extractor feature extraction failed.")
        if 'ecgspecific' in self.extractor_groups:
            try:
                ecg_feats = self._ecg_specific_features(TimeSerie, channel)
            except:
                print(f"Error applying ecg_specific: {e}\n TimeSerie: {TimeSerie}")
                raise ValueError("ECG-specific feature extraction failed.")
        if 'acf' in self.extractor_groups:
            try:
                acf_feats = self._reg_features(TimeSerie, channel)
            except Exception as e:
                print(f"Error applying acf: {e}\n TimeSerie: {TimeSerie}")
                raise ValueError("ACF feature extraction failed.")
        if 'tsfel' in self.extractor_groups:
            try:
                tsfel_feats = self._tsfel_features(TimeSerie, channel)
            except Exception as e:
                print(f"Error applying tsfel: {e}\n TimeSerie: {TimeSerie}")
                raise ValueError("TSFEL feature extraction failed.")

        _features = {
            **wavelet_feats,
            **extractor_feats,
            **ecg_feats,
            **acf_feats,
            **tsfel_feats
        }
        return _features

    def _extract_single_multichannel(self, TimeSeries: NDArray2D) -> ndarray:
        assert TimeSeries.ndim == 2

        features = [self._extract_features_for_single_channel(TimeSeries[:, ch], channel=ch)
                    for ch in range(TimeSeries.shape[1])]

        # features is a list of dictionaries, merge into on
        _features = {}
        for f in features:
            _features.update(f)
        return _features

    def extract_from_list(self, TimeSeries: List[ndarray]) -> List[ndarray]:
        extracted_features = []
        for ts in TimeSeries:
            if ts.ndim == 1:
                feats = self._extract_features_for_single_channel(ts)
            elif ts.ndim == 2:
                feats = self._extract_single_multichannel(ts)
            else:
                raise ValueError(f"Unsupported ndarray dimension: {ts.ndim}")

            extracted_features.append(feats)

        return extracted_features

    def extract_from_dict(self, TimeSeries: Dict,
                          SignalCol: str='signal',
                          FsCol: str='fs') -> Dict[str, ndarray]:
        extracted_features = {}
        for key, ts in tqdm(TimeSeries.items()):
            self.current_sampling_rate = ts[FsCol]
            self.feature_names = []
            if ts[SignalCol].ndim == 1:
                tsignal = ts[SignalCol].T.numpy()
                # sanitycheck
                if self.sanity_check==False:
                    feats = self._extract_features_for_single_channel(tsignal)
                elif self._sanity_check(tsignal):
                    feats = self._extract_features_for_single_channel(tsignal)
                else:
                    print(f"Signal {key} failed sanity check. Skipping.")
                    continue
            elif ts[SignalCol].ndim == 2:
                tsignal = ts[SignalCol].T.numpy()
                if self.sanity_check==False:
                    feats = self._extract_single_multichannel(tsignal)
                elif self._sanity_check(tsignal):
                    feats = self._extract_single_multichannel(tsignal)
                else:
                    print(f"Signal {key} failed sanity check. Skipping.")
                    continue
            else:
                raise ValueError(f"Unsupported ndarray dimension: {ts[SignalCol].T.ndim}")

            extracted_features[key] = feats

        return extracted_features

if __name__ == "__main__":
    print("Starting...")
