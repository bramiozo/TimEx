# Complete implementation using neurokit2, scipy, catch22, tsfresh
from typing import Literal, Dict, List, Annotated, Optional, Sequence
from numpy import ndarray, concatenate
import numpy as np

import neurokit2 as nk
from pycatch22 import catch22_all
from tsfresh.feature_extraction import extract_features
from tsfeatures import pacf_features, acf_features, stl_features, hurst
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

from scipy.signal import butter, filtfilt, detrend, savgol_filter

sys.path.append(os.path.join(os.getcwd(), '..', 'src'))
import extractor
import wavelets

NDArray2D = Annotated[ndarray, "2-dimensional ndarray"]

class Config:
    # Data settings
    LEADS = 12
    INPUT_LENGTH = 5_000
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    RANDOM_SEED = 42

    # Training settings
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 5


    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Enhanced augmentation for better generalization
    USE_AUGMENTATION = True
    GAUSSIAN_NOISE_PROB = 0.5
    GAUSSIAN_NOISE_SCALE = 0.02
    TIME_SHIFT_PROB = 0.6
    TIME_SHIFT_MAX = 0.2
    AMPLITUDE_SCALE_PROB = 0.5
    AMPLITUDE_SCALE_RANGE = (0.7, 1.2)

    # Preprocessing settings
    SAMPLING_RATE = 500

    # Stratified sampling to handle imbalance
    USE_STRATIFIED_SAMPLING = True
    OVERSAMPLE_RARE_DISEASES = Truere_diagnosis = re.compile(r'(Dx|Diagnosis):\s?([0-9]+)', re.IGNORECASE)


    # Further optimized class-specific thresholds
    USE_CLASS_THRESHOLDS = True
    DEFAULT_THRESHOLD = 0.5
    RARE_DISEASE_THRESHOLD = 0.2
    VERY_RARE_DISEASE_THRESHOLD = 0.15

    # Mixed precision to speed up training
    USE_MIXED_PRECISION = True

    # Balanced sampling weights
    BALANCE_SAMPLING_BY_CLASS = True

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
                 extractor_type: Literal['catch22', 'tsfresh', 'both'] = 'catch22',
                 aggregation: Literal['concatenate'] = 'concatenate',
                 extractor_groups: List[Literal['extractor', 'wavelets', 'ecgspecific', 'acf']] = ['wavelets'],
                 trimming_kwargs: Dict = {},
                 smoothing_kwargs: Dict = {'lowcut': 0.5, 'highcut': 40.0, 'order': 3}):
        self.sampling_rate = sampling_rate
        self.smoothing = smoothing
        self.trimming = trimming
        self.extractor_type = extractor_type
        self.aggregation = aggregation
        self.trimming_kwargs = trimming_kwargs
        self.smoothing_kwargs = smoothing_kwargs
        self.extractor_groups = extractor_groups
        self.feature_names = []

    @staticmethod
    def sanity_check(signal: ndarray) -> bool:
        """
        Perform a sanity check on the ECG signal.
        Check if the signal is not empty and contains valid values.
        """
        if signal is None or len(signal) == 0:
            return False
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return False
        if np.all(signal == 0):
            return False
        if np.std(signal) == 0:
            return False
        return True

    @staticmethod
    def _band_power(freqs, power, low, high):
        # Calculate the band power between low and high frequencies
        indices = np.logical_and(freqs >= low, freqs <= high)
        return np.trapz(power[indices], freqs[indices])

    @staticmethod
    def spectral_edge_freq(freqs, power, percent=95):
        """
        Calculate the spectral edge frequency (frequency below which percent% of total power is contained).
        
        Parameters:
        -----------
        freqs : array-like
            Frequency values
        power : array-like
            Power spectrum values corresponding to freqs
        percent : float
            Percentage of power (0-100) for which to find the edge frequency
            
        Returns:
        --------
        float
            Spectral edge frequency
        """
        if not 0 <= percent <= 100:
            raise ValueError("Percent must be between 0 and 100")
            
        # Calculate total power using trapezoidal rule
        total_power = np.trapz(power, freqs)
        
        # Target power
        target_power = total_power * percent/100
        
        # Calculate cumulative power using the same integration method
        cum_power = np.zeros_like(power)
        for i in range(len(freqs)):
            cum_power[i] = np.trapz(power[:i+1], freqs[:i+1])
        
        # Find where cumulative power exceeds target
        idx_exceeds = np.where(cum_power >= target_power)[0]
        
        if len(idx_exceeds) == 0:
            return freqs[-1]  # Return the highest frequency if target not reached
        
        idx = idx_exceeds[0]

            # Interpolate for more accurate result (optional)
        if idx > 0 and cum_power[idx] > target_power:
            # Linear interpolation between points
            x0, x1 = freqs[idx-1], freqs[idx]
            y0, y1 = cum_power[idx-1], cum_power[idx]
            return x0 + (x1 - x0) * (target_power - y0) / (y1 - y0)
        return freqs[idx]
    
    @staticmethod
    def spectral_entropy(power):
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
        # Add small constant to avoid log(0)
        eps = 1e-10
        # Normalize power spectrum
        power_normalized = power / (np.sum(power) + eps)
        # Calculate entropy
        entropy = -np.sum(power_normalized * np.log2(power_normalized + eps))
        return entropy

    @staticmethod
    def spectral_flatness(power):
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
        # Ensure all values are positive
        eps = 1e-10
        power_positive = np.maximum(power, eps)
        
        # Calculate geometric mean
        geometric_mean = np.exp(np.mean(np.log(power_positive)))
        
        # Calculate arithmetic mean
        arithmetic_mean = np.mean(power_positive)
        
        if arithmetic_mean == 0:
            return 0
        
        return geometric_mean / arithmetic_mean

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
        # Calculate the total power
        total_power = np.sum(power) * (freqs[1] - freqs[0])
        
        # Calculate the cumulative power
        cumulative_power = np.cumsum(power) * (freqs[1] - freqs[0])
        
        # Find the frequency where cumulative power reaches half of total power
        median_freq_idx = np.argmin(np.abs(cumulative_power - (total_power / 2)))
        
        return freqs[median_freq_idx]
    
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

        acf_v = np.array(list(_acf_features.values()))
        pacf_v = np.array(list(_pacf_features.values()))
        #stl_v = np.array(list(_stl_features.values()))
        #hurst_v = np.array(list(_hurst.values()))
        
        # concat names
        acf_names = list(_acf_features.keys())
        pacf_names = list(_pacf_features.keys())
        #stl_names = list(_stl_features.keys())
        #hurst_names = list(_hurst.keys())
        self.feature_names += [f'CHANNEL_{channel}_{n}' for n in acf_names] + \
                              [f'CHANNEL_{channel}_{n}' for n in pacf_names] 

        return np.concatenate([acf_v, pacf_v])


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
        
        psd_v = np.array(list(psd_features.values()))
        wvc_v = np.array(list(wvc_features.values()))
        fcs_v = np.array(list(fcs_features.values()))

        # Concatenate all names
        psd_names = list(psd_features.keys())
        wvc_names = list(wvc_features.keys())
        fcs_names = list(fcs_features.keys())
        self.feature_names += [f'CHANNEL_{channel}_{n}' for n in psd_names] + \
                              [f'CHANNEL_{channel}_{n}' for n in wvc_names] + \
                              [f'CHANNEL_{channel}_{n}' for n in fcs_names]

        return np.concatenate([psd_v, wvc_v, fcs_v])

    def _extractor_features(self, signal: ndarray, channel: int):
        features = []
        if self.extractor_type in ['catch22', 'both']:
            c22_features_df = catch22_all(signal)
            c22_features = c22_features_df['values']
            features.extend(c22_features)
            self.feature_names += list(c22_features_df['names'])

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
            tsfresh_features = tsfresh_features_df.values.flatten()
            features.extend(tsfresh_features)
            self.feature_names += [f'CHANNEL_{channel}_{n}' 
                                   for n in  list(tsfresh_features_df.columns)]

        return np.array(features)

    def _ecg_specific_features(self, signal: ndarray, channel: int):
        # Extract ECG-specific features: RR intervals, QRS features
        try:
            signals, _ = nk.ecg_process(signal, sampling_rate=self.sampling_rate)
            hr_features = nk.ecg_intervalrelated(signals)
            self.feature_names += [f'CHANNEL_{channel}_{n}' for n in list(hr_features.columns)]
            return hr_features.values.flatten()
        except Exception:
            return np.array([])

    def _extract_features_for_single_channel(self, 
                                             TimeSerie: ndarray, 
                                             channel: int) -> ndarray:
        # check if the signal is not empty
        # check if any of the features are empty

        if self.current_sampling_rate!=self.sampling_rate:
            TimeSerie = self._standardize_sampling_rate(TimeSerie)
        
        if self.smoothing:
            TimeSerie = self._smoothing(TimeSerie)

        if len(TimeSerie) == 0:
            return None
        
        wavelet_feats = np.array([])
        extractor_feats = np.array([])
        ecg_feats = np.array([])
        acf_feats = np.array([])
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
 
        # Concatenate all features
        return concatenate([wavelet_feats, 
                            extractor_feats, 
                            ecg_feats, 
                            acf_feats])

    def _extract_single_multichannel(self, TimeSeries: NDArray2D) -> ndarray:
        assert TimeSeries.ndim == 2

        features = [self._extract_features_for_single_channel(TimeSeries[:, ch], channel=ch) 
                    for ch in range(TimeSeries.shape[1])]

        # Concatenate all channel features
        return concatenate(features)

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
                if self.sanity_check(tsignal):
                    feats = self._extract_features_for_single_channel(tsignal)
                else:
                    print(f"Signal {key} failed sanity check. Skipping.")
                    continue
            elif ts[SignalCol].ndim == 2:
                tsignal = ts[SignalCol].T.numpy()
                if self.sanity_check(tsignal):
                    feats = self._extract_single_multichannel(tsignal)
                else:
                    print(f"Signal {key} failed sanity check. Skipping.")
                    continue
            else:
                raise ValueError(f"Unsupported ndarray dimension: {ts[SignalCol].T.ndim}")

            extracted_features[key] = dict(zip(self.feature_names, feats))

        return extracted_features



class ECGDataset(Dataset):
    """
        ECGDataset class for PhysioNet files
    """

    re_diagnosis = re.compile(r'(Dx|Diagnosis):\s?([0-9A-z]+)', re.IGNORECASE)
    re_age = re.compile(r'Age:\s?(\d+)', re.IGNORECASE) 
    re_sex = re.compile(r'Sex:\s?(\w+)', re.IGNORECASE)
    re_height = re.compile(r'Height:\s?(\d+)', re.IGNORECASE)
    re_weight = re.compile(r'Weight:\s?(\d+)', re.IGNORECASE)
    re_bmi = re.compile(r'BMI:\s?(\d+)', re.IGNORECASE)
    re_history = re.compile(r'Hx:\s?(\w+)', re.IGNORECASE)
    re_meds = re.compile(r'Rx:\s?(\w+)', re.IGNORECASE)

    def __init__(self,
        df,
        label_binarizer,
        is_train=True,
        supervised=True,
        extract_label=True,
        extract_metadata=True,
        config=None,
        augmentations: List[Literal['gauss', 'shift', 'scale', 'dropout']]=['gauss'],
        preprocessing: List[Literal['nan', 'bandpass', 'savgol',
                                     'powerline', 'standardscaler', 'resampler', 'truncate', 'detrend']]
                                     = ['bandpass', 'resampler']
        ):

        if isinstance(df, pd.DataFrame):
            self.df = df

            assert 'file_name' in df.columns, "The DataFrame must contain a 'file_name' column, referring to the .hea files"
            if (supervised) and ('label' not in df.columns) and (not extract_label):
                raise ValueError("The DataFrame must contain a 'label' column for supervised learning.")

            self.file_list = df['file_name'].tolist()
        elif isinstance(df, list):
            self.file_list = df
        elif isinstance(df, str) and os.path.isdir(df):
            self.file_list = []
            for root, _, files in os.walk(df):
                for file in files:
                    if file.endswith(('.hea', '.npy', '.h5')):
                        self.file_list.append(os.path.join(root, file))
        else:
            raise ValueError("Unsupported data source format." \
            " Must be DataFrame, list of paths, or folder.")

        self.supervised = supervised
        self.label_binarizer = label_binarizer
        self.is_train = is_train
        self.config = config if config else Config()
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.extract_label = extract_label
        self.extract_metadata = extract_metadata

    def __len__(self):
        return len(self.file_list)

    def _load_signal_from_source(self, source):
        """
        Load ECG signal from different formats:
        - WFDB (.hea + .dat)
        - HDF5 (.h5)
        - NumPy (.npy)
        - Folder structure (directory of .npy or .txt files)
        """
        if isinstance(source, str) and source.endswith('.npy'):
            signal = np.load(source)
            comment = None  # No WFDB record available
            self.format = 'npy'
            self.bands = None
            self.fs = None
            raise UserWarning("Be aware. NPY format does not contain metadata.")
        elif isinstance(source, str) and source.endswith('.h5'):
            with h5py.File(source, 'r') as f:
                signal = f['ecg'][:]  # or the correct key
            comment = None  # No WFDB record available
            self.format = 'h5'
            self.bands = None
            self.fs = None
            raise UserWarning("HDF5 format is not fully implemented yet. Metadata not yet parsed")
        elif isinstance(source, str) and os.path.isdir(source):
            # Load and stack individual lead files (e.g. lead_1.npy, lead_2.npy, ...)
            # TODO: this should be more flexible to handle different lead names
            leads = []
            lead_names = []
            for i in range(self.config.LEADS):
                bnd_name = f'lead_{i}'
                lead_file = os.path.join(source, f'{bnd_name}.npy')
                leads.append(np.load(lead_file))
                lead_names.append(bnd_name)
            signal = np.stack(leads, axis=0)
            comment = None  # No WFDB record available
            self.format = 'npy'
            self.fs = None
            self.bands = lead_names
            raise UserWarning("Be aware. NPY format does not contain metadata.")
        elif isinstance(source, str) and source.endswith('.hea'):
            record = wfdb.rdrecord(source.strip('.hea'))
            self.fs = record.fs
            self.bands = record.sig_name
            signal = record.p_signal.T
            comment = ",".join(record.__dict__['comments'])
            self.format = 'hea'
        else:
            raise ValueError(f"Unsupported input source: {source}")

        return torch.tensor(signal, dtype=torch.float32), comment


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

    def _standardize_sampling_rate(self, signal):
        signal_resampled, _ = resample_sig(signal.numpy(),
                                            self.fs, self.config.SAMPLING_RATE)
        signal = torch.tensor(signal_resampled)
        return signal

    def _bandpass_filter(self, signal, lowcut=0.5, highcut=40.0, order=4):
        """
        Apply a Butterworth bandpass filter to the ECG signal.

        Args:
            signal: np.array or torch tensor [channels, samples].
            lowcut: Lower cutoff frequency in Hz.
            highcut: Upper cutoff frequency in Hz.
            order: Order of the Butterworth filter.

        Returns:
            Filtered signal with same shape.
        """
        nyquist = 0.5 * self.config.SAMPLING_RATE
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(order, [low, high], btype='band')

        # Convert torch tensor to numpy if needed
        if isinstance(signal, torch.Tensor):
            signal_np = signal.numpy()
        else:
            signal_np = signal

        filtered_signal = filtfilt(b, a, signal_np, axis=1)

        # Convert back to tensor if original was a tensor
        if isinstance(signal, torch.Tensor):
            filtered_signal = torch.tensor(filtered_signal, dtype=signal.dtype)

        return filtered_signal

    def _powerline_filter(self, signal, freq=60, bandwidth=1.0, order=2):
        """
        Apply a powerline notch filter (bandstop) to remove mains interference.

        Args:
            signal: np.array or torch tensor with shape [channels, samples].
            freq: Central frequency of powerline interference (50 or 60 Hz).
            bandwidth: Bandwidth around the powerline frequency to remove (default ±1 Hz).
            order: Order of the Butterworth filter (default=2 recommended).

        Returns:
            Filtered signal with same shape.
        """
        nyquist = 0.5 * self.config.SAMPLING_RATE
        low = (freq - bandwidth) / nyquist
        high = (freq + bandwidth) / nyquist
        b, a = butter(order, [low, high], btype='bandstop')

        # Ensure it's numpy for scipy functions
        if isinstance(signal, torch.Tensor):
            signal_np = signal.numpy()
        else:
            signal_np = signal

        filtered_signal = filtfilt(b, a, signal_np, axis=1)

        # Convert back to original type if needed
        if isinstance(signal, torch.Tensor):
            filtered_signal = torch.tensor(filtered_signal, dtype=signal.dtype)

        return filtered_signal

    def _savgol_filter(self, signal, window_length=31, polyorder=3):
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
        if isinstance(signal, torch.Tensor):
            signal_np = signal.numpy()
        else:
            signal_np = signal

        # Apply Savitzky-Golay filter across each lead
        smoothed = savgol_filter(signal_np, window_length=window_length, polyorder=polyorder, axis=1)

        # Convert back if needed
        if isinstance(signal, torch.Tensor):
            smoothed = torch.tensor(smoothed, dtype=signal.dtype)

        return smoothed

    def _standardize_signal(self, signal):
        # Normalize each lead using robust standardization
        for i in range(signal.shape[0]):
            # Use percentile-based normalization instead of mean/std
            # This helps with handling outliers and artifacts in ECG
            sorted_vals, _ = torch.sort(signal[i])
            q_low = sorted_vals[int(0.25 * len(sorted_vals))]
            q_high = sorted_vals[int(0.75 * len(sorted_vals))]
            iqr = q_high - q_low + 1e-6

            # Center around median instead of mean
            median = torch.median(signal[i])
            signal[i] = (signal[i] - median) / iqr

        # Clip values to prevent extreme outliers
        signal = torch.clamp(signal, -5.0, 5.0)

    def _standardize_signal_length(self, signal):
        if signal.shape[1] > self.config.INPUT_LENGTH:
            # Center crop
            start = (signal.shape[1] - self.config.INPUT_LENGTH)
            signal = signal[:, start:start + self.config.INPUT_LENGTH]
        elif signal.shape[1] < self.config.INPUT_LENGTH:
            # Zero-padding
            pad = torch.zeros((self.config.LEADS, self.config.INPUT_LENGTH - signal.shape[1]))
            signal = torch.cat((signal, pad), dim=1)
        return signal

    def _detrend_signal(self, signal):
        """
        Remove linear trend from each lead (channel).

        Args:label_binarizer
            Detrended signal of the same shape.
        """
        if isinstance(signal, torch.Tensor):
            signal_np = signal.numpy()
        else:
            signal_np = signal

        detrended = detrend(signal_np, axis=1, type='linear')

        if isinstance(signal, torch.Tensor):
            detrended = torch.tensor(detrended, dtype=signal.dtype)

        return detrended

    def preprocess_signal(self, signal):
        """Enhanced preprocessing with better handling of ECG characteristics"""

        # nan: Handle NaN values and outliers
        if 'nan' in self.preprocessing:
            signal = torch.nan_to_num(signal, nan=0.0, posinf=3.0, neginf=-3.0)
        # detrend: Remove linear trend
        if 'detrend' in self.preprocessing:
            signal = self._detrend_signal(signal)
        # bandpass: Remove baseline wander (high-pass filter simulation)
        if 'bandpass' in self.preprocessing:
            signal = self._bandpass_filter(signal, lowcut=0.5, highcut=40.0)
        # savgol: Apply Savitzky-Golay filter for smoothing
        if 'savgol' in self.preprocessing:
            signal = self._savgol_filter(signal, window_length=31, polyorder=3)
        # powerline: Remove powerline interference (notch filter)
        if 'powerline' in self.preprocessing:
            signal = self._powerline_filter(signal, freq=60, bandwidth=1.0)
        # resampler: Resample to a standard sampling rate
        if 'resampler' in self.preprocessing:
            signal = self._standardize_sampling_rate(signal)
        # standardscaler: Standardizesignal
        # truncate: Standardize signal length
        if 'truncate' in self.preprocessing:
            signal = self._standardize_signal_length(signal)
        return signal

    def __getitem__(self, idx:int)-> tuple:
        try:
            file_path = self.file_list[idx]
            signal, record = self._load_signal_from_source(file_path)
            label = None
            metadata = None

            if len(self.preprocessing) > 0:
                signal = self.preprocess_signal(signal)

            if len(self.augmentations) > 0:
                signal = self.augment_signal(signal)

            if record is not None:
                if self.extract_label:
                    try:
                        label = re.findall(self.re_diagnosis, record)[0][1]
                    except:
                        label = None

                if self.extract_metadata:
                    try:
                        age  = int(re.findall(self.re_age, record)[0])
                    except:
                        age = None
                    
                    try:
                        sex = re.findall(self.re_sex, record)[0]
                    except:
                        sex = None

                    try:
                        height = int(re.findall(self.re_height, record)[0])
                    except:
                        height = None

                    try:
                        weight = int(re.findall(self.re_weight, record)[0])
                    except:
                        weight = None
                    
                    try:
                        bmi = int(re.findall(self.re_bmi, record)[0])
                    except:
                        bmi = None

                    try:
                        meds = re.findall(self.re_meds, record)[0]
                    except:
                        meds = None

                    try:
                        history = re.findall(self.re_history, record)[0]
                    except:
                        history = None

                    metadata = {
                        'age': age,
                        'sex': sex,
                        'height': height,
                        'weight': weight,
                        'bmi': bmi,
                        'meds': meds,
                        'history': history,
                        'fs': self.fs,
                        'bands': self.bands
                    }
                # Convert labels
                if self.supervised and not self.extract_label:
                    label = torch.tensor(
                        self.label_binarizer.transform([label])[0],
                        dtype=torch.float32
                    )
            else:
                # If no record, set metadata to None
                metadata = None


            return file_path, idx, signal, label, metadata
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
            signal = torch.zeros((self.config.LEADS, self.config.INPUT_LENGTH))
            label = torch.zeros(len(self.label_binarizer.classes_))
            return file_path, idx, signal, label, None
