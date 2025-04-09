# Complete implementation using neurokit2, scipy, catch22, tsfresh
from typing import Literal, Dict, List, Annotated, Optional, Sequence
from numpy import ndarray, concatenate
import numpy as np
import neurokit2 as nk
from pycatch22 import catch22_all
from tsfresh.feature_extraction import extract_features
import pandas as pd
import torch
import os
import sys
import gc
import time
import wfdb
from wfdb.processing import resample_sig
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
re_diagnosis = re.compile(r'(Dx|Diagnosis):\s?([0-9A-z]+)', re.IGNORECASE)

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
    def __init__(self,
                 sampling_rate: int = 1000,
                 smoothing: bool = True,
                 trimming: bool = True,
                 extractor: Literal['catch22', 'tsfresh', 'both'] = 'catch22',
                 aggregation: Literal['concatenate'] = 'concatenate',
                 trimming_kwargs: Dict = {},
                 smoothing_kwargs: Dict = {}):
        self.sampling_rate = sampling_rate
        self.smoothing = smoothing
        self.trimming = trimming
        self.extractor = extractor
        self.aggregation = aggregation
        self.trimming_kwargs = trimming_kwargs
        self.smoothing_kwargs = smoothing_kwargs

    def _smoothing(self, TimeSerie: ndarray) -> ndarray:
        # Default Neurokit smoothing (low-pass filtering)
        return nk.signal_filter(TimeSerie, sampling_rate=self.sampling_rate, 
                                **self.smoothing_kwargs)

    def _trimming(self, TimeSerie: ndarray) -> ndarray:
        # Trim ECG between the first and last R-peaks
        peaks, _ = nk.ecg_peaks(TimeSerie, sampling_rate=self.sampling_rate)
        peak_indices = peaks['ECG_R_Peaks']
        return TimeSerie[peak_indices[0]:peak_indices[-1]]

    def _wavelet_features(self, signal: ndarray):
        # Continuous Wavelet Transform (CWT) based features
        coefs, freqs = nk.signal_cwt(signal, sampling_rate=self.sampling_rate)
        re_diagnosis = re.compile(r'(Dx|Diagnosis):\s?([0-9]+)', re.IGNORECASE)

        wvc = wavelets.extract_wavelet_features(signal, wavelet='db4',
                                                level=3, num_features=6)
        fcs = extractor.extract_fft_features(signal, num_features=8,
                                             max_frequency=40)

        return np.concat([coefs.mean(axis=1),
                          wvc,
                          fcs])

    def _extractor_features(self, signal: ndarray):
        features = []
        if self.extractor in ['catch22', 'both']:
            c22_features = catch22_all(signal)['values']
            features.extend(c22_features)

        if self.extractor in ['tsfresh', 'both']:
            tsfresh_df = pd.DataFrame({'signal': signal})
            tsfresh_features = extract_features(tsfresh_df, column_id=None, disable_progressbar=True).values.flatten()
            features.extend(tsfresh_features)

        return np.array(features)

    def _ecg_specific_features(self, signal: ndarray):
        # Extract ECG-specific features: RR intervals, QRS features
        try:
            signals, info = nk.ecg_process(signal, sampling_rate=self.sampling_rate)
            hr_features = nk.ecg_intervalrelated(info)
            return hr_features.values.flatten()
        except Exception:
            return np.array([])

    def _extract_features_for_single_channel(self, TimeSerie: ndarray) -> ndarray:
        if self.smoothing:
            TimeSerie = self._smoothing(TimeSerie)

        if self.trimming:
            TimeSerie = self._trimming(TimeSerie)

        wavelet_feats = self._wavelet_features(TimeSerie)
        extractor_feats = self._extractor_features(TimeSerie)
        ecg_feats = self._ecg_specific_features(TimeSerie)

        # Concatenate all features
        return concatenate([wavelet_feats, extractor_feats, ecg_feats])

    def _extract_single_multichannel(self, TimeSeries: NDArray2D) -> ndarray:
        assert TimeSeries.ndim == 2

        features = [self._extract_features_for_single_channel(TimeSeries[:, ch]) 
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

class ECGDataset(Dataset):
    """
        ECGDataset class for PhysioNet files
    """
    def __init__(self,
        df,
        label_binarizer,
        is_train=True,
        supervised=True,
        extract_label=True,
        config=None,
        augmentations: List[Literal['gauss', 'shift', 'scale', 'dropout']]=['gauss'],
        preprocessing: List[Literal['nan', 'bandpass', 'savgol',
                                     'powerline', 'standardscaler', 'resampler', 'truncate', 'detrend']]
                                     =['bandpass']
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
        elif isinstance(source, str) and source.endswith('.h5'):
            with h5py.File(source, 'r') as f:
                signal = f['ecg'][:]  # or the correct key
            comment = None  # No WFDB record available
        elif isinstance(source, str) and os.path.isdir(source):
            # Load and stack individual lead files (e.g. lead_1.npy, lead_2.npy, ...)
            leads = []
            for i in range(self.config.LEADS):
                lead_file = os.path.join(source, f'lead_{i}.npy')
                leads.append(np.load(lead_file))
            signal = np.stack(leads, axis=0)
            comment = None  # No WFDB record available
        elif isinstance(source, str) and source.endswith('.hea'):
            record = wfdb.rdrecord(source.strip('.hea'))
            self.fs = record.fs
            signal = record.p_signal.T
            comment = ",".join(record.__dict__['comments'])
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
        signal_resampled, _ = resample_sig(signal.numpy(), self.fs, self.config.SAMPLING_RATE)
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
        # standardscaler: Standardize
        if 'standardscaler' in self.preprocessing:
            signal = self._standardize_signal(signal)
        # truncate: Standardize signal length
        if 'truncate' in self.preprocessing:
            signal = self._standardize_signal_length(signal)
        return signal

    def __getitem__(self, idx):
        try:
            file_path = self.file_list[idx]
            signal, record = self._load_signal_from_source(file_path)
            label = None

            if len(self.preprocessing) > 0:
                signal = self.preprocess_signal(signal)

            if len(self.augmentations) > 0:
                signal = self.augment_signal(signal)

            if self.extract_label and (record is not None):
                try:
                    label = re.findall(re_diagnosis, record)[0][1]
                except:
                    label = None

            # Convert labels
            if self.supervised and not self.extract_label:
                label = torch.tensor(
                    self.label_binarizer.transform([label])[0],
                    dtype=torch.float32
                )

            return signal, label, file_path
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
            signal = torch.zeros((self.config.LEADS, self.config.INPUT_LENGTH))
            label = torch.zeros(len(self.label_binarizer.classes_))
            return signal, label, idx
