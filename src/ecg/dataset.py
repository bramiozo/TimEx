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

class Config:
    # Data settings
    LEADS = 12
    INPUT_LENGTH = 5_000
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    RANDOM_SEED = 42
    SAMPLING_RATE = 500

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

NDArray2D = Annotated[ndarray, "2-dimensional ndarray"]

class ECGDataset(Dataset):
    """
        ECGDataset class for PhysioNet files
    """

    re_diagnosis = re.compile(r'(Dx|Diagnosis):\s?([0-9A-z,]+)', re.IGNORECASE)
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
        supervised=False,
        extract_label=True,
        extract_metadata=True,
        config=None,
        augmentations: List[Literal['gauss', 'shift', 'scale', 'dropout']]=['gauss', 'dropout'],
        preprocessing: List[Literal['nan', 'bandpass', 'savgol',
                                    'powerline', 'standardscaler', 'resampler', 'truncate', 'detrend']]
                                     = ['bandpass', 'resampler', 'detrend']
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

        if supervised is False:
            extract_label = False

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
            comment = "|".join(record.__dict__['comments'])
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
            return None, idx, signal, label, None
