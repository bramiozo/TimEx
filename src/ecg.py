# Complete implementation using neurokit2, scipy, catch22, tsfresh
from typing import Literal, Dict, List, Annotated, Optional
from numpy import ndarray, concatenate
import numpy as np
import neurokit2 as nk
from catch22 import catch22_all
from tsfresh.feature_extraction import extract_features
import pandas as pd
import torch
import os
import sys
import gc
import time
import wfdb
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
# PyTorch imports
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.append(os.path.join(os.getcwd(), '..', 'src'))
import extractor
import wavelets

NDArray2D = Annotated[ndarray, "2-dimensional ndarray"]

class Config:
    # Data settings
    LEADS = 12
    INPUT_LENGTH = 10_000
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    RANDOM_SEED = 42

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

    # Stratified sampling to handle imbalance
    USE_STRATIFIED_SAMPLING = True
    OVERSAMPLE_RARE_DISEASES = True

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
        return nk.signal_filter(TimeSerie, sampling_rate=self.sampling_rate, **self.smoothing_kwargs)

    def _trimming(self, TimeSerie: ndarray) -> ndarray:
        # Trim ECG between the first and last R-peaks
        peaks, _ = nk.ecg_peaks(TimeSerie, sampling_rate=self.sampling_rate)
        peak_indices = peaks['ECG_R_Peaks']
        return TimeSerie[peak_indices[0]:peak_indices[-1]]

    def _wavelet_features(self, signal: ndarray):
        # Continuous Wavelet Transform (CWT) based features
        coefs, freqs = nk.signal_cwt(signal, sampling_rate=self.sampling_rate)
    
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

        features = [self._extract_features_for_single_channel(TimeSeries[:, ch]) for ch in range(TimeSeries.shape[1])]

        # Concatenate all channel features
        return concatenate(features)

    def extract(self, TimeSeries: List[ndarray]) -> List[ndarray]:
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
    def __init__(self, df, label_binarizer, is_train=True, config=None):
        self.df = df
        self.label_binarizer = label_binarizer
        self.is_train = is_train
        self.config = config if config else Config()

    def __len__(self):
        return len(self.df)

    def augment_signal(self, signal):
        """Apply enhanced augmentations to the signal"""
        if not self.is_train or not self.config.USE_AUGMENTATION:
            return signal

        # Time shift augmentation (increased probability and range)
        if np.random.random() < self.config.TIME_SHIFT_PROB:
            shift_factor = np.random.uniform(-self.config.TIME_SHIFT_MAX, self.config.TIME_SHIFT_MAX)
            shift_amount = int(shift_factor * signal.shape[1])
            if shift_amount > 0:
                signal = torch.cat([signal[:, shift_amount:], signal[:, :shift_amount]], dim=1)
            elif shift_amount < 0:
                shift_amount = abs(shift_amount)
                signal = torch.cat([signal[:, -shift_amount:], signal[:, :-shift_amount]], dim=1)

        # Amplitude scaling augmentation
        if np.random.random() < self.config.AMPLITUDE_SCALE_PROB:
            scale_factor = np.random.uniform(*self.config.AMPLITUDE_SCALE_RANGE)
            signal = signal * scale_factor

        # Enhanced Gaussian noise
        if np.random.random() < self.config.GAUSSIAN_NOISE_PROB:
            noise = torch.randn_like(signal) * self.config.GAUSSIAN_NOISE_SCALE
            signal = signal + noise

        # Lead dropout (randomly zero out leads to improve robustness)
        # if np.random.random() < 0.1:  # 10% chance
        #     lead_idx = np.random.randint(0, int(0.05*signal.shape[0]))
        #     signal[lead_idx] = signal[lead_idx] * 0.0

        return signal

    def preprocess_signal(self, signal):
        """Enhanced preprocessing with better handling of ECG characteristics"""
        # Handle NaN values and outliers
        signal = torch.nan_to_num(signal, nan=0.0, posinf=3.0, neginf=-3.0)

        # Remove baseline wander (high-pass filter simulation)
        window_size = min(201, signal.shape[1] // 2)
        if window_size > 5:  # Only if the signal is long enough
            signal_mean = torch.nn.functional.avg_pool1d(
                signal.unsqueeze(0),
                kernel_size=window_size,
                stride=1,
                padding=window_size//2
            ).squeeze(0)
            signal = signal - signal_mean[:, :signal.shape[1]]

        # Normalize each lead using robust standardization
        for i in range(signal.shape[0]):
            # Use percentile-based normalization instead of mean/std
            # This helps with handling outliers and artifacts in ECG
            sorted_vals, _ = torch.sort(signal[i])
            q_low = sorted_vals[int(0.05 * len(sorted_vals))]
            q_high = sorted_vals[int(0.95 * len(sorted_vals))]
            iqr = q_high - q_low + 1e-6

            # Center around median instead of mean
            median = torch.median(signal[i])
            signal[i] = (signal[i] - median) / iqr

        # Clip values to prevent extreme outliers
        signal = torch.clamp(signal, -5.0, 5.0)

        # Handle signal length
        if signal.shape[1] > self.config.INPUT_LENGTH:
            # Center crop
            start = (signal.shape[1] - self.config.INPUT_LENGTH) // 2
            signal = signal[:, start:start + self.config.INPUT_LENGTH]
        elif signal.shape[1] < self.config.INPUT_LENGTH:
            # Zero-padding
            pad = torch.zeros((self.config.LEADS, self.config.INPUT_LENGTH - signal.shape[1]))
            signal = torch.cat((signal, pad), dim=1)

        return signal

    def __getitem__(self, idx):
      try:
          row = self.df.iloc[idx]
          file_path = row['file_name']
          record = wfdb.rdrecord(file_path)
          signal = torch.tensor(record.p_signal.T, dtype=torch.float32)

          # Create minimal preprocessing here on CPU
          # Handle NaN values and basic normalization only
          signal = torch.nan_to_num(signal, nan=0.0, posinf=3.0, neginf=-3.0)

          # Handle signal length (basic reshaping only on CPU)
          if signal.shape[1] > self.config.INPUT_LENGTH:
              start = (signal.shape[1] - self.config.INPUT_LENGTH) // 2
              signal = signal[:, start:start + self.config.INPUT_LENGTH]
          elif signal.shape[1] < self.config.INPUT_LENGTH:
              pad = torch.zeros((self.config.LEADS, self.config.INPUT_LENGTH - signal.shape[1]))
              signal = torch.cat((signal, pad), dim=1)

          # Convert labels
          label = torch.tensor(
              self.label_binarizer.transform([row['disease_categories']])[0],
              dtype=torch.float32
          )

          return signal, label
      except Exception as e:
          print(f"Error processing record {idx}: {e}")
          signal = torch.zeros((self.config.LEADS, self.config.INPUT_LENGTH))
          label = torch.zeros(len(self.label_binarizer.classes_))
          return signal, label
