from site import USER_BASE
# Complete implementation using neurokit2, scipy, catch22, tsfresh
from typing import Literal, Dict, List, Annotated, Optional, Sequence, Union, Tuple
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
import seaborn as sns
from matplotlib import pyplot as plt


# move all preprocessing logic like filtering, smoothing and detrending to ecg.preprocessor
from timex.ecg import preprocessor


from sklearn.preprocessing import label_binarize

# add logging
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Initializing Config class")

class Config:
    # Data settings
    LEADS = 12
    INPUT_LENGTH = 10_000
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    RANDOM_SEED = 42
    SAMPLING_RATE = 500

    MAX_OUTPUT_LENGTH = 5_000  # Maximum length of the output signal after re-sampling

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

# TODO: the pre-processing should be off-loaded to the ecg.preprocessor class
# TODO: add class weights as an attribute

def check_if_var_is_list_of_str_or_int(var: List[str|int])-> Tuple[bool,bool]:
    if not isinstance(var, list):
        raise TypeError(f"Expected a list, got {type(var)}")
    # Check if all items are the same type (all str or all int)
    if not var:  # empty list case
        return True

    first_type = type(var[0])
    if first_type not in (str, int):
        raise TypeError(f"Expected a list of strings or integers, got {first_type}")

    for item in var:
        if type(item) != first_type:
            raise TypeError(f"Expected a list of only {first_type.__name__}s, but found mixed types")

    list_of_strings = isinstance(var[0], str)
    return True, list_of_strings

class ECGDataset(Dataset):
    """
        ECGDataset class for PhysioNet files

        This class is focused on the ETL of ECG data.
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
        data: Union[List[str],List[Tuple[str,int]], str],
        data_type: Literal['folder_with_npy',
                           'folder_with_hea',
                           'folder_with_xml',
                           'folder_with_hdf5',
                           'list_of_filenames',
                           'list_of_filename_label_tuples',
                           'tsv_with_filenames_labels']='folder_with_hea',
        labels: Union[List[str|int],str, None]=None,
        label_binarizer: Optional[object] = None,
        is_train: bool=True,
        supervised: bool=False,
        extract_label: bool=True,
        extract_metadata: bool=True,
        config: Config|Dict|None=None,
        visualisation: bool=False,
        output_dir: str='',

        augmentations: List[Literal['gauss', 'shift', 'scale', 'dropout']]=['gauss', 'dropout'],
        preprocessing: List[Literal['nan', 'bandpass', 'savgol',
                                    'powerline', 'standardscaler', 'resampler', 'truncate', 'detrend']]
                                     = ['bandpass', 'resampler', 'detrend']
        ):

        if data_type not in ['folder_with_hea', 'folder_with_npy']:
            raise ValueError(f"Unsupported data type: {data_type}. Must be one of ['folder_with_hea', 'folder_with_npy']. Others options are work in progress.")

        if data_type in ['folder_with_npy', 'folder_with_hea']:
            if isinstance(data, list):
                # TODO: assert is list of strings
                _, list_of_str = check_if_var_is_list_of_str_or_int(data)
                assert(list_of_str), "Data must be a list of strings (filepaths to hea/npy) or a string (folder with hea/npy)"
                self.file_list = data
            elif isinstance(data, str) and os.path.isdir(data):
                self.file_list = []
                for root, _, files in os.walk(data):
                    for file in files:
                        if file.endswith(('.hea')):
                            self.file_list.append(os.path.join(root, file))

                        if file.endswith(('.npy')):
                            self.file_list.append(os.path.join(root, file))
            else:
                raise ValueError("Unsupported data source format." \
                " Must be DataFrame, list of paths, or folder.")

            print(f"First 5 elements of the file_list: {self.file_list[:5]}", flush=True)


        if supervised:
            if data_type == 'folder_with_hea':
                logging.info("Assuming folder with HEA files, assumption is that labels are part of HEA metadata")
                self.labels = labels
            elif data_type == 'folder_with_npy':
                logging.info("Assuming folder with NPY files, assumption is that labels are added seperately")
                if labels is None:
                    raise ValueError("Labels must be provided when using folder_with_npy data type")
                elif isinstance(labels, str):
                    # could be NPY file, could be a TXT file, could be a TSV
                    if labels.endswith('.npy'):
                        self.labels = list(np.load(labels))
                    elif labels.endswith('.txt'):
                        self.labels = list(np.loadtxt(labels, dtype=str))
                    elif labels.endswith('.tsv'):
                        data_labels = np.loadtxt(labels, dtype=str, delimiter='\t')
                        self.labels = list(data_labels[:, 1])
                        self.file_list = list(data_labels[:, 0])

                    assert(len(self.labels) == len(self.file_list)), f"Number of labels must match number of samples: {len(self.labels)}/{len(self.file_list)}"

                elif check_if_var_is_list_of_str_or_int(labels)[0]:
                    # list of str: assume the str's are labels, give warning about this
                    # list of int: assume the int's are labels, give warning about this
                    # assumption for both: ordering is the same as the list of samples
                    raise UserWarning("Assumption: You are passing a list of str/int as labels")

                    assert(len(labels) == len(self.file_list)), f"Number of labels must match number of samples: {len(self.labels)}/{len(self.file_list)}"

                    self.labels = labels

        if hasattr(self, 'labels') and self.labels is not None:
            unique_labels = list(set(self.labels)) if isinstance(self.labels, list) else None
            print(f"Unique labels are: {unique_labels}")
        else:
            print("No labels available to show unique values")

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
        self.visualisation = visualisation
        self.output_dir = output_dir

    def __len__(self):
        return len(self.file_list)

    def _load_signal_from_source(self, source):
        """
        Load ECG signal from different formats:
        - WFDB (.hea + .dat)
        - HDF5 (.h5)
        - NumPy (.npy)
        - TODO: XML(with.. lists?) see https://github.com/DFNOsorio/GEMuseXMLReader/blob/master/GEMuseXMLReader.py
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

    def compute_weights(self):
        #logger.info("Computing weights...")
        if len(self.diagnoses_cols) > 1:
            weights = []
            for label in self.diagnoses_cols:
                count = self.ecg_dataframe[label].sum()
                weight = (self.ecg_dataframe.__len__() - count) / (count + 1e-9)
                weights.append(weight)
        else:
            num_labels = self.ecg_dataframe[self.diagnoses_cols[0]].max() + 1
            weights = num_labels / self.ecg_dataframe[self.diagnoses_cols].value_counts()
            weights = weights.values.tolist()
        #logger.info("Done with the weights.")
        return torch.FloatTensor(weights)


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
        if signal.shape[1] > self.config.MAX_OUTPUT_LENGTH:
            # Center crop
            start = (signal.shape[1] - self.config.MAX_OUTPUT_LENGTH)
            signal = signal[:, start:start + self.config.MAX_OUTPUT_LENGTH]
        elif signal.shape[1] < self.config.MAX_OUTPUT_LENGTH:
            # Zero-padding
            pad = torch.zeros((self.config.LEADS, self.config.MAX_OUTPUT_LENGTH - signal.shape[1]))
            signal = torch.cat((signal, pad), dim=1)
        return signal

    def _detrend_signal(self, signal):
        """
        Remove linear trend from each lead (channel).

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
    
    @staticmethod
    def visualize(ts: np.ndarray, file_name: str="ecg"):
        # plot in one figure
        max_rows = ts.shape[0]//2+1
        fig, ax = plt.subplots(ncols=2, nrows=max_rows, figsize=(18,20))

        for k, _ts in enumerate(ts):
            i = k // 2
            j = k %  2
            ax[i,j].plot(ts[k, :])
            ax[i,j].set_title(f"Lead {k}")
        
        fig.savefig(f'{file_name}.pdf', dpi=300)
        
    def compute_attention_mask_for_padding(self, array):
            # credits:https://github.com/Edoar-do/HuBERT-ECG/blob/master/code/dataset.py
            array = array.reshape(12, -1)     # 12 x SAMPLES_IN_5_SECONDS_AT_500HZ
            for index in range(array.shape[1]):
                if np.any(array[:, index]):
                    break
            start = index
            for index in range(array.shape[1]-1, -1, -1):
                if np.any(array[:, index]):
                    break
            end = index
            attention_mask = np.zeros(array.shape[1])
            attention_mask[start:end+1] = 1
            attention_mask = np.repeat([attention_mask], 12, axis=0)
            attention_mask = np.concatenate(attention_mask, axis=0)
            return attention_mask

    def compute_beat_based_attention_mask(self, ecg_data):
        '''
        Computes attention mask focusing only on P wave, QRS complex and T wave
        Credits:https://github.com/Edoar-do/HuBERT-ECG/blob/master/code/dataset.py
        '''

        ecg_data = ecg_data.reshape(12, self.config.MAX_OUTPUT_LENGTH)
        _, rpeaks = nk.ecg_peaks(ecg_data[1], sampling_rate=500) #compute R peaks from II
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_data[1], rpeaks, sampling_rate=500, method="dwt", show=False, show_type='all')
        signal_dwt['ECG_R_Peaks'] = 0
        signal_dwt['ECG_R_Peaks'].iloc[rpeaks['ECG_R_Peaks']] = 1

        p_wave = signal_dwt['ECG_P_Onsets'] | signal_dwt['ECG_P_Offsets'] # binary serie with 1 where P waves start and stop
        qrs_complex = signal_dwt['ECG_Q_Peaks'] | signal_dwt['ECG_S_Peaks'] # binary serie with 1 where QRS complexes start and stop
        t_wave = signal_dwt['ECG_T_Onsets'] | signal_dwt['ECG_T_Offsets'] # binary serie with 1s where T waves start and stop

        p_starts_stops = p_wave[p_wave != 0].index.tolist()
        if len(p_starts_stops) % 2 != 0:
            p_starts_stops.append(min(p_starts_stops[-1]+1, 2499))
        p_starts_stops = np.array(p_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each P wave detected

        t_starts_stops = t_wave[t_wave != 0].index.tolist()
        if len(t_starts_stops) % 2 != 0:
            t_starts_stops.append(min(t_starts_stops[-1]+1, 2499))
        t_starts_stops = np.array(t_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each T wave detected


        qrs_starts_stops = qrs_complex[qrs_complex != 0].index.tolist()
        if len(qrs_starts_stops) % 2 != 0:
            qrs_starts_stops.append(min(qrs_starts_stops[-1]+1, 2499))
        qrs_starts_stops = np.array(qrs_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each QRS complex detected

        # building the attention mask in order to attend only samples in the p waves
        for start, stop in p_starts_stops:
            p_wave.iloc[start : stop] = 1

        # building the attention mask in order to attend only samples in the t waves
        for start, stop in t_starts_stops:
            t_wave.iloc[start : stop] = 1

        # building the attention mask in order to attend only samples in the qrs complexes
        for start, stop in qrs_starts_stops:
            qrs_complex.iloc[start : stop] = 1

        # global attention mask merging all interest regions
        attention_mask = (p_wave | t_wave | qrs_complex).tolist()
        attention_mask = np.repeat([attention_mask], 12, axis=0) # since the leads are temporally aligned, interest regions should be located within the same intervals
        attention_mask = np.concatenate(attention_mask, axis=0)

        return attention_mask


    def __getitem__(self, idx:int)-> tuple:
        try:
            file_path = self.file_list[idx]
            signal, record = self._load_signal_from_source(file_path)
            label = None # not correct if labels are provided
            metadata = None

            if self.visualisation:
                print(f"Visualize ts data: {signal.shape}")
                self.visualize(np.array(signal), file_name=os.path.join(self.output_dir, f'pre_{idx}_ecg.pdf'))


            if len(self.preprocessing) > 0:
                signal = self.preprocess_signal(signal)

            if len(self.augmentations) > 0:
                signal = self.augment_signal(signal)

            # Ensure signal has the correct shape
            signal = signal[:self.config.LEADS, :self.config.MAX_OUTPUT_LENGTH]

            if self.visualisation:
                self.visualize(np.array(signal),  file_name=os.path.join(self.output_dir, f'post_{idx}_ecg.pdf'))

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
            return signal, file_path, idx, label, metadata
        except Exception as e:
            raise ValueError(f"Error processing record {idx}: {e}")


if __name__ == "__main__":
    # Example usage
    dataset = ECGDataset(
        config=Config(),
        data_dir="path/to/data",
        label_binarizer=LabelBinarizer(),
        supervised=True,
        extract_label=True
    )
    print(dataset[0])
