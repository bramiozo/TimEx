import numpy as np
import scipy as sc
import pywt
from typing import List, Union, Literal, Generator, Tuple

from scipy.fft import fft, fftfreq
from scipy.signal import ShortTimeFFT, stft, ricker, cwt
from scipy.signal.windows import gaussian
from scipy.signal import spectrogram, get_window

import pandas as pd

import matplotlib.pyplot as plt

import os

ContinuousWavelets = ['gaus', 'morl', 'mexh']
DiscreteWavelets = ['dmey', 'rbio', 'bior', 'coif', 'sym', 'db']



'''
wavelet = pywt.ContinuousWavelet('gaus1')
scales = np.arange(1, 128)
wave_res = pywt.cwt(ts, scales=scales, wavelet = wavelet)

plt.imshow(wave_res[0], cmap='hot', interpolation='nearest')
plt.show()

'''

def make_spectroplot(times, scales, spectrogram, path):
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, scales, spectrogram, cmap='hot')  # Use times for x-axis  cmap='hot', interpolation='nearest'
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(f"Spectrogram for ID: {_id}")
    plt.savefig(f"{path}/{_id}_spectrogram_wavelet.png")
    plt.close()


class SingleChannelCWT():
    def __init__(self, 
                 wavelet_family:Literal['morl', 'gaus', 'mexh']='morl', 
                 wavelet_number: int=1, 
                 num_scales: Union[List[int], int]=128,
                 base_path: str="./Spectrograms",
                 write_spectrograms: bool=False):
        
        self.base_path = base_path
        if os.path.isdir(self.base_path) is False:
            os.mkdir(self.base_path)
            
        self.available_wavelet_numbers = pywt.wavelist(wavelet_family)
        
        if wavelet_family == 'gaus':
            wavelet_str = wavelet_family+str(wavelet_number)
        else:
            wavelet_str = wavelet_family
                            
        self.wavelet = pywt.ContinuousWavelet(wavelet_str)
        if isinstance(num_scales, int):
            self.scales = np.arange(1,num_scales)
        else:
            self.scales = num_scales
            
        self.write_spectrograms = write_spectrograms
            
    def _get_cwt(self, 
                 timeseries=List[np.ndarray])-> Generator:
        for ts in timeseries:
            wave_res, _ = pywt.cwt(ts, scales=self.scales, wavelet = self.wavelet)
            yield wave_res
            
    def process(self, 
                df,
                id_col: str='id', 
                val_col : str='value',
                time_col: str='dt')->List[Tuple[int, np.ndarray]]:
        ts_list = [df.loc[df[id_col] == _id, val_col].values 
                   for _id in df[id_col].unique()] 

        results = []
        for _id, spectrogram in zip(df[id_col].unique(), self._get_cwt(ts_list)):
            results.append(spectrogram)
            if self.write_spectrograms:
                # TODO: write to files with ID as filename.
                if self.write_spectrograms:
                    times = np.arange(0,spectrogram.shape[1])     
                    make_spectroplot(times, 
                                     self.scales, 
                                     spectrogram,
                                     path=f"{self.base_path}/{_id}_spectrogram_wavelet.png")
        return results
    
class SingleChannelFT:
    def __init__(self, 
                 nperseg: int = 8, 
                 nfreqs: int = 64,
                 noverlap: int = 4, 
                 window: str = 'gaussian', 
                 write_spectrograms: bool = False, 
                 sampling_rate: int = 2, 
                 base_path: str="./Spectrograms",
                 std_dev: float = 3.0):
        
        self.base_path = base_path
        if os.path.isdir(self.base_path) is False:
            os.mkdir(self.base_path)
        
        self.nperseg = nperseg
        self.nfreqs = nfreqs
        self.noverlap = noverlap if noverlap is not None else int(nperseg*0.75)
        self.window = get_window((window, std_dev), nperseg)
        self.write_spectrograms = write_spectrograms
        self.sampling_rate = sampling_rate

    def _get_stft(self, timeseries: List[np.ndarray]):
        for ts in timeseries:
            freqs, times, Zxx = stft(ts,
                                     fs=self.sampling_rate, 
                                     window='hann',
                                     nfft=self.nfreqs,
                                     nperseg=self.nperseg, 
                                     noverlap=self.noverlap)
            yield freqs, times, np.abs(Zxx)  # Yield frequencies and magnitude spectrogram

    def process(self, df: pd.DataFrame, id_col: str = 'id', val_col: str = 'value', time_col: str = 'dt'):
        ts_list = [df.loc[df[id_col] == _id, val_col].values 
                   for _id in df[id_col].unique()]

        results = []
        for _id, (freqs, times, spectrogram) in zip(df[id_col].unique(), self._get_stft(ts_list)):
            results.append((_id, freqs, spectrogram))
            if self.write_spectrograms:
                make_spectroplot(times, 
                                 self.scales, 
                                 spectrogram,
                                 path=f"{self.base_path}/{_id}_spectrogram_wavelet.png")
        return results
    
def SpectroGramEmbedder():
    def __init__(self, method: Literal['ravel', 'image_embedder']):
        pass

def wavelet(signal, function=ricker, widths=np.arange(1, 10)):
    """Computes CWT (continuous wavelet transform) of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    nd-array
        The result of the CWT along the time axis
        matrix with size (len(widths),len(signal))
    """

    if isinstance(function, str):
        function = eval(function)

    if isinstance(widths, str):
        widths = eval(widths)

    _cwt = cwt(signal, function, widths)

    return _cwt
def wavelet_entropy(signal, function=ricker, widths=np.arange(1, 10)):
    """Computes CWT entropy of the signal.

    Implementation details in:
    https://dsp.stackexchange.com/questions/13055/how-to-calculate-cwt-shannon-entropy
    B.F. Yan, A. Miyamoto, E. Bruhwiler, Wavelet transform-based modal parameter identification considering uncertainty

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    float
        wavelet entropy
    """
    if np.sum(signal) == 0:
        return 0.0

    cwt = wavelet(signal, function, widths)
    energy_scale = np.sum(np.abs(cwt), axis=1)
    t_energy = np.sum(energy_scale)
    prob = energy_scale / t_energy
    w_entropy = -np.sum(prob * np.log(prob))

    return w_entropy


def wavelet_abs_mean(signal, function=ricker, widths=np.arange(1, 10)):
    """Computes CWT absolute mean value of each wavelet scale.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT absolute mean value
    """
    res = tuple(np.abs(np.mean(wavelet(signal, function, widths), axis=1)))
    return {'WVL_amean_{k}': v for k, v in enumerate(res)}


def wavelet_std(signal, function=ricker, widths=np.arange(1, 10)):
    """Computes CWT std value of each wavelet scale.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT std
    """
    res = tuple(np.std(wavelet(signal, function, widths), axis=1))
    return {'WVL_std_{k}': v for k, v in enumerate(res)}


def wavelet_var(signal, function=ricker, widths=np.arange(1, 10)):
    """Computes CWT variance value of each wavelet scale.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT variance
    """
    res = tuple(np.var(wavelet(signal, function, widths), axis=1))
    return {'WVL_var_{k}': v for k, v in enumerate(res)}

def wavelet_energy(signal, function=ricker, widths=np.arange(1, 10)):
    """Computes CWT energy of each wavelet scale.

    Implementation details:
    https://stackoverflow.com/questions/37659422/energy-for-1-d-wavelet-in-python

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT energy
    """
    cwt = wavelet(signal, function, widths)
    res = tuple(np.sqrt(np.sum(cwt**2, axis=1) / np.shape(cwt)[1]))

    return {'WVL_energy_{k}': v for k, v in enumerate(res)}

def extract_wavelet_features(y, wavelet='db4', level=3, num_features=5):
    # Credits: https://towardsdatascience.com/feature-extraction-for-time-series-from-theory-to-practice-with-python-25631c6d8fcb
    y = y - np.mean(y)  # Remove the mean

    # Perform the Discrete Wavelet Transform
    coeffs = pywt.wavedec(y, wavelet, level=level)

    # Flatten the list of coefficients into a single array
    coeffs_flat = np.hstack(coeffs)

    # Get the absolute values of the coefficients
    coeffs_abs = np.abs(coeffs_flat)

    # Find the indices of the largest coefficients
    largest_coeff_indices = np.flip(np.argsort(coeffs_abs))[0:num_features]

    # Extract the largest coefficients as features
    top_coeffs = coeffs_flat[largest_coeff_indices]

    # Generate feature names for the wavelet features
    feature_keys = ['Wavelet Coeff ' + str(i+1) for i in range(num_features)]

    # Create a dictionary for the features
    wavelet_dict = {feature_keys[i]: top_coeffs[i] for i in range(num_features)}
    return wavelet_dict
