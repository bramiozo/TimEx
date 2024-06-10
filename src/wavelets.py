import numpy as np
import scipy as sc
import pywt
from typing import List, Union, Literal, Generator, Tuple

from scipy.fft import fft, fftfreq
from scipy.signal import ShortTimeFFT, stft
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
     
    