from scipy.signal import find_peaks
import numpy as np
from scipy.signal import butter, lfilter, freqz
from meegkit.detrend import detrend as meeg_detrend, regress
from statsmodels.tsa import tsatools, seasonal
from typing import Literal
from sklearn.decomposition import FastICA

class Trimmer:
    def __init__(self,
                 threshold_quantile: float = 0.99,
                 num_inner_peaks: int = 8,
                 min_peak_distance: int = 100,
                 prominence: float = 0.,
                 how_to_trim: str = 'first'):
        '''
        Constructor for Trimmer class

        Assume index column is numeric and represent timesteps with standard interval 1/100 sec
        '''
        self.threshold_quantile = threshold_quantile
        self.num_inner_peaks = num_inner_peaks
        self.min_peak_distance = min_peak_distance
        self.prominence = prominence
        self.how_to_trime = how_to_trim

    def _find_threshold(self):
        self.threshold = np.quantile(self.data, self.threshold_quantile)
        return self

    def _find_peaks(self):
        peaks, _ = find_peaks(self.data,
                              height=self.threshold,
                              distance=self.min_peak_distance,
                              prominence=self.prominence
                              )

        self.num_peaks = len(peaks)
        self.peaks_indcs = self.vindcs[peaks]
        return self

    def _get_inner_peaks(self):
        self.inner_peaks_indcs = self.peaks_indcs[self.num_peaks//2-self.inner_peaks_num//2:
                                                  self.num_peaks//2+self.inner_peaks_num//2]
        return self

    def _find_range(self):
        start_inner = min(self.inner_peaks_indcs),
        stop_inner = max(self.inner_peaks_indcs)

        start_outer = min(self.peaks_indcs)
        stop_outer = max(self.peaks_indcs)

        return start_inner, start_outer, stop_inner, stop_outer

    # np.array with multiple columns
    def get_boundaries(self, data: np.array = None, indcs: np.array = None):
        self.data = data
        self.vindcs = indcs
        self._find_threshold()
        self._find_peaks()
        self._get_inner_peaks()
        self.start_inner, self.start_outer, self.stop_inner, self.stop_outer = self._find_range()
        return self


'''
 Class to perform low-pass filtering on the data
'''
# Bandpass filtering
# filter out high frequency noise
# now we want to apply a bandpass filter per channel (a column from the meas_columns list) and per segment


def _filter_segment(df,
                    group_cols: list = ['patient', 'segment'],
                    meas_columns: list = [
                        'I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'LA', 'RA', 'LL'],
                    sampling_frequency=500,
                    cutoff_freq=50,
                    order=4):
    b, a = butter(order, cutoff_freq, btype='low',
                  analog=False, fs=sampling_frequency)

    def _lfilter(x):
        return lfilter(b, a, x)

    for col in meas_columns:
        df[col] = df.groupby(group_cols)[col].transform(_lfilter)
    return df



def DeTrender(ts: np.ndarray,
              how: Literal=['poly', 'MSTL', 'meeg'],
              order=1, 
              period=100, 
              windows=101, 
              cutsize=250,
              median_window=25,
              iterate=5):
    if how == 'poly':
        return tsatools.detrend(ts, order=order)
    elif how == 'poly_np':
        z = np.polyfit(np.arange(0,ts.shape[0],1), ts, order)
        y_poly = np.poly1d(z)
        return ts-y_poly(np.arange(0,ts.shape[0],1))
    elif how == 'MSTL':
        detrender= seasonal.MSTL(ts, periods=period, windows=windows, iterate=iterate).fit()
        return detrender.seasonal
    elif how == 'meeg':
        '''
         Use sinusoids as basis functions:
          This will suffer from the Gibbs-phenomenon. 
          To mitigate this, we have to cut the head/tail
        '''
        _ts = np.array(ts).astype(float)
        return meeg_detrend(_ts, order=order, basis="sinusoids")[0][cutsize:-cutsize] 
    elif how == 'sliding_median':
        '''
         Make base plot using sliding mean, subtract from raw 
        '''
        sl= np.median(np.lib.stride_tricks.sliding_window_view(ts, (median_window,)), axis=1)
        return ts[median_window-1:] - sl
    else:
        raise ValueError(f"For now we only accept ['poly', 'MSTL', 'meeg]")
    

class ButterworthBandpassFilter:
    def __init__(self, lowcut, highcut, fs, order=5):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.b, self.a = self._design_filter()

    def _design_filter(self):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        return signal.butter(self.order, [low, high], btype='band')

    def apply_filter(self, data):
        return signal.lfilter(self.b, self.a, data)


class ICASignalSeparator:
    def __init__(self, n_components=None, max_iter=250):
        self.n_components = n_components
        self.ica = FastICA(n_components=n_components, random_state=42, max_iter=max_iter)
        
    def separate(self, mixed_signals):
        """
        Separate mixed signals using ICA.
        
        :param mixed_signals: 2D array, shape (n_samples, n_features)
        :return: 2D array of separated signals
        """
        return self.ica.fit_transform(mixed_signals.T).T
    
    def plot_signals(self, original_signals, mixed_signals, separated_signals):
        n_signals = original_signals.shape[0]
        time = np.linspace(0, 10, original_signals.shape[1])
        
        fig, axs = plt.subplots(3, n_signals, figsize=(15, 10))
        fig.suptitle('Original, Mixed, and Separated Signals')
        
        for i in range(n_signals):
            axs[0, i].plot(time, original_signals[i])
            axs[0, i].set_title(f'Original Signal {i+1}')
            
            axs[1, i].plot(time, mixed_signals[i])
            axs[1, i].set_title(f'Mixed Signal {i+1}')
            
            axs[2, i].plot(time, separated_signals[i])
            axs[2, i].set_title(f'Separated Signal {i+1}')
        
        for ax in axs.flat:
            ax.set(xlabel='Time', ylabel='Amplitude')
        
        plt.tight_layout()
        plt.show()