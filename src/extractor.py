import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import cwt, ricker
import matplotlib.pyplot as plt

class Extractor:
    def __init__(self):
        self.mean = None
        self.min = None
        self.max = None
        self.variance = None
        self.skewness = None
        self.kurtosis = None
        self.frequency_domain_features = None
        self.mann_kendall_stat = None
        self.wavelet_transform = None
    
    def fit(self, ts_data):
        self.mean = np.mean(ts_data)
        self.variance = np.var(ts_data)
        self.skewness = skew(ts_data)
        self.kurtosis = kurtosis(ts_data)
        self.min = np.min(ts_data)
        self.max = np.max(ts_data)
        
        N = len(ts_data)
        T = 1.0 / 800.0
        yf = rfft(ts_data)
        xf = rfftfreq(N, T)[:N//2]
        self.frequency_domain_features = (xf, 2.0/N * np.abs(yf[0:N//2]))
        
        self.mann_kendall_stat = self._mann_kendall_test(ts_data)
        
        # Perform Continuous Wavelet Transform
        self.wavelet_transform = self._wavelet_transform(ts_data)
    
    def _mann_kendall_test(self, data):
        n = len(data)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(data[j] - data[i])
        return s

    def _wavelet_transform(self, data):
        widths = np.arange(1, 31)  # Define the range of scales
        cwtmatr = cwt(data, ricker, widths)
        return cwtmatr
    
    def plot_wavelet_transform(self):
        if self.wavelet_transform is None:
            print("No wavelet transform available. Please run the fit method first.")
            return
        plt.imshow(self.wavelet_transform, extent=[0, 1, 1, 31], cmap='PRGn', aspect='auto',
                   vmax=abs(self.wavelet_transform).max(), vmin=-abs(self.wavelet_transform).max())
        plt.colorbar()
        plt.title("Wavelet Transform (CWT) of Time Series")
        plt.ylabel("Scale")
        plt.xlabel("Time")
        plt.show()

    def print_features(self):
        print(f"Mean: {self.mean}")
        print(f"Variance: {self.variance}")
        print(f"Skewness: {self.skewness}")
        print(f"Kurtosis: {self.kurtosis}")
        print(f"Mann-Kendall Test Statistic: {self.mann_kendall_stat}")


