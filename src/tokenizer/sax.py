import wfdb
import tslearn
import pandas as pd
import numpy as np
import math
import json
import scipy as sc
from scipy import signal
import neurokit2 as nk
import matplotlib.pyplot as plt
from tslearn.piecewise import SymbolicAggregateApproximation,OneD_SymbolicAggregateApproximation
from scipy.signal import find_peaks, butter, filtfilt, detrend, savgol_filter, sosfilt
from pathlib import Path
import importlib
from sklearn import preprocessing
from preprocessor import ECGSignalProcessor
import pickle


class ECGTokenizer:
    def __init__(self, n_segments_freq=5, alphabet_size=500, min_value=-2, max_value=2):
        # ASSUMES THAT SAMPLING RATE IS 250Hz!!!!!!!
        
        self.n_segments_freq = n_segments_freq
        self.alphabet_size = alphabet_size
        self.toks_sax = None
        self.toks_sax_inv = None
        self.toks_ld = None
        self.toks_ld_inv = None
        self.min_value = min_value
        self.max_value = max_value
        self.n_segments = self.n_segments_freq*data.shape[2]//250

    def tokenize_sax(self, data):
        # TODO: move data loading to function after init
        his_dat = []
        his_inv_dat = []
        for sig_group in data:
            for sig in sig_group:
                signal1 = sig.reshape(1, -1)
                sax = SymbolicAggregateApproximation(n_segments=self.n_segments, alphabet_size_avg=self.alphabet_size)
                sax_data = sax.fit_transform(signal1)
                his_dat.append(sax_data.reshape(self.n_segments,))
                sax_inv = sax.inverse_transform(sax_data).reshape(-1)
                his_inv_dat.append(sax_inv)

        self.toks_sax = np.array(his_dat)
        self.toks_sax_inv = his_inv_dat
        return self.toks_sax, self.toks_sax_inv

    def tokenize_1d_sax(self, data):
        # TODO: move data loading to function after init
        # TODO: 1d SAX not working, or slow
        his_dat = []
        his_inv_dat = []
        for sig_group in data:
          for sig in sig_group:
              signal1 = sig.reshape(1, -1)
              ld_sax = OneD_SymbolicAggregateApproximation(n_segments=self.n_segments, alphabet_size_avg=self.alphabet_size, alphabet_size_slope=self.alphabet_size)
              ld_sax_data = ld_sax.fit_transform(signal1)
              his_dat.append(ld_sax_data.reshape(self.n_segments * 2,))
              ld_sax_inv = ld_sax.inverse_transform(ld_sax_data).reshape(len(ld_sax.inverse_transform(ld_sax_data)[0]))
              his_inv_dat.append(ld_sax_inv)

        self.toks_ld = np.array(his_dat)
        self.toks_ld_inv = his_inv_dat
        return self.toks_ld, self.toks_ld_inv


    def clean_ld_tokens(self):
        cleaned = []
        for seq in self.toks_ld_inv:
          seq = np.nan_to_num(seq)  # Remove NaNs
          seq = np.where((-self.min_value <= seq) & (seq <= self.max_value), seq, 0)  # Clamp extreme values
          cleaned.append(seq)
        self.toks_ld_inv = cleaned

    def plot_histograms(self):
        if self.toks_sax is not None:
          plt.hist(self.toks_sax.reshape(-1))
          plt.title("SAX Histogram")
          plt.show()


        if self.toks_ld is not None:
          plt.hist(self.toks_ld.reshape(-1))
          plt.title("1d-SAX Histogram")
          plt.show()

    def save_tokens(self, toks_sax, toks_ld, filename="tokens.json"):
        if self.toks_sax is None or self.toks_ld is None:
          raise ValueError("Tokenization must be done before saving.")
        tokens = {"toks_sax": toks_sax.tolist(), "toks_ld": toks_ld.tolist()}
        with open(filename, "w") as f:
          json.dump(tokens, f)

    def save_model(self, save_directory: str="sax_model.pkl"):
        with open(save_directory, 'wb') as file:
            file.write(pickle.dumps(self))
