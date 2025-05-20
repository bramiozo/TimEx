

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
from sklearn import preprocessing


filepaths=['drive/MyDrive/data_ecg/102511170/102511170', 'drive/MyDrive/data_ecg/100780919/100780919', 'drive/MyDrive/data_ecg/102144047/102144047', 'drive/MyDrive/data_ecg/102147240/102147240', 'drive/MyDrive/data_ecg/102172660/102172660',
            'drive/MyDrive/data_ecg/102241375/102241375', 'drive/MyDrive/data_ecg/102616671/102616671', 'drive/MyDrive/data_ecg/103036945/103036945', 'drive/MyDrive/data_ecg/105362569/105362569', 'drive/MyDrive/data_ecg/107143276/107143276' ]

class ECGSignalProcessor:
    def __init__(self, p_signal, n_sig, sig_len, sig_name, x, fs=500):


        self.tokenizer=[]
        self.fs = fs
        self.n_sig = n_sig
        self.sig_len = sig_len
        self.sig_name = sig_name
        self.p_signal = p_signal
        self.x = x
        self.filtered_signal = None
        self.filtered_signal2 = None
        self.smoothed = None
        self.detrend = None
        self.trim_arr = None
        self.peaks = None
        self.dips = None
        self.hist = None






    def plot_signals(self):
        fig, axes = plt.subplots(nrows=self.n_sig, ncols=1, figsize=(18, 35))
        for i in range(self.n_sig):
            axes[i].plot(self.x, self.p_signal[i])
            axes[i].set_title(self.sig_name[i])
        plt.show()

    def detect_peaks_and_dips(self):

        for i in range(self.n_sig):
          self.peaks, _ = find_peaks(self.p_signal[i], distance=30, prominence=(0.1))
          self.dips, _ = find_peaks(-self.p_signal[i], distance=250)


    def apply_bandpass_filter(self, lowcut=0.5, highcut=40.0, order=4):
        nyq = 0.5 * self.fs
        sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
        self.filtered_signal= np.array([sosfilt(sos, sig) for sig in self.p_signal], dtype='float32')
        return self.filtered_signal




    def apply_notch_filter(self, filtered, freq=60.0, bandwidth=1.0, order=4):
        nyq = 0.5 * self.fs
        b, a = butter(order, [(freq - bandwidth) / nyq, (freq + bandwidth) / nyq], btype='bandstop')
        self.filtered_signal2= np.array([filtfilt(b, a, sig) for sig in filtered], dtype='float32')
        return self.filtered_signal2




    def apply_savgol_filter(self, filtered, window_length=31, polyorder=3):
        #for i in range(self.n_sig):
          self.smoothed= np.array([savgol_filter(sig, window_length, polyorder) for sig in filtered], dtype='float32')
          t = np.linspace(0, 10, len(self.smoothed))
          return self.smoothed



    def apply_detrend(self, filtered):
        #for i in range(self.n_sig):
          self.detrend=np.array([detrend(sig, type='linear') for sig in filtered], dtype='float32')
          t = np.linspace(0, 10, len(self.detrend))




    def trim_signal(self, filtered):
        trim_signals = []
        trim_times = []
        for i in range(self.n_sig):
          peaks, _ = find_peaks(filtered[i], distance=30, prominence=(0.1))
          dips, _ = find_peaks(-filtered[i], distance=250)
          start_point = min(peaks[0], dips[0])
          end_point = max(peaks[-1], dips[-1])
          trimmed = filtered[i][start_point:end_point]
          time_trimmed = np.linspace(self.x[start_point], self.x[end_point], len(trimmed))
          trim_signals.append(trimmed)
          trim_times.append(time_trimmed)

#          plt.figure(figsize=(15, 5))
#          plt.plot(self.x[peaks], filtered[i][peaks], 'o', color='red')
#          plt.plot(self.x[dips], filtered[i][dips], 'o', color='green')
#          plt.plot(self.x, filtered[i])
#          plt.plot(time_trimmed, trimmed, color='orange')
#          plt.title('Trimmed Signal')
#          plt.show()
          self.trim_arr=np.array(trim_signals, dtype='object')
          self.trim_time_arr= np.array(trim_times, dtype='object')
        return self.trim_arr



    def quality_check(self, trimmed):
        for sig in trimmed:
          quality = nk.ecg_quality(sig, method= 'zhao2018', approach= 'fuzzy', sampling_rate=self.fs)
          print(quality)


p_signals=[]
n_sig=[]
sig_len=[]
sig_name=[]



for fname in filepaths:
    records = wfdb.rdrecord(fname)
    p_signals.append(np.transpose(records.p_signal))
    p_signal=np.array(p_signals)
    n_sig.append(records.n_sig)
    sig_len.append(records.sig_len)
    sig_name.append(records.sig_name)
    x=np.linspace(0, 10, records.sig_len)

all_data=[]
for i in range(len(filepaths)):
      processor=ECGSignalProcessor(p_signal[i], records.n_sig, records.sig_len, records.sig_name, x)
#      processor.plot_signals()
      processor.detect_peaks_and_dips()
      processor.apply_bandpass_filter()
      processor.apply_notch_filter(processor.filtered_signal)
      processor.apply_savgol_filter(processor.filtered_signal)
      processor.apply_detrend(processor.smoothed)
      processor.trim_signal(processor.detrend)
      #processor.quality_check(processor.trim_arr)
      all_data.append(processor.trim_arr)




def tokenizer_(data):
  his_dat=[]
  his_inv_dat=[]

  for i in range(len(data[0])):
    for sig in data[0]:

      signal1=sig.reshape(1,-1)
      sax = SymbolicAggregateApproximation(n_segments=1000, alphabet_size_avg=500)
      sax_data = sax.fit_transform(signal1)
      sax_d=sax_data.reshape(1000,)
      his_dat.append(sax_d)
      hist=np.array(his_dat)

      sax_inv=sax.inverse_transform(sax_data)
      sax_inv_d=sax_inv.reshape(-1)
      his_inv_dat.append(sax_inv_d)


  return hist, his_inv_dat

def tokenizer_ld(data):
    his_dat=[]
    his_inv_dat=[]

    for i in range(len(data[0])):
      for sig in data[0]:

        signal1=sig.reshape(1,-1)
        ld_sax = OneD_SymbolicAggregateApproximation(n_segments=1000, alphabet_size_avg=500, alphabet_size_slope=500)
        ld_sax_data = ld_sax.fit_transform(signal1)
        ld_sax_d=ld_sax_data.reshape(2000,)
        his_dat.append(ld_sax_d)
        hist=np.array(his_dat)

        ld_sax_inv=ld_sax.inverse_transform(ld_sax_data)
        ld_sax_inv_d=ld_sax_inv.reshape(len(ld_sax_inv[0]))
        his_inv_dat.append(ld_sax_inv_d)
       # hist_inv=np.array(his_inv_dat)
    return hist, his_inv_dat





toks_sax, toks_sax_inv=tokenizer_(all_data)
toks_sax_new=toks_sax.reshape(-1)
plt.hist(toks_sax_new)
plt.show()



toks_ld, toks_ld_inv=tokenizer_ld(all_data)

#Remove nan values
for i in range(len(toks_ld_inv)):
    for j in range(len(toks_ld_inv[i])):
        if toks_ld_inv[i][j] == float("nan"):
            toks_ld_inv.remove([i][j])



#Remove extremely small and large values
for i in range(len(toks_ld_inv)):
    for j in range(len(toks_ld_inv[i])):
        if not (-2 <= toks_ld_inv[i][j] <= 2)  :
            toks_ld_inv[i][j]=0


toks_ld_new=toks_ld.reshape(-1)
plt.hist(toks_ld_new)
plt.show()


#Create .json

tokens={"toks_sax" : toks_sax.tolist(), "toks_ld" : toks_ld.tolist()}
with open("tokens.json", "w") as json_file:
    json.dump(tokens, json_file)
