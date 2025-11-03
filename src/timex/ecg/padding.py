import numpy as np
def padding(signal, target_length):  #To be used in a loop with file names. Signal has length 12.
    signal = np.asarray(signal)
    padded_signal = np.zeros((len(signal), target_length)) #Zero array with shape(12,5000)
    for i in range(len(signal)):
      current_length = len(signal[i])
      if current_length >= target_length:
        padded_signal[i] = signal[i][:target_length]

      else:
        pad_len = target_length - current_length
        padded_signal[i] = np.pad(signal[i], (0, pad_len), mode='wrap')
    return padded_signal