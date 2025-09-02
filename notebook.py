import numpy as np
import pickle
from tslearn.piecewise import SymbolicAggregateApproximation

import tok
import ECGSignalPreprocessor_new


all_data = np.memmap("all_data_memmap.npy", dtype = "float32", mode = "r", shape = (540, 12, 5000))
ecg_sentence = np.memmap("ecg_sentence.npy", dtype = "float32", mode = "r")

print(all_data.shape)

sax = SymbolicAggregateApproximation(n_segments = 100, alphabet_size_avg = 300)
sax_list = sax.fit(ecg_sentence.reshape(-1,1))
my_tokenizer = tok.Word_Tokenizer(SAX_List = sax_list)
encodings = my_tokenizer.encode(all_data, symbolizer_model = "sax")


with open("encodings_small.pkl", "wb") as f:
    pickle.dump(encodings, f)


