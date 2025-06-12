import numpy as np
import json



class ECG_token_ts:
    def __init__(self, filepaths, record_name, n_segments, n_sig, sig_len):

        
        self.filepaths = filepaths
        self.record_name = record_name
        self.n_segments = n_segments
        self.n_sig = n_sig
        self.sig_len = sig_len
        self.segm = None
        self.toks = None
        self.ecg_signals = []
        


    def segment_signals(self, data, toks_sax):
        self.toks=toks_sax.reshape(len(self.filepaths), self.n_sig[0], self.n_segments) 
        segments = []
        for i in range(len(data)):
            for j in range(len(data[0])):
                segments.append(np.split(data[i][j], self.n_segments))

        self.segm = np.array(segments).reshape(
            len(self.filepaths),
            self.n_sig[0],
            self.n_segments,
            int(self.sig_len[0]) // self.n_segments
        )
        return self.segm, self.toks

    def generate_ecg_dict(self):
        for i in range(len(self.filepaths)):
            record_dict = {
                "record_name": self.record_name[i],
                "tokens": {}
            }

            for j in range(self.n_sig[0]):
                for k in range(self.n_segments):
                    token = str(self.toks[i][j][k])
                    time_series = self.segm[i][j][k].tolist()
                    token_key = f"lead{j}-segm{k}"

                    record_dict["tokens"][token_key] = {
                        "token": token,
                        "time_series": time_series
                    }

            self.ecg_signals.append(record_dict)
        return ecg_signals    

    def save_to_json(self, filename="new_ecg.json"):
        with open(filename, "w") as f:
            json.dump(self.ecg_signals, f)

  
