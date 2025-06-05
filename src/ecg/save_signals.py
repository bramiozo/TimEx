import json
import numpy as np

class save_signals:
    def __init__(self, filepaths, record_name, sig_name,
                 toks_sax, toks_sax_inv, toks_ld, toks_ld_inv,
                 n_sig, sig_len, tokenizer):
        self.filepaths = filepaths
        self.record_name = record_name
        self.sig_name = sig_name
        self.n_sig = n_sig
        self.sig_len = sig_len
        self.tokenizer = tokenizer

        # Reshape and convert arrays
        self.toks_sax = np.asarray(toks_sax).reshape(len(filepaths), self.n_sig[0], tokenizer.n_segments)
        self.toks_sax_inv = np.asarray(toks_sax_inv, dtype="object").reshape(len(filepaths), self.n_sig[0])
        self.toks_ld = np.asarray(toks_ld).reshape(len(filepaths), self.n_sig[0], 2 * tokenizer.n_segments)
        self.toks_ld_inv = np.asarray(toks_ld_inv, dtype="object").reshape(len(filepaths), self.n_sig[0])

        self.ecg_signals = []

    def prepare_signals(self):
        self.ecg_signals = []
        for i in range(len(self.filepaths)):
            for j in range(self.n_sig[0]):
                signals = {
                    "id": self.record_name[i],
                    "channel": {
                        self.sig_name[i][j]: {
                            "tokens_sax": self.toks_sax[i][j].tolist(),
                            "tokens_sax_inv": self.toks_sax_inv[i][j].tolist(),
                            "tokens_1dsax": self.toks_ld[i][j].tolist(),
                            "tokens_1dsax_inv": self.toks_ld_inv[i][j].tolist(),
                        }
                    },
                    "n_sig": self.n_sig[i],
                    "sig_len": self.sig_len[i]
                }
                self.ecg_signals.append(signals)

    def save_to_json(self, filename="ecg_signals.json"):
        if not self.ecg_signals:
            self.prepare_signals()
        with open(filename, "w") as f:
            json.dump(self.ecg_signals, f, indent=1)