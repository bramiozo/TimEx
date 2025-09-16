from tokenizers import Tokenizer, models, pre_tokenizers, decoders, AddedToken
from transformers import PreTrainedTokenizer
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation, PiecewiseAggregateApproximation
from typing import Type, List
import numpy as np
import math
import torch
import warnings
import neurokit2 as nk
from scipy.signal import find_peaks

class ECGTokenizer(PreTrainedTokenizer):

    model_input_names = ["input_ids", "attention_mask"]
    prefix_tokens: list[int] = []

    def __init__(
        self,
        SAX_List: Type[OneD_SymbolicAggregateApproximation] | Type[SymbolicAggregateApproximation] | PiecewiseAggregateApproximation | None = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
        cls_token: str = "[CLS]",
        beat_token: str = "[BEAT]",
        add_beat_mask: bool=False,
        **kwargs,
    ):
        # TODO: implement beat_based_attention_mask
        assert (
            isinstance(SAX_List, OneD_SymbolicAggregateApproximation)
            or isinstance(SAX_List, SymbolicAggregateApproximation) 
            or isinstance(SAX_List, PiecewiseAggregateApproximation)
        )

        self.SAX_List = SAX_List

        if isinstance(SAX_List, OneD_SymbolicAggregateApproximation):
            self.SAX_List._is_fitted()
            self.symbolizer_model = '1dSAX'
            vocab_size = SAX_List.alphabet_size_avg + SAX_List.alphabet_size_slope
        elif isinstance(SAX_List, SymbolicAggregateApproximation):
            self.SAX_List._is_fitted()
            self.symbolizer_model = 'SAX'
            vocab_size = SAX_List.alphabet_size_avg
        elif isintance(SAX_List, PiecewiseAggregateApproximation):
            self.SAX_List._is_fitted()
            self.symbolizer_model = 'PAA'
            vocab_size = SAX_List.alphabet_size_avg

        
        # Special tokens
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        beat_token = AddedToken(beat_token, lstrip=True, rstrip=False) if isinstance(beat_token, str) else beat_token
        # Build vocab: numbers 1..vocab_size plus specials
        self.vocab = {str(i): i + 7 for i in range(0, vocab_size)}
        self.vocab.update({
            "<unk>": 3,
            "<s>": 1,
            "</s>": 2,
            "<pad>": 0,
            "[CLS]": 4,
            "[SEP]": 5,
            "[MASK]": 6,
            "[BEAT]": 7
        })
        self._vocab_size = len(self.vocab)

        # Hugging Face tokenizer-core
        self.tokenizer = Tokenizer(models.WordLevel(self.vocab, unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.decoder = decoders.WordPiece()

        self.toks_sax = None
        self.toks_sax_inv = None
        self.toks_1d = None
        self.toks_1d_inv = None
        self.n_segments = self.SAX_List.n_segments
        self.alphabet_size = self.SAX_List.alphabet_size_avg
        #self.max_length = 12*(self.n_segments)
        self.target_length = 5000

        self.add_beat_mask = add_beat_mask
        self.beat_token = "[BEAT]"
        self.beat_token_id = 7

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            mask_token=mask_token,
            cls_token=cls_token,
            **kwargs,
        )

    # ---- Hugging Face required methods ----
    def vocab_size(self) -> int:
        assert(self.SAX_List._is_fitted()),  "The Symbolizer is not fitted yet!"
        return self._vocab_size

    def get_vocab(self):
        assert(self.SAX_List._is_fitted()),  "The Symbolizer is not fitted yet!"
        return self.vocab

    def _tokenize(self, text: str) -> List[str]:
        assert(self.SAX_List._is_fitted()),  "The Symbolizer is not fitted yet!"
        encoding = self.tokenizer.encode(text)
        return encoding.tokens

    def _convert_token_to_id(self, token: str) -> int:
        assert(self.SAX_List._is_fitted()),  "The Symbolizer is not fitted yet!"
        token_id = self.tokenizer.token_to_id(token)
        return token_id if token_id is not None else self.tokenizer.token_to_id(self.unk_token.content)

    def _convert_id_to_token(self, index: int) -> str:
        assert(self.SAX_List._is_fitted()),  "The Symbolizer is not fitted yet!"
        return self.tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        assert(self.SAX_List._is_fitted()),  "The Symbolizer is not fitted yet!"
        return " ".join(tokens)

    def _decode(self, token_ids, skip_special_tokens: bool = True, **kwargs):
        special_token_ids = {self.vocab[tok] for tok in ["<unk>", "<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[MASK]", "[BEAT]"] if tok in self.vocab}
        
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        #Delete special token ids
        decoded = []
        for sequence in token_ids:  
            filtered_ids = [int(tok_id) for tok_id in sequence
                 if not (skip_special_tokens and int(tok_id) in special_token_ids)]
            decoded.append(filtered_ids)
            
        #Decode sax/1dsax
        sax_decoded = []
        for i in range(len(decoded)):
            sax_decoded.append(self.SAX_List.inverse_transform(np.array(decoded[i]).reshape(-1, 1)).reshape(-1))
            
        
        return sax_decoder # np.array(sax_decoded).reshape(len(sax_decoded),12, )
       

    def build_inputs_with_special_tokens(self, data):
        #Insert [SEP] after every self.n_segment tokens
        #Insert </s> after every 12*self.n_segment tokens
        #Single sequence: `<s> X </s>`
        #- pair of sequences: `<s> A [SEP] B </s>`

        #tokens = data.reshape(len(data), len(data[0])*len(data[0][0]))
        tokens = np.array(data).flatten()

        bos = self.bos_token
        sep = self.sep_token
        eos = self.eos_token

        new_seq = []  # Start with <s>
        new_seq.append(bos)
        for i, token in enumerate(tokens, 1):
            token_str = str(token)


            if token_str == str(self.pad_token_id):
                new_seq.append(self.pad_token)
            else:
                new_seq.append(token_str)
                    

            next_sep_pos = i % self.n_segments == 0
            eos_pos = i % (12 * self.n_segments) == 0
            
            if eos_pos:
                new_seq.append(eos)

            elif next_sep_pos:
                new_seq.append(sep)

        return " ".join(map(str,new_seq))

    # ---- SAX Tokenization ----
    def tokenize_sax(self, data, decode=False):
        his_dat, his_inv_dat = [], []
        for sig_group in data:
            for sig in sig_group:
                signal1 = sig.reshape(1, -1)
                sax_data = self.SAX_List.transform(signal1)
                his_dat.append(sax_data.reshape(self.n_segments,))
                if decode:
                    sax_inv = self.SAX_List.inverse_transform(sax_data).reshape(-1)
                    his_inv_dat.append(sax_inv)
        if decode:
            self.toks_sax, self.toks_sax_inv = np.array(his_dat), his_inv_dat
            return self.toks_sax, self.toks_sax_inv
        else:
            self.toks_sax = np.array(his_dat)
            self.toks_sax_inv = None
            return self.toks_sax 

    def tokenize_1d_sax(self, data, decode=False):
        his_dat, his_inv_dat = [], []
        for sig_group in data:
            for _, sig in enumerate(sig_group):
                signal1 = sig.reshape(1, -1)
                sax_data = self.SAX_List.transform(signal1)
                his_dat.append(sax_data.reshape(self.n_segments * 2,))
                if decode:
                    sax_inv = self.SAX_List.inverse_transform(sax_data).reshape(-1)
                    his_inv_dat.append(sax_inv)
        if decode:
            self.toks_1d, self.toks_1d_inv = np.array(his_dat), his_inv_dat
            return self.toks_1d, self.toks_1d_inv
        else:
            self.toks_1d = np.array(his_dat)
            self.toks_1d_inv = None
            return self.toks_1d       

    def reshape_tokens(self,data, n_sig):
        if self.symbolizer_model in ["SAX", "PAA"]:
            toks = data.reshape(n_sig, 12, self.n_segments)
            return toks
        if self.symbolizer_model == "1dSAX":
            toks = data.reshape(n_sig, 12, self.n_segments*2)
            return toks

    def token_to_string(self, data):
        return [[[str(token) for token in segment] for segment in lead] for lead in data]

    
    def pad_signal(self, data, target_length):
        padded_signals = np.zeros((len(data),len(data[0]), target_length))
        for i in range(len(data)):
            for j in range(len(data[i])):
                #padded_signals[i][j] = np.pad(data[i][j], (0, max(0, target_length-len(data[i][j]))), "constant")
                sig = data[i][j]
                padded_signals[i][j, :len(sig)] = sig #This copies the original signal in the beginning
        return padded_signals    

    def truncate_signal(self, data, target_length):
        truncated_signals = np.zeros((len(data),len(data[0]), target_length))
        for i in range(len(data)):
            for j in range(len(data[i])):
                sig = data[i][j]

                if len(sig) >= target_length:
                    truncated_signals[i][j] = sig[:target_length] 
        return truncated_signals

    def encode(
        self,
        signals,
        padding: bool = True,
        truncation: bool = True,
    ):      
        n_signals = len(signals)
        n_leads = len(signals[0])
        target_length = self.target_length  

        # Pad/truncate 
        padded_signals = np.zeros((n_signals, n_leads, target_length))
        n_padded_elements = np.zeros((n_signals, n_leads), dtype=int)

        for i in range(n_signals):
            for j in range(n_leads):
                sig = signals[i][j]
                sig_len = len(sig)

                if sig_len >= target_length:
                    padded_signals[i, j, :] = sig[:target_length]
                    n_padded_elements[i, j] = 0
                else:
                    padded_signals[i, j, :sig_len] = sig
                    n_padded_elements[i, j] = target_length - sig_len

        padded_signals = np.swapaxes(padded_signals, 1,2)
        print(f"Padded signals: {padded_signals.shape}")

        #print("number of padded elements:", n_padded_elements)
        # Tokenize the sample
        if self.symbolizer_model in ["SAX", "PAA"]:
            toks, _ = self.tokenize_sax(padded_signals)
        elif self.symbolizer_model == "1dSAX":
            toks, _ = self.tokenize_1d_sax(padded_signals)
        else:
            raise ValueError("symbolizer_model must be either 'PAA', 'SAX' or '1dSAX'")
        
        segment_size = target_length / self.n_segments
        n_padded_segments = np.ceil(n_padded_elements / segment_size).astype(int)

        clean_toks = []
        for i in range(n_signals):
            leads = []
            for j in range(n_leads):
                valid_length = self.n_segments - n_padded_segments[i, j]
                if valid_length <= 0:
                    valid_length = 1  # keep at least 1 segment to avoid empty leads
                leads.append(toks[i, j, :valid_length])
            clean_toks.append(leads)

        clean_toks = np.array(clean_toks, dtype=object)

        # Convert tokens to strings and then IDs
        all_sentences = []
        all_encodings = []

        for i in range(n_signals):
            input_ids = []
            attention_mask = []

            for j in range(n_leads):
                lead_tokens = clean_toks[i][j]
                # SAX tokens to vocab IDs
                lead_ids = [self.vocab.get(str(tok), self.vocab["<unk>"]) for tok in lead_tokens]
                lead_mask = [1] * len(lead_ids)

                # Pad or truncate to fixed number of segments
                if truncation and len(lead_ids) > self.n_segments:
                    lead_ids = lead_ids[:self.n_segments]
                    lead_mask = lead_mask[:self.n_segments]

                if padding and len(lead_ids) < self.n_segments:
                    pad_len = self.n_segments - len(lead_ids)
                    lead_ids += [self.vocab["<pad>"]] * pad_len
                    lead_mask += [0] * pad_len

                input_ids.append(lead_ids)
                attention_mask.append(lead_mask)

            # Build special tokens sequence (string)
            sentence_w_st = self.build_inputs_with_special_tokens(input_ids)
            all_sentences.append(sentence_w_st)

            # Flatten
            flat_input_ids = [token for lead in input_ids for token in lead]
            flat_attention_mask = [mask for lead in attention_mask for mask in lead]

            all_encodings.append({
                "input_ids": torch.tensor(flat_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(flat_attention_mask, dtype=torch.long)
            })

        return all_sentences, all_encodings if len(all_encodings) > 1 else all_encodings[0]    

    def identify_beatmarks(self, array):
        # TODO we probably want to move this to the ECGDataset, where the ECGDataset produces segment id's.
        simple_peaks = find_peaks(cleaned_signal, distance=500)
        mean_int = int(np.diff(simple_peaks[0]).mean()//2)
        simple_segments = [(max(int(pi)-mean_int, 0), min(int(pi)+mean_int, len(cleaned_signal))) for pi in simple_peaks[0]]

    def compute_beat_based_attention_mask(self, ecg_data):
        # TODO: NOT ACTIVE NOW
        '''
        Computes attention mask focusing only on P wave, QRS complex and T wave
        Credits:https://github.com/Edoar-do/HuBERT-ECG/blob/master/code/dataset.py
        '''

        ecg_data = ecg_data.reshape(12, self.config.MAX_OUTPUT_LENGTH)
        _, rpeaks = nk.ecg_peaks(ecg_data[1], sampling_rate=self.fs_out) #compute R peaks from II
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_data[1], rpeaks, sampling_rate=500, method="dwt", show=False, show_type='all')
        signal_dwt['ECG_R_Peaks'] = 0
        signal_dwt['ECG_R_Peaks'].iloc[rpeaks['ECG_R_Peaks']] = 1

        p_wave = signal_dwt['ECG_P_Onsets'] | signal_dwt['ECG_P_Offsets'] # binary serie with 1 where P waves start and stop
        qrs_complex = signal_dwt['ECG_Q_Peaks'] | signal_dwt['ECG_S_Peaks'] # binary serie with 1 where QRS complexes start and stop
        t_wave = signal_dwt['ECG_T_Onsets'] | signal_dwt['ECG_T_Offsets'] # binary serie with 1s where T waves start and stop

        p_starts_stops = p_wave[p_wave != 0].index.tolist()
        if len(p_starts_stops) % 2 != 0:
            p_starts_stops.append(min(p_starts_stops[-1]+1, 2499))
        p_starts_stops = np.array(p_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each P wave detected

        t_starts_stops = t_wave[t_wave != 0].index.tolist()
        if len(t_starts_stops) % 2 != 0:
            t_starts_stops.append(min(t_starts_stops[-1]+1, 2499))
        t_starts_stops = np.array(t_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each T wave detected


        qrs_starts_stops = qrs_complex[qrs_complex != 0].index.tolist()
        if len(qrs_starts_stops) % 2 != 0:
            qrs_starts_stops.append(min(qrs_starts_stops[-1]+1, 2499))
        qrs_starts_stops = np.array(qrs_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each QRS complex detected

        # building the attention mask in order to attend only samples in the p waves
        for start, stop in p_starts_stops:
            p_wave.iloc[start : stop] = 1

        # building the attention mask in order to attend only samples in the t waves
        for start, stop in t_starts_stops:
            t_wave.iloc[start : stop] = 1

        # building the attention mask in order to attend only samples in the qrs complexes
        for start, stop in qrs_starts_stops:
            qrs_complex.iloc[start : stop] = 1

        # global attention mask merging all interest regions
        attention_mask = (p_wave | t_wave | qrs_complex).tolist()
        attention_mask = np.repeat([attention_mask], 12, axis=0)
        attention_mask = np.concatenate(attention_mask, axis=0)

        return attention_mask