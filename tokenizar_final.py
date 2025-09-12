from tokenizers import Tokenizer, models, pre_tokenizers, decoders, AddedToken
import tokenizers
from transformers import PreTrainedTokenizer
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation
from typing import Type, List
import numpy as np
import math
import torch

class Word_Tokenizer(PreTrainedTokenizer):

    model_input_names = ["input_ids", "attention_mask"]
    prefix_tokens: list[int] = []

    def __init__(
        self,
        SAX_List: Type[OneD_SymbolicAggregateApproximation] | Type[SymbolicAggregateApproximation] | None = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
        cls_token: str = "[CLS]",
        **kwargs,
    ):
        assert (
            SAX_List is None
            or isinstance(SAX_List, OneD_SymbolicAggregateApproximation)
            or isinstance(SAX_List, SymbolicAggregateApproximation)
        )

        if isinstance(SAX_List, OneD_SymbolicAggregateApproximation):
            vocab_size = SAX_List.alphabet_size_avg + SAX_List.alphabet_size_slope
        elif isinstance(SAX_List, SymbolicAggregateApproximation):
            vocab_size = SAX_List.alphabet_size_avg
    

        # Special tokens
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # Build vocab: numbers 1..vocab_size plus specials
        self.vocab = {str(i): i + 6 for i in range(0, vocab_size + 1)}
        self.vocab.update({
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "<pad>": 3,
            "[CLS]": 4,
            "[SEP]": 5
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
        self.SAX_List = SAX_List
        self.n_segments = self.SAX_List.n_segments
        self.alphabet_size = self.SAX_List.alphabet_size_avg
        self.target_length = 5000

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
        return self._vocab_size

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text: str) -> List[str]:
        encoding = self.tokenizer.encode(text)
        return encoding.tokens

    def _convert_token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        return token_id if token_id is not None else self.tokenizer.token_to_id(self.unk_token.content)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def _decode(self, token_ids, skip_special_tokens: bool = True, **kwargs):
        special_token_ids = {self.vocab[tok] for tok in ["<unk>", "<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[MASK]"] if tok in self.vocab}
        
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        #Delete special token ids
        decoded = []
        for sequence in token_ids:  
            filtered_ids = [
                int(tok_id) for tok_id in sequence
                if not (skip_special_tokens and int(tok_id) in special_token_ids)]
            decoded.append(filtered_ids)
            

        #Decode sax/1dsax
        sax_decoded = []
        for i in range(len(decoded)):
            sax_decoded.append(self.SAX_List.inverse_transform(np.array(decoded[i]).reshape(-1, 1)).reshape(-1))


        return sax_decoded

    def build_inputs_with_special_tokens(self, data):
        #Insert [SEP] after every self.n_segment tokens
        #Insert </s> after every 12*self.n_segment tokens
        #Single sequence: `<s> X </s>`
        #pair of sequences: `<s> A [SEP] B </s>`


        #tokens = np.array(data).reshape(len(data), len(data[0])*len(data[0][0]))
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
    def tokenize_sax(self, data):
        n_signals = len(data)
        n_leads = len(data[0])
        his_dat, his_inv_dat = [], []
        for sig_group in data:
            for sig in sig_group:
                signal1 = sig.reshape(1, -1)
                sax_data = self.SAX_List.transform(signal1)
                his_dat.append(sax_data.reshape(self.n_segments,))
                sax_inv = self.SAX_List.inverse_transform(sax_data).reshape(-1)
                his_inv_dat.append(sax_inv)

        toks_sax = np.array(his_dat).reshape(n_signals, n_leads, self.n_segments)        
        self.toks_sax, self.toks_sax_inv = np.array(toks_sax), his_inv_dat
        return self.toks_sax, self.toks_sax_inv

    def tokenize_1d_sax(self, data):
        n_signals = len(data)
        n_leads = len(data[0])
        his_dat, his_inv_dat = [], []
        for sig_group in data:
            for _, sig in enumerate(sig_group):
                signal1 = sig.reshape(1, -1)
                sax_data = self.SAX_List.transform(signal1)
                his_dat.append(sax_data.reshape(self.n_segments * 2,))
                sax_inv = self.SAX_List.inverse_transform(sax_data).reshape(-1)
                his_inv_dat.append(sax_inv)

        toks_sax = np.array(his_dat).reshape(n_signals, n_leads, 2*self.n_segments)        
        self.toks_1d, self.toks_1d_inv = np.array(toks_sax), his_inv_dat
        return self.toks_1d, self.toks_1d_inv


    def reshape_tokens(self,data, n_sig,symbolizer_model):
        if symbolizer_model == "sax":
            toks = data.reshape(n_sig, 12, self.n_segments)
            return toks
        if symbolizer_model == "1d_sax":
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


    def encode(self, signals, symbolizer_model: str, padding: bool = True, truncation: bool = True):
    

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
        print("number of padded elements:", n_padded_elements)
        # Apply SAX 
        if symbolizer_model == "sax":
            toks, toks_inv = self.tokenize_sax(padded_signals)
 
        elif symbolizer_model == "1d_sax":
            toks, toks_inv = self.tokenize_1d_sax(padded_signals)

        else:
            raise ValueError("symbolizer_model must be either 'sax' or '1d_sax'")

        # Remove padded SAX segments
        print("token shape:",toks.shape)
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





            

            
           
