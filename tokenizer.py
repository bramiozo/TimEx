from tokenizers import Tokenizer, models, pre_tokenizers, decoders, AddedToken
from transformers import PreTrainedTokenizer
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation
from typing import Type, List
import numpy as np
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
        self.vocab = {str(i): i + 6 for i in range(1, vocab_size + 1)}
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


        self.toks_sax = None
        self.toks_sax_inv = None
        self.toks_1d = None
        self.toks_1d_inv = None
        self.SAX_List = SAX_List
        self.n_segments = self.SAX_List.n_segments
        self.alphabet_size = self.SAX_List.alphabet_size_avg
        self.max_length = 12*(self.n_segments)

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
        #special_token_ids = {self.vocab[tok] for tok in ["<unk>", "<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[MASK]"] if tok in self.vocab}
        
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        #decoded = []
        #for sequence in token_ids:  
         #   filtered_ids = [
          #  int(tok_id) for tok_id in sequence
           # if not (skip_special_tokens and int(tok_id) in special_token_ids)]
            #decoded.append(filtered_ids)
            

        sax_decoded = []
        #for i in range(len(decoded)):
         #   sax_decoded.append(self.SAX_List.inverse_transform(np.array(decoded[i]).reshape(-1, 1)).reshape(-1))
        for i in range(len(token_ids)):
            sax_decoded.append(self.SAX_List.inverse_transform(np.array(token_ids[i]).reshape(-1, 1)).reshape(-1))
       
<<<<<<< HEAD
        
        return np.array(sax_decoded).reshape(len(sax_decoded),12,self.n_segments)


     def build_inputs_with_special_tokens(self, data):
        #Insert [SEP] after every self.n_segment tokens
        #Insert </s> after every 12*self.n_segment tokens
        #Single sequence: `[CLS] X </s>`
        #- pair of sequences: `[CLS] A [SEP] B </s>`

        tokens = data.reshape(len(data), len(data[0])*len(data[0][0]))
        
        cls = self.cls_token
        sep = self.sep_token
        eos = self.eos_token

        all_sequences = []
        for seq in tokens:

            new_seq = []  # Start with [CLS]
            new_seq.append(cls)
            for i, token in enumerate(seq, 1):
                token_str = str(token)
                new_seq.append(token_str)

                next_sep_pos = i % self.n_segments == 0
                eos_pos = i % (12 * self.n_segments) == 0
            
                if eos_pos:
                    new_seq.append(eos)

                elif next_sep_pos:
                    new_seq.append(sep)

            all_sequences.append(" ".join(map(str,new_seq)))
        return all_sequences   
=======
        return sax_decoded

>>>>>>> origin/Oykudetachedhead

    # ---- SAX Tokenization ----
    def tokenize_sax(self, data):
        his_dat, his_inv_dat = [], []
        for sig_group in data:
            for sig in sig_group:
                signal1 = sig.reshape(1, -1)
                sax_data = self.SAX_List.transform(signal1)
                his_dat.append(sax_data.reshape(self.n_segments,))
                sax_inv = self.SAX_List.inverse_transform(sax_data).reshape(-1)
                his_inv_dat.append(sax_inv)
        self.toks_sax, self.toks_sax_inv = np.array(his_dat), his_inv_dat
        return self.toks_sax, self.toks_sax_inv

    def tokenize_1d_sax(self, data):
        his_dat, his_inv_dat = [], []
        for sig_group in data:
            for _, sig in enumerate(sig_group):
                signal1 = sig.reshape(1, -1)
                sax_data = self.SAX_List.transform(signal1)
                his_dat.append(sax_data.reshape(self.n_segments * 2,))
                sax_inv = self.SAX_List.inverse_transform(sax_data).reshape(-1)
                his_inv_dat.append(sax_inv)
        self.toks_1d, self.toks_1d_inv = np.array(his_dat), his_inv_dat
        return self.toks_1d, self.toks_1d_inv


    def reshape_tokens(self,data, n_sig,symbolizer_model):
        if symbolizer_model == "sax":
            toks = data.reshape(n_sig, 12, self.n_segments)
            return toks
        if symbolizer_model == "1d_sax":
            toks = data.reshape(n_sig, 12, self.n_segments*2)
            return toks

    def token_to_string(self, data):
        return [" ".join(str(token) for lead in sample for token in lead) for sample in data]


    def encode(
        self,
        signals,
        symbolizer_model: str,
        padding: bool = True,
        truncation: bool = True,
    ):

            all_encodings = []
            # Tokenize the sample
            if symbolizer_model == "sax":
                toks, _ = self.tokenize_sax(signals)

            elif symbolizer_model == "1d_sax":
                toks, _ = self.tokenize_1d_sax(signals)

            else:
                raise ValueError("symbolizer_model must be either 'sax' or '1d_sax'")

            #Reshape
            reshaped = self.reshape_tokens(toks, len(signals), symbolizer_model)

            # Convert tokens into a string
            text = self.token_to_string(reshaped)
            
            #Token sentence with the special tokens
            sentence_w_st = self.build_inputs_with_special_tokens(reshaped)

            # Tokenizer encoding
            for i in range(len(text)):
                encoding = self.tokenizer.encode(text[i])
                input_ids = encoding.ids
                attention_mask = [1] * len(input_ids)
            
                # Truncate if necessary
                if truncation and len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]

                # Pad if necessary
                if padding and len(input_ids) < self.max_length:
                    pad_len = max_length - len(input_ids)
                    input_ids += [self.vocab["<pad>"]] * pad_len
                    attention_mask += [0] * pad_len

                # Append encoding to results
                all_encodings.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    })


            return sentence_w_st, all_encodings if len(all_encodings) > 1 else all_encodings[0]
