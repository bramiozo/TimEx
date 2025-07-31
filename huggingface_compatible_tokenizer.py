from tokenizers import Tokenizer, models, pre_tokenizers, decoders, AddedToken
from transformers import PreTrainedTokenizer
from tslearn.piecewise import SymbolicAggregateApproximation
import numpy as np
import json
import os
from shutil import copyfile
from typing import Optional, Any, List
import torch
from torch.utils.data import Dataset

class Word_Tokenizer():

    model_input_names = ["input_ids", "attention_mask"]
    prefix_tokens: list[int] = []
    

 


    
    def __init__(
        self,
        filepaths,
        vocab_file=str,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        sep_token="[SEP]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        **kwargs):

        # Special tokens setup
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token


        # Define vocabulary
        self.vocab = {str(i): i + 6 for i in range(1,301)}
        self.vocab.update({
                  "<unk>": 0,
                  "<s>" : 1,
                  "</s>" : 2,
                  "<pad>": 3,
                  "[CLS]": 4,
                  "[SEP]": 5                  
                           })

        self.tokenizer = Tokenizer(models.WordLevel(self.vocab, unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.decoder = decoders.WordPiece()
        self.n_segments = 100
        self.alphabet_size = 300
        self.toks_sax = None
        self.toks_sax_inv = None
        self.toks_ld = None
        self.toks_ld_inv = None
        self.filepaths = filepaths
        self.unk_token = unk_token
        self.sax = SymbolicAggregateApproximation(n_segments=self.n_segments, alphabet_size_avg=self.alphabet_size)


        # SAX transformer
    def tokenize_sax(self, data):
            his_dat = []
            his_inv_dat = []

            for sig_group in data:
                for sig in sig_group:
                    signal1 = sig.reshape(1, -1)
                    sax_data = self.sax.fit_transform(signal1)
                    his_dat.append(sax_data.reshape(self.n_segments,))
                    sax_inv = self.sax.inverse_transform(sax_data).reshape(-1)
                    his_inv_dat.append(sax_inv)

            self.toks_sax = np.array(his_dat)
            self.toks_sax_inv = his_inv_dat

            return self.toks_sax, self.toks_sax_inv

    def reshape_tokens(self,data):
          toks = self.toks_sax.reshape(len(self.filepaths), 12, self.n_segments)
          return toks

    def token_to_string(self,data):
          return [" ".join(str(token) for lead in sample for token in lead) for sample in data]





    def create_ecg_dataset(self, samples, labels, max_length=512):
        dataset = []

        for i in range(len(samples)):
            text = samples[i]
            label = labels[i]
            
            encoding = self.tokenizer.encode(text)
            
            input_ids = encoding.ids[:max_length]
            attention_mask = [1] * len(input_ids)


            # Padding
            pad_length = max_length - len(input_ids)
            input_ids += [self.vocab["<pad>"]] * pad_length
            attention_mask += [0] * pad_length

            item = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long)
            }
            dataset.append(item)

        return dataset


    def vocab_size(self):
        return len(self.tokenizer.get_vocab())

    def get_vocab(self):
        vocab = self.tokenizer.get_vocab()
        #vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str):
        # Here you can tokenize SAX tokens (space-separated string) or plain text
        return self.tokenizer.encode(text).tokens

    def _convert_token_to_id(self, token):
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            token_id = self.tokenizer.token_to_id(self.unk_token.content)
        return token_id

    def _convert_id_to_token(self, index):
        return self.tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens): 
        return " ".join(tokens)

    def _decode(
        self,
        token_ids,
        skip_special_tokens=False,
        **kwargs,
    ):

        special_token_ids = {self.vocab[tok] for tok in ["<unk>", "<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[MASK]"] if tok in self.vocab}
        tokens = [self._convert_id_to_token(tok_id) for tok_id in token_ids if not (skip_special_tokens and tok_id in special_token_ids)]

        text = [self.convert_tokens_to_string(tokens)]

        numeric_tokens = [int(tok) for tok in tokens if tok.isdigit()]
        token_array = np.array(numeric_tokens).reshape(1, -1)

        sax_inverse = self.sax.inverse_transform(token_array).reshape(-1)
        return text , sax_inverse

 
