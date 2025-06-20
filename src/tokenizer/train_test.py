from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import wfdb
import pickle
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
from transformers import PreTrainedTokenizerFast
import transformers
from tokenizers import SentencePieceBPETokenizer
import sys
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, default_data_collator
from transformers import AutoTokenizer, BigBirdForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import BigBirdConfig, BigBirdModel, BigBirdForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, IntervalStrategy
import os
import torch
from typing import List, Dict
from sklearn.model_selection import train_test_split




import sax
import ECGSignalPreprocessor



filepaths=[r"C:\Users\ASUS\Downloads\data\mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1\files\p10000032\s100780919\100780919",
           r"C:\Users\ASUS\Downloads\data\mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1\files\p10000032\s102511170\102511170",
           r"C:\Users\ASUS\Downloads\data\mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1\files\p10000032\s107143276\107143276",
           r"C:\Users\ASUS\Downloads\data\mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1\files\p10001217\s102172660\102172660",
           r"C:\Users\ASUS\Downloads\data\mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1\files\p10001217\s105362569\105362569",
           r"C:\Users\ASUS\Downloads\data\mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1\files\p10001725\s102147240\102147240",
           r"C:\Users\ASUS\Downloads\data\mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1\files\p10002428\s102144047\102144047",
           r"C:\Users\ASUS\Downloads\data\mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1\files\p10002428\s102241375\102241375",
           r"C:\Users\ASUS\Downloads\data\mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1\files\p10002428\s102616671\102616671",
           r"C:\Users\ASUS\Downloads\data\mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1\files\p10002428\s103036945\103036945"]



record_name=[]
p_signals=[]
n_sig=[]
sig_len=[]
sig_name=[]
ecg_signals=[]


for fname in filepaths:
    records = wfdb.rdrecord(fname)
    record_name.append(records.record_name)
    p_signals.append(np.transpose(records.p_signal))
    n_sig.append(records.n_sig)
    sig_len.append(records.sig_len)
    sig_name.append(records.sig_name)
p_signal=np.array(p_signals)



all_data=[]
for i in range(len(filepaths)):
      x = np.linspace(0, 10, sig_len[i])
      processor=ECGSignalPreprocessor.ECGSignalProcessor(p_signal[i], records.n_sig, records.sig_len, records.sig_name, x)
#      processor.plot_signals()
      processor.detect_peaks_and_dips()
      processor.apply_bandpass_filter()
      processor.apply_notch_filter(processor.filtered_signal)
      processor.apply_savgol_filter(processor.filtered_signal)
      processor.apply_detrend(processor.smoothed)
      processor.trim_signal(processor.detrend)
      #processor.quality_check(processor.trim_arr)
      all_data.append(processor.trim_arr)


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

padded_signals=[]
for i in range(len(all_data)):
    padded_signal=padding(all_data[i],5000)
    padded_signals.append(padded_signal)

n_segments=100
alphabet_size=300


tokenizers = sax.ECGTokenizer()
toks_sax, toks_sax_inv = tokenizers.tokenize_sax(padded_signals)
#toks=toks_sax.reshape(len(filepaths),n_sig[0]*n_segments)

#toks = np.array([' '.join(map(str, row)) for row in toks])
toks = toks_sax.reshape(len(filepaths), n_sig[0]*n_segments)
toks = np.array([' '.join([f"tok_{x}" for x in row]) for row in toks])

tokenizer = AutoTokenizer.from_pretrained("drive/MyDrive/tokenizer_file")

tokenized = tokenizer(toks.tolist(), padding="max_length", truncation=True, max_length=256, return_tensors="pt")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)





class ArrayTextDataset:
    def __init__(self, texts: List[str], tokenizer, block_size: int):
        # Filter out empty or whitespace-only strings (similar to your original)
        filtered_texts = [text for text in texts if text and not text.isspace()]

        # Tokenize all texts at once
        batch_encoding = tokenizer(
            filtered_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            padding="max_length"  # no padding here; adjust if needed
        )

        # Store tokenized input_ids as tensors in a list of dicts
        self.examples = [
            {"input_ids": torch.tensor(batch_encoding["input_ids"][i]),
            "attention_mask": torch.tensor(batch_encoding["attention_mask"][i])}
            for i in range(len(batch_encoding["input_ids"]))
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.examples[i]


bs = 256

train_texts, val_texts= train_test_split(toks, test_size=0.2)


class PreTokenizedDataset_new(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }

train_encodings = tokenizer(train_texts.tolist(), padding="max_length", truncation=True, max_length=256, return_tensors="pt")
val_encodings = tokenizer(val_texts.tolist(), padding="max_length", truncation=True, max_length=256, return_tensors="pt")

train_dataset_new = PreTokenizedDataset_new(train_encodings)
eval_dataset_new = PreTokenizedDataset_new(val_encodings)



model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base")

model.resize_token_embeddings(len(tokenizer))



training_args = TrainingArguments(
    output_dir="C:\Users\ASUS\Downloads",
    overwrite_output_dir=True,
    num_train_epochs=1000,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_strategy='epoch',
    eval_strategy="epoch",
    save_strategy='epoch',
    metric_for_best_model='loss',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model.to("cpu"),
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset_new,
    eval_dataset=eval_dataset_new)

trainer.train()

!nvidia-smi
