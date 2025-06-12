# Pre-training script for PatchTST
# args
# --------
# context_length: int=512
# forecast_horizon: int=64
# patch_length: int=32
# num_workers: int=1
# batch_size: int=16  # 128
# d_model: int=512
# num_hidden_layers: int=12
# num_attention_heads: int=16
# pre_train_type: Literal['forecast', 'masked']
# train_folder: str|None=None
#
# Train folder consists of .hea/.dat or .npy files.
#
# Each timeseries is pre-processed with some ECGPreProcessor
import numpy as np
import torch
import argparse
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    PatchTSTForPretraining,
    Trainer,
    TrainingArguments
)


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, array, context_len):
        self.x = torch.as_tensor(array, dtype=torch.float32)
        self.context_len = context_len
    def __len__(self):
        return self.x.shape[0] - self.context_len + 1
    def __getitem__(self, idx):
        window = self.x[idx : idx + self.context_len]
        return {"past_values": window}


def main(args):
    config = PatchTSTConfig(
        num_input_channels=args.input_channels,
        context_length=args.context_length,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        mask_type=args.mask_type,
        random_mask_ratio=args.random_mask_ratio,   # 40 % of patches are blanked
        use_cls_token=True
    )

    model = PatchTSTForPretraining(config)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--context_length', type = int, default = 512)
    argparser.add_argument('--forecast_horizon', type = int, default=64)
    argparser.add_argument('--patch_length', type = int, default=32)
    argparser.add_argument('--patch_stride', type = int, default=32)
    argparser.add_argument('--num_workers', type = int, default=1)
    argparser.add_argument('--batch_size', type = int, default=16)
    argparser.add_argument('--d_model', type = int, default=256)
    argparser.add_argument('--num_hidden_layers', type = int, default=4)
    argparser.add_argument('--num_attention_heads', type = int, default=6)
    argparser.add_argument('--mask_type', type = str, options=['forecast', 'random'])
    argparser.add_argument('--random_mask_ratio', type = float, default=0.25)
