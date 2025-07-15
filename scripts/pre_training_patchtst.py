# Pre-training script for PatchTST
# args
# --------
# context_length: int=512
# num_forecast_mask_patch: int=64
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
import os
import torch
import argparse
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    PatchTSTForPretraining,
    Trainer,
    TrainingArguments,
    DataCollatorForMaskedLanguageModelling
)
import copy
from src.ecg import preprocessor, dataset

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
        num_forecast_mask_patches = args.num_forecast_mask_patch,
        random_mask_ratio=args.random_mask_ratio,   # 40 % of patches are blanked
        use_cls_token=True
    )

    model = PatchTSTForPretraining(config)

    training_args = TrainingArguments(
        output_dir="ptst_pretrain",
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.decay_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        fp16=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=3,
        save_strategy = "epoch",
        evaluation_strategy = "epoch",
        eval_steps = 1,
        save_steps = 1,
        load_best_model_at_end = True,
        metric_for_best_model  = "eval_loss",
        greater_is_better      = False,
    )

    Prepper = preprocessor.ECGSignalProcessor()
    for _file in os.listdir(args.input_dir):
        fn = os.path.join(args.input_dir, _file)

        if fn.endswith('.npy'):
            arr = np.load(fn)
            # pre-process
            _arr = Prepper.filter(arr)
            _ds = SlidingWindowDataset(arr, config.context_length)
        elif fn.endswith('.hea'):
            arr = dataset.get_hea(fn)
            _arr = Prepper.filter(arr)
            _ds = SlidingWindowDataset(_arr, config.context_length)

        # somehow add to total dataset DS
        #

    trainer = Trainer(model=model, args=training_args, train_dataset=DS)
    trainer.train()

    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.save_pretrained(args.output_dir, save_serialization=True)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--context_length', type = int, default = 512)
    argparser.add_argument('--patch_length', type = int, default=32)
    argparser.add_argument('--patch_stride', type = int, default=32)
    argparser.add_argument('--num_workers', type = int, default=1)
    argparser.add_argument('--batch_size', type = int, default=16)
    argparser.add_argument('--d_model', type = int, default=256)
    argparser.add_argument('--num_hidden_layers', type = int, default=4)
    argparser.add_argument('--num_attention_heads', type = int, default=6)
    argparser.add_argument('--mask_type', type = str, options=['forecast', 'random'])
    argparser.add_argument('--random_mask_ratio', type = float, default=0.25)
    argparser.add_argument('--num_forecast_mask_patch', type=int, default=64)
    argparser.add_argument('--learning_rate', type = float, default=1e-5)
    argparser.add_argument('--decay_rate', type = float, default = 1e-3)
    argparser.add_argument('--learning_schedule', type= str, options=['linear', 'cyclic'])
    argparser.add_argument('--num_epochs', type=int, default=1)
    argparser.add_argument('--patience', type=int, default=5)
    argparser.add_argument('--warmup_steps', type=int, default=1000)
    argparser.add_argument('--logging_steps', type=int, default=100)
    argparser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    argparser.add_argument('--output_dir', type=str, default='./output')
    argparser.add_argument('--input_dir', type=str, required=True)
