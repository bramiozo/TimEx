from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPretraining,
    Trainer,
    TrainingArguments,
    set_seed,
)

from timex.ecg.preprocessor import ECGSignalProcessor


LOGGER = logging.getLogger(__name__)


def _record_base_path(hea_path: Path) -> str:
    """Return WFDB record path without extension for rdrecord/rdheader."""
    return str(hea_path.with_suffix(""))


def discover_hea_files(data_dir: Path, recursive: bool = True) -> list[Path]:
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {data_dir}")

    if recursive:
        files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".hea"]
    else:
        files = [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".hea"]

    files.sort()
    return files


def split_train_eval(
    files: Sequence[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if len(files) == 0:
        return [], []

    files_copy = list(files)
    rng = random.Random(seed)
    rng.shuffle(files_copy)

    if val_ratio <= 0 or len(files_copy) < 2:
        return files_copy, []

    n_val = int(round(len(files_copy) * val_ratio))
    n_val = max(1, n_val)
    n_val = min(n_val, len(files_copy) - 1)

    eval_files = files_copy[:n_val]
    train_files = files_copy[n_val:]
    return train_files, eval_files


class ECGPretrainingDataset(Dataset):
    """
    Dataset for HF time-series pretraining from WFDB .hea/.dat ECG records.

    Output format is compatible with PatchTSTForPretraining:
      - past_values: [context_length, num_input_channels]
      - past_observed_mask: [context_length, num_input_channels]
    """

    def __init__(
        self,
        hea_files: Sequence[Path],
        *,
        context_length: int,
        num_input_channels: int = 12,
        target_sampling_rate: int = 500,
        windows_per_record: int = 1,
        random_crop: bool = True,
        strict_num_input_channels: bool = True,
        apply_preprocessing: bool = True,
        preprocessing_steps: Sequence[str] = ("detrend", "notch", "bandpass", "resample"),
        detrend_method: str = "sliding_median",
        notch_freq: float | Sequence[float] = (50.0,),
        notch_bandwidth: float = 1.0,
        bandpass_lowcut: float = 0.5,
        bandpass_highcut: float = 40.0,
        filter_order: int = 4,
        normalize_per_lead: bool = True,
        cache_records: bool = False,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        if windows_per_record <= 0:
            raise ValueError("windows_per_record must be > 0")

        self.hea_files = list(hea_files)
        self.context_length = context_length
        self.num_input_channels = num_input_channels
        self.target_sampling_rate = target_sampling_rate
        self.windows_per_record = windows_per_record
        self.random_crop = random_crop
        self.strict_num_input_channels = strict_num_input_channels
        self.apply_preprocessing = apply_preprocessing
        self.preprocessing_steps = tuple(preprocessing_steps)
        self.detrend_method = detrend_method

        if isinstance(notch_freq, (int, float)):
            notch_freqs = [float(notch_freq)]
        else:
            notch_freqs = [float(f) for f in notch_freq]

        if len(notch_freqs) == 0:
            raise ValueError("notch_freq must contain at least one frequency")
        if any(f <= 0 for f in notch_freqs):
            raise ValueError(f"All notch frequencies must be > 0, got {notch_freqs}")

        self.notch_freqs = notch_freqs
        self.notch_bandwidth = notch_bandwidth
        self.bandpass_lowcut = bandpass_lowcut
        self.bandpass_highcut = bandpass_highcut
        self.filter_order = filter_order
        self.normalize_per_lead = normalize_per_lead
        self.cache_records = cache_records

        self._cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.hea_files) * self.windows_per_record

    def _fix_num_channels(self, signal: np.ndarray, hea_path: Path) -> np.ndarray:
        n_channels, n_samples = signal.shape

        if n_channels == self.num_input_channels:
            return signal

        if self.strict_num_input_channels:
            raise ValueError(
                f"Record {hea_path} has {n_channels} channels, expected {self.num_input_channels}"
            )

        if n_channels > self.num_input_channels:
            LOGGER.warning(
                "Record %s has %d channels; truncating to first %d",
                hea_path,
                n_channels,
                self.num_input_channels,
            )
            return signal[: self.num_input_channels, :]

        LOGGER.warning(
            "Record %s has %d channels; zero-padding channels to %d",
            hea_path,
            n_channels,
            self.num_input_channels,
        )
        out = np.zeros((self.num_input_channels, n_samples), dtype=np.float32)
        out[:n_channels, :] = signal
        return out

    def _load_and_preprocess_record(self, record_idx: int) -> torch.Tensor:
        hea_path = self.hea_files[record_idx]
        record = wfdb.rdrecord(_record_base_path(hea_path))

        signal = np.asarray(record.p_signal.T, dtype=np.float32)  # [channels, samples]
        fs = int(round(float(record.fs))) if record.fs is not None else self.target_sampling_rate

        signal = self._fix_num_channels(signal, hea_path)

        if self.apply_preprocessing:
            proc = ECGSignalProcessor(signal, num_channels=self.num_input_channels, fs=fs)

            if "detrend" in self.preprocessing_steps:
                proc.apply_detrend(method=self.detrend_method)

            if "notch" in self.preprocessing_steps:
                for notch_freq in self.notch_freqs:
                    max_notch = notch_freq + self.notch_bandwidth
                    if fs > 2 * max_notch:
                        proc.apply_notch_filter(
                            freq=notch_freq,
                            bandwidth=self.notch_bandwidth,
                            order=self.filter_order,
                        )
                    else:
                        LOGGER.warning(
                            "Skipping notch for %s because fs=%s is too low for notch %s±%s Hz",
                            hea_path,
                            fs,
                            notch_freq,
                            self.notch_bandwidth,
                        )

            if "bandpass" in self.preprocessing_steps:
                if fs > 2 * self.bandpass_highcut:
                    proc.apply_bandpass_filter(
                        lowcut=self.bandpass_lowcut,
                        highcut=self.bandpass_highcut,
                        order=self.filter_order,
                    )
                else:
                    LOGGER.warning(
                        "Skipping bandpass for %s because fs=%s is too low for highcut=%s Hz",
                        hea_path,
                        fs,
                        self.bandpass_highcut,
                    )

            if "resample" in self.preprocessing_steps and fs != self.target_sampling_rate:
                proc.standardize_sampling_rate(fs_target=self.target_sampling_rate)

            x = proc.get()  # [channels, samples]
        else:
            x = torch.tensor(signal, dtype=torch.float32)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize_per_lead:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        x = torch.clamp(x, min=-10.0, max=10.0)
        return x

    def _to_context_window(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert [channels, samples] ->
          past_values [context_length, channels]
          past_observed_mask [context_length, channels]
        """
        n_channels, n_samples = x.shape

        if n_channels != self.num_input_channels:
            raise ValueError(
                f"Internal channel mismatch after preprocessing: {n_channels} vs {self.num_input_channels}"
            )

        if n_samples >= self.context_length:
            max_start = n_samples - self.context_length
            if self.random_crop and max_start > 0:
                start = random.randint(0, max_start)
            else:
                start = max_start // 2

            window = x[:, start : start + self.context_length]
            observed = torch.ones(
                (self.context_length, self.num_input_channels),
                dtype=torch.float32,
            )
        else:
            pad_len = self.context_length - n_samples
            pad = torch.zeros((self.num_input_channels, pad_len), dtype=x.dtype)
            window = torch.cat([x, pad], dim=1)

            observed_real = torch.ones((n_samples, self.num_input_channels), dtype=torch.float32)
            observed_pad = torch.zeros((pad_len, self.num_input_channels), dtype=torch.float32)
            observed = torch.cat([observed_real, observed_pad], dim=0)

        past_values = window.transpose(0, 1).contiguous()  # [time, channels]
        return past_values, observed

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record_idx = idx // self.windows_per_record

        if self.cache_records and record_idx in self._cache:
            x = self._cache[record_idx]
        else:
            x = self._load_and_preprocess_record(record_idx)
            if self.cache_records:
                self._cache[record_idx] = x

        past_values, past_observed_mask = self._to_context_window(x)

        return {
            "past_values": past_values,
            "past_observed_mask": past_observed_mask,
        }


@dataclass
class ECGPretrainingCollator:
    """Simple batch collator for PatchTST pretraining batches."""

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        past_values = torch.stack([f["past_values"] for f in features], dim=0)
        past_observed_mask = torch.stack([f["past_observed_mask"] for f in features], dim=0)
        return {
            "past_values": past_values,
            "past_observed_mask": past_observed_mask,
        }


def _resolve_muon_optimizer_class() -> tuple[type[torch.optim.Optimizer], str]:
    """
    Try several known module paths for Muon optimizer and return (class, source).
    """
    candidates = [
        ("torch.optim", "Muon"),
        ("muon", "Muon"),
        ("muon_optimizer", "Muon"),
        ("optimizers.muon", "Muon"),
    ]

    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            klass = getattr(module, class_name)
            return klass, f"{module_name}.{class_name}"
        except (ImportError, AttributeError):
            continue

    tried = ", ".join([f"{m}.{c}" for m, c in candidates])
    raise ImportError(
        "Requested optimizer 'muon' but no Muon class was found. "
        f"Tried: {tried}. Install/provide a Muon optimizer implementation first."
    )


class ECGPretrainingTrainer(Trainer):
    """Trainer with optional MUON optimizer and custom cyclic cosine+warmup scheduler."""

    def __init__(
        self,
        *args: Any,
        optimizer_name: str = "adamw_torch",
        scheduler_name: str = "cosine",
        cycle_steps: int = 0,
        cycle_warmup_steps: int = 0,
        cycle_warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.optimizer_name = optimizer_name.lower().strip()
        self.scheduler_name = scheduler_name.lower().strip()
        self.cycle_steps = int(cycle_steps)
        self.cycle_warmup_steps = int(cycle_warmup_steps)
        self.cycle_warmup_ratio = float(cycle_warmup_ratio)
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if self.optimizer_name != "muon":
            return super().create_optimizer()

        muon_cls, source = _resolve_muon_optimizer_class()
        LOGGER.info("Using MUON optimizer from %s", source)

        optim_kwargs: dict[str, Any] = {
            "lr": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
        }

        signature = inspect.signature(muon_cls.__init__)
        if "betas" in signature.parameters:
            optim_kwargs["betas"] = (self.args.adam_beta1, self.args.adam_beta2)
        if "eps" in signature.parameters:
            optim_kwargs["eps"] = self.args.adam_epsilon

        self.optimizer = muon_cls(self.model.parameters(), **optim_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        if self.scheduler_name != "cosine_warmup_restarts":
            return super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

        if optimizer is None:
            optimizer = self.optimizer if self.optimizer is not None else self.create_optimizer()

        cycle_steps = self.cycle_steps if self.cycle_steps > 0 else num_training_steps
        cycle_steps = max(2, int(cycle_steps))

        if self.cycle_warmup_steps > 0:
            warmup_steps = min(self.cycle_warmup_steps, cycle_steps - 1)
        else:
            warmup_steps = int(round(cycle_steps * self.cycle_warmup_ratio))
            warmup_steps = min(max(0, warmup_steps), cycle_steps - 1)

        min_lr_ratio = min(max(self.min_lr_ratio, 0.0), 1.0)

        LOGGER.info(
            "Using custom scheduler cosine_warmup_restarts | cycle_steps=%d | cycle_warmup_steps=%d | min_lr_ratio=%.5f",
            cycle_steps,
            warmup_steps,
            min_lr_ratio,
        )

        def lr_lambda(current_step: int) -> float:
            step_in_cycle = int(current_step) % cycle_steps

            if warmup_steps > 0 and step_in_cycle < warmup_steps:
                return float(step_in_cycle) / float(max(1, warmup_steps))

            decay_steps = max(1, cycle_steps - warmup_steps)
            progress = float(step_in_cycle - warmup_steps) / float(decay_steps)
            progress = min(max(progress, 0.0), 1.0)

            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return self.lr_scheduler


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    signature = inspect.signature(callable_obj)
    valid_keys = set(signature.parameters.keys())

    supported = {k: v for k, v in kwargs.items() if k in valid_keys}
    dropped = sorted(set(kwargs.keys()) - set(supported.keys()))
    return supported, dropped


def build_patchtst_model(args: argparse.Namespace) -> PatchTSTForPretraining:
    if args.model_name_or_path:
        LOGGER.info("Loading PatchTSTForPretraining from %s", args.model_name_or_path)
        return PatchTSTForPretraining.from_pretrained(args.model_name_or_path)

    config_kwargs = {
        "num_input_channels": args.num_input_channels,
        "context_length": args.context_length,
        "patch_length": args.patch_length,
        "patch_stride": args.patch_stride,
        "d_model": args.d_model,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "ffn_dim": args.ffn_dim,
        "dropout": args.dropout,
        "head_dropout": args.head_dropout,
        "mask_type": args.mask_type,
        "random_mask_ratio": args.random_mask_ratio,
        "num_forecast_mask_patches": args.num_forecast_mask_patches,
        "use_cls_token": args.use_cls_token,
    }

    supported_kwargs, dropped = _filter_supported_kwargs(PatchTSTConfig.__init__, config_kwargs)
    if dropped:
        LOGGER.warning("Ignoring unsupported PatchTSTConfig arguments: %s", dropped)

    config = PatchTSTConfig(**supported_kwargs)
    return PatchTSTForPretraining(config)


def build_model(args: argparse.Namespace):
    model_type = args.model_type.lower()
    if model_type == "patchtst":
        return build_patchtst_model(args)

    raise ValueError(
        f"Unsupported model_type='{args.model_type}'. "
        "Only 'patchtst' is currently implemented for pretraining in this script."
    )


def build_preprocessing_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "apply_preprocessing": args.apply_preprocessing,
        "preprocessing_steps": list(args.preprocessing_steps),
        "detrend_method": args.detrend_method,
        "notch_freq": args.notch_freq,
        "notch_bandwidth": args.notch_bandwidth,
        "bandpass_lowcut": args.bandpass_lowcut,
        "bandpass_highcut": args.bandpass_highcut,
        "filter_order": args.filter_order,
        "target_sampling_rate": args.target_sampling_rate,
        "normalize_per_lead": args.normalize_per_lead,
    }


def build_training_args(args: argparse.Namespace, has_eval: bool) -> TrainingArguments:
    # TrainingArguments only accepts built-in scheduler names.
    # For custom cyclic warmup+cosine restarts, we keep a valid placeholder here and override in Trainer.
    lr_scheduler_for_hf = args.lr_scheduler_type
    if lr_scheduler_for_hf == "cosine_warmup_restarts":
        lr_scheduler_for_hf = "cosine"

    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": lr_scheduler_for_hf,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "dataloader_num_workers": args.num_workers,
        "remove_unused_columns": False,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
        "report_to": [],
    }

    if args.lr_scheduler_type != "cosine_warmup_restarts":
        if args.warmup_steps > 0:
            kwargs["warmup_steps"] = args.warmup_steps
        else:
            kwargs["warmup_ratio"] = args.warmup_ratio

    if args.optimizer.lower().strip() != "muon":
        kwargs["optim"] = args.optimizer

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    if has_eval:
        if "evaluation_strategy" in ta_params:
            kwargs["evaluation_strategy"] = args.evaluation_strategy
        elif "eval_strategy" in ta_params:
            kwargs["eval_strategy"] = args.evaluation_strategy

        kwargs["load_best_model_at_end"] = args.load_best_model_at_end
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False
    else:
        if "evaluation_strategy" in ta_params:
            kwargs["evaluation_strategy"] = "no"
        elif "eval_strategy" in ta_params:
            kwargs["eval_strategy"] = "no"
        kwargs["load_best_model_at_end"] = False

    if "save_strategy" in ta_params:
        kwargs["save_strategy"] = args.save_strategy

    supported_kwargs, dropped = _filter_supported_kwargs(TrainingArguments.__init__, kwargs)
    if dropped:
        LOGGER.warning("Ignoring unsupported TrainingArguments arguments: %s", dropped)

    return TrainingArguments(**supported_kwargs)


def validate_records(
    hea_files: Sequence[Path],
    expected_num_channels: int,
    strict_num_input_channels: bool,
) -> list[Path]:
    usable: list[Path] = []

    for hea_path in hea_files:
        try:
            header = wfdb.rdheader(_record_base_path(hea_path))
        except Exception as exc:
            LOGGER.warning("Skipping unreadable record %s: %s", hea_path, exc)
            continue

        if strict_num_input_channels and int(header.n_sig) != expected_num_channels:
            LOGGER.warning(
                "Skipping %s because n_sig=%s (expected %s)",
                hea_path,
                header.n_sig,
                expected_num_channels,
            )
            continue

        usable.append(hea_path)

    return usable


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HF time-series pretraining for ECG (.hea) files. Default model: PatchTST."
    )

    # Data
    parser.add_argument("--data_dir", "--input_dir", dest="data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/hf_patchtst_pretraining")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    # Model selection
    parser.add_argument("--model_type", type=str, default="patchtst", choices=["patchtst"])
    parser.add_argument("--model_name_or_path", type=str, default=None)

    # Input shape / preprocessing
    parser.add_argument("--num_input_channels", type=int, default=12)
    parser.add_argument("--context_length", type=int, default=5000)
    parser.add_argument("--target_sampling_rate", type=int, default=250)
    parser.add_argument("--windows_per_record", type=int, default=1)
    parser.add_argument("--random_crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict_num_input_channels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_records", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--apply_preprocessing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preprocessing_steps",
        nargs="+",
        default=["notch", "bandpass", "resample"],
        help="Any subset of: detrend notch bandpass resample",
    )
    parser.add_argument("--detrend_method", type=str, default="linear")
    parser.add_argument(
        "--notch_freq",
        type=float,
        nargs="+",
        default=[50.0, 60.0],
        help="One or more notch center frequencies in Hz, e.g. --notch_freq 50 60",
    )
    parser.add_argument("--notch_bandwidth", type=float, default=1.0)
    parser.add_argument("--bandpass_lowcut", type=float, default=0.05)
    parser.add_argument("--bandpass_highcut", type=float, default=150.0)
    parser.add_argument("--filter_order", type=int, default=20)
    parser.add_argument("--normalize_per_lead", action=argparse.BooleanOptionalAction, default=True)

    # PatchTST config
    parser.add_argument("--patch_length", type=int, default=32)
    parser.add_argument("--patch_stride", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--mask_type", type=str, default="random", choices=["random", "forecast"])
    parser.add_argument("--random_mask_ratio", type=float, default=0.4)
    parser.add_argument("--num_forecast_mask_patches", type=int, default=64)
    parser.add_argument("--use_cls_token", action=argparse.BooleanOptionalAction, default=True)

    # Trainer args
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=float, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_torch",
        help="HF optimizer name (e.g. adamw_torch) or 'muon' for MUON optimizer.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
            "cosine_warmup_restarts",
        ],
        help=(
            "Scheduler type. Use 'cosine' for one warmup+cosine run, or "
            "'cosine_warmup_restarts' for repeated linear warmup -> cosine decay cycles."
        ),
    )
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument(
        "--lr_cycle_steps",
        type=int,
        default=0,
        help=(
            "Cycle length in optimizer steps for 'cosine_warmup_restarts'. "
            "If 0, uses all training steps as one cycle."
        ),
    )
    parser.add_argument(
        "--cycle_warmup_steps",
        type=int,
        default=0,
        help=(
            "Warmup steps per cycle for 'cosine_warmup_restarts'. "
            "If 0, uses --cycle_warmup_ratio."
        ),
    )
    parser.add_argument(
        "--cycle_warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio per cycle when --cycle_warmup_steps is 0.",
    )
    parser.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.0,
        help="Minimum LR multiplier reached at end of each cosine decay cycle.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"])
    parser.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    set_seed(args.seed)

    if args.lr_scheduler_type == "cosine_warmup_restarts":
        if args.lr_cycle_steps < 0:
            raise ValueError("--lr_cycle_steps must be >= 0")
        if args.cycle_warmup_steps < 0:
            raise ValueError("--cycle_warmup_steps must be >= 0")
        if args.cycle_warmup_ratio < 0:
            raise ValueError("--cycle_warmup_ratio must be >= 0")
        if not (0.0 <= args.lr_min_ratio <= 1.0):
            raise ValueError("--lr_min_ratio must be in [0, 1]")

    data_dir = Path(args.data_dir)
    hea_files = discover_hea_files(data_dir=data_dir, recursive=args.recursive)

    if args.max_records is not None:
        hea_files = hea_files[: args.max_records]

    if not hea_files:
        raise ValueError(f"No .hea files found in {data_dir}")

    LOGGER.info("Found %d candidate .hea files", len(hea_files))

    hea_files = validate_records(
        hea_files,
        expected_num_channels=args.num_input_channels,
        strict_num_input_channels=args.strict_num_input_channels,
    )

    if not hea_files:
        raise ValueError("No usable .hea records after validation")

    train_files, eval_files = split_train_eval(hea_files, val_ratio=args.val_ratio, seed=args.seed)
    LOGGER.info("Using %d train records and %d eval records", len(train_files), len(eval_files))

    train_dataset = ECGPretrainingDataset(
        train_files,
        context_length=args.context_length,
        num_input_channels=args.num_input_channels,
        target_sampling_rate=args.target_sampling_rate,
        windows_per_record=args.windows_per_record,
        random_crop=args.random_crop,
        strict_num_input_channels=args.strict_num_input_channels,
        apply_preprocessing=args.apply_preprocessing,
        preprocessing_steps=args.preprocessing_steps,
        detrend_method=args.detrend_method,
        notch_freq=args.notch_freq,
        notch_bandwidth=args.notch_bandwidth,
        bandpass_lowcut=args.bandpass_lowcut,
        bandpass_highcut=args.bandpass_highcut,
        filter_order=args.filter_order,
        normalize_per_lead=args.normalize_per_lead,
        cache_records=args.cache_records,
    )

    eval_dataset = None
    if eval_files:
        eval_dataset = ECGPretrainingDataset(
            eval_files,
            context_length=args.context_length,
            num_input_channels=args.num_input_channels,
            target_sampling_rate=args.target_sampling_rate,
            windows_per_record=1,
            random_crop=False,
            strict_num_input_channels=args.strict_num_input_channels,
            apply_preprocessing=args.apply_preprocessing,
            preprocessing_steps=args.preprocessing_steps,
            detrend_method=args.detrend_method,
            notch_freq=args.notch_freq,
            notch_bandwidth=args.notch_bandwidth,
            bandpass_lowcut=args.bandpass_lowcut,
            bandpass_highcut=args.bandpass_highcut,
            filter_order=args.filter_order,
            normalize_per_lead=args.normalize_per_lead,
            cache_records=args.cache_records,
        )

    model = build_model(args)

    # Persist effective preprocessing settings into model config.json
    preprocessing_cfg = build_preprocessing_config(args)
    if getattr(model, "config", None) is not None:
        model.config.timex_preprocessing = preprocessing_cfg
        LOGGER.info("Attached preprocessing settings to model config under key: timex_preprocessing")

    training_args = build_training_args(args, has_eval=eval_dataset is not None)
    collator = ECGPretrainingCollator()

    warmup_desc = (
        f"{args.warmup_steps} steps" if args.warmup_steps > 0 else f"{args.warmup_ratio:.4f} ratio"
    )
    if args.lr_scheduler_type == "cosine_warmup_restarts":
        warmup_desc = (
            f"per-cycle {args.cycle_warmup_steps} steps"
            if args.cycle_warmup_steps > 0
            else f"per-cycle {args.cycle_warmup_ratio:.4f} ratio"
        )

    LOGGER.info(
        "Optimizer=%s | Scheduler=%s | Warmup=%s",
        args.optimizer,
        args.lr_scheduler_type,
        warmup_desc,
    )

    callbacks = []
    if eval_dataset is not None and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = ECGPretrainingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        optimizer_name=args.optimizer,
        scheduler_name=args.lr_scheduler_type,
        cycle_steps=args.lr_cycle_steps,
        cycle_warmup_steps=args.cycle_warmup_steps,
        cycle_warmup_ratio=args.cycle_warmup_ratio,
        min_lr_ratio=args.lr_min_ratio,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(args.output_dir)
    trainer.save_state()

    LOGGER.info("Finished pretraining. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    parser = make_arg_parser()
    main(parser.parse_args())
