from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from transformers import set_seed

from timex.ecg.hf_pretraining import (
    ECGPretrainingDataset,
    context_length_ms_to_samples,
    discover_hea_files,
    split_train_eval,
    validate_records,
)

LOGGER = logging.getLogger(__name__)


def _build_preprocessing_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "apply_preprocessing": args.apply_preprocessing,
        "preprocessing_steps": list(args.preprocessing_steps),
        "detrend_method": args.detrend_method,
        "notch_freq": list(args.notch_freq),
        "notch_bandwidth": args.notch_bandwidth,
        "bandpass_lowcut": args.bandpass_lowcut,
        "bandpass_highcut": args.bandpass_highcut,
        "filter_order": args.filter_order,
        "target_sampling_rate": args.target_sampling_rate,
        "normalize_per_lead": args.normalize_per_lead,
    }


def _write_window_sources(dataset: ECGPretrainingDataset, out_path: Path) -> None:
    windows_per_record = int(dataset.windows_per_record)
    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(len(dataset)):
            record_idx = idx // windows_per_record
            f.write(str(dataset.hea_files[record_idx]))
            f.write("\n")


def write_split_memmap(
    *,
    split: str,
    dataset: ECGPretrainingDataset,
    output_dir: Path,
    context_length: int,
    num_input_channels: int,
) -> dict[str, Any]:
    n_windows = len(dataset)
    shape = (n_windows, context_length, num_input_channels)

    if n_windows == 0:
        return {
            f"{split}_past_values_file": None,
            f"{split}_past_values_shape": [0, context_length, num_input_channels],
            f"{split}_past_observed_mask_file": None,
            f"{split}_past_observed_mask_shape": [0, context_length, num_input_channels],
        }

    values_file = f"{split}_past_values.dat"
    mask_file = f"{split}_past_observed_mask.dat"

    values_path = output_dir / values_file
    mask_path = output_dir / mask_file

    values_mm = np.memmap(values_path, mode="w+", dtype=np.float32, shape=shape)
    mask_mm = np.memmap(mask_path, mode="w+", dtype=np.float32, shape=shape)

    for i in tqdm(range(n_windows), desc=f"Writing {split} memmap", unit="window"):
        item = dataset[i]
        values_mm[i] = item["past_values"].numpy()
        mask_mm[i] = item["past_observed_mask"].numpy()

    values_mm.flush()
    mask_mm.flush()

    del values_mm
    del mask_mm

    return {
        f"{split}_past_values_file": values_file,
        f"{split}_past_values_shape": list(shape),
        f"{split}_past_observed_mask_file": mask_file,
        f"{split}_past_observed_mask_shape": list(shape),
    }


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute ECG pretraining windows into memmap format for fast I/O during HF PatchTST pretraining."
        )
    )

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--num_input_channels", type=int, default=12)
    parser.add_argument("--context_length_ms", type=int, default=3000)
    parser.add_argument(
        "--context_length",
        type=int,
        default=None,
        help="Deprecated: context length in samples/ticks. Prefer --context_length_ms.",
    )
    parser.add_argument("--target_sampling_rate", type=int, default=250)

    parser.add_argument("--train_windows_per_record", type=int, default=1)
    parser.add_argument("--eval_windows_per_record", type=int, default=1)
    parser.add_argument("--train_random_crop", action=argparse.BooleanOptionalAction, default=True)

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
    parser.add_argument("--bandpass_highcut", type=float, default=100.0)
    parser.add_argument("--filter_order", type=int, default=20)
    parser.add_argument("--normalize_per_lead", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_records", action=argparse.BooleanOptionalAction, default=True)

    return parser


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    set_seed(args.seed)

    if args.train_windows_per_record <= 0:
        raise ValueError("--train_windows_per_record must be > 0")
    if args.eval_windows_per_record <= 0:
        raise ValueError("--eval_windows_per_record must be > 0")

    if args.context_length is not None:
        if args.context_length <= 0:
            raise ValueError("--context_length must be > 0")
        LOGGER.warning("--context_length is deprecated; prefer --context_length_ms")
        args.context_length = int(args.context_length)
        args.context_length_ms = int(round(1000.0 * args.context_length / float(args.target_sampling_rate)))
    else:
        args.context_length = context_length_ms_to_samples(
            context_length_ms=int(args.context_length_ms),
            target_sampling_rate=int(args.target_sampling_rate),
        )

    LOGGER.info(
        "Preparing memmap with context_length=%d samples (from %d ms at %d Hz)",
        args.context_length,
        args.context_length_ms,
        args.target_sampling_rate,
    )

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise ValueError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to allow rewriting files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    hea_files = discover_hea_files(data_dir=data_dir, recursive=args.recursive)

    if args.max_records is not None:
        hea_files = hea_files[: args.max_records]

    if not hea_files:
        raise ValueError(f"No .hea files found in {data_dir}")

    LOGGER.info("Found %d candidate .hea files", len(hea_files))

    hea_files = validate_records(hea_files, expected_num_channels=args.num_input_channels)
    if not hea_files:
        raise ValueError("No usable .hea records after validation")

    train_files, eval_files = split_train_eval(hea_files, val_ratio=args.val_ratio, seed=args.seed)
    LOGGER.info("Using %d train records and %d eval records", len(train_files), len(eval_files))

    train_dataset = ECGPretrainingDataset(
        train_files,
        context_length=args.context_length,
        num_input_channels=args.num_input_channels,
        target_sampling_rate=args.target_sampling_rate,
        windows_per_record=args.train_windows_per_record,
        random_crop=args.train_random_crop,
        strict_num_input_channels=True,
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

    eval_dataset = ECGPretrainingDataset(
        eval_files,
        context_length=args.context_length,
        num_input_channels=args.num_input_channels,
        target_sampling_rate=args.target_sampling_rate,
        windows_per_record=args.eval_windows_per_record,
        random_crop=False,
        strict_num_input_channels=True,
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

    train_meta = write_split_memmap(
        split="train",
        dataset=train_dataset,
        output_dir=output_dir,
        context_length=args.context_length,
        num_input_channels=args.num_input_channels,
    )
    eval_meta = write_split_memmap(
        split="eval",
        dataset=eval_dataset,
        output_dir=output_dir,
        context_length=args.context_length,
        num_input_channels=args.num_input_channels,
    )

    _write_window_sources(train_dataset, output_dir / "train_window_sources.txt")
    _write_window_sources(eval_dataset, output_dir / "eval_window_sources.txt")

    metadata = {
        "format_version": 1,
        "dtype": "float32",
        "context_length": args.context_length,
        "context_length_ms": args.context_length_ms,
        "num_input_channels": args.num_input_channels,
        "target_sampling_rate": args.target_sampling_rate,
        "train_records": len(train_files),
        "eval_records": len(eval_files),
        "train_windows": len(train_dataset),
        "eval_windows": len(eval_dataset),
        "preprocessing": _build_preprocessing_config(args),
        **train_meta,
        **eval_meta,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info("Memmap prep finished. Output written to %s", output_dir)


if __name__ == "__main__":
    parser = make_arg_parser()
    main(parser.parse_args())
