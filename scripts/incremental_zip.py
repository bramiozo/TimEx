"""Incrementally zip subfolders containing .hea files.

Usage:
    python scripts/incremental_zip.py --input_folder <in> --output_folder <out> [--continue] [--dry_run]

Behavior:
- Finds all folders under --input_folder that directly contain one or more `.hea` files.
- Creates one zip per such folder in --output_folder.
- Shows per-file progress while adding files to each zip.
- Zip names are based on the folder path relative to --input_folder, with path
  separators replaced by `-`.
- With --continue, skips folders whose target zip already exists.
- With --dry_run, prints planned actions without creating archives.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally compress subfolders containing .hea files."
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        type=Path,
        help="Root folder to scan recursively.",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        type=Path,
        help="Folder where zip files are written.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_mode",
        action="store_true",
        help="Skip archives that already exist in output_folder.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be zipped/skipped without creating files.",
    )
    return parser.parse_args()


def find_folders_with_hea(root: Path) -> list[Path]:
    """Find folders that directly contain one or more .hea files.

    Prints periodic scan progress so long scans don't appear stuck.
    """
    folders: list[Path] = []
    scanned_dirs = 0

    for dirpath, _dirnames, filenames in os.walk(root):
        scanned_dirs += 1

        if any(Path(name).suffix.lower() == ".hea" for name in filenames):
            folders.append(Path(dirpath))

        if scanned_dirs % 500 == 0:
            print(
                f"\rScanning folders... {scanned_dirs} checked, {len(folders)} candidates",
                end="",
                flush=True,
            )

    if scanned_dirs >= 500:
        print(
            f"\rScanning folders... {scanned_dirs} checked, {len(folders)} candidates",
            flush=True,
        )

    # Stable ordering (shorter paths first, then lexical)
    folders = sorted(set(folders), key=lambda p: (len(p.parts), str(p).lower()))
    return folders


def sanitize_zip_stem(relative_folder: Path) -> str:
    """Create a filesystem-safe stem from a relative path."""
    if str(relative_folder) in {"", "."}:
        base = "root"
    else:
        base = "-".join(relative_folder.parts)

    # Keep letters, digits, dots, hyphens, underscores; replace others with '_'
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def target_zip_path(folder: Path, input_root: Path, output_folder: Path) -> Path:
    rel = folder.relative_to(input_root)
    stem = sanitize_zip_stem(rel)
    return output_folder / f"{stem}.zip"


def filetype_of(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix else "[no_ext]"


def zip_folder(folder: Path, zip_path: Path) -> tuple[int, Counter[str]]:
    """Zip an entire folder, preserving that folder as top-level inside archive.

    Returns the number of files added and a filetype counter.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        (p for p in folder.rglob("*") if p.is_file()), key=lambda p: str(p).lower()
    )
    total_files = len(files)

    if total_files == 0:
        print("    No files to add.", flush=True)
        return 0, Counter()

    filetypes = Counter(filetype_of(p) for p in files)

    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for file_idx, file in enumerate(files, start=1):
            # include folder name in archive
            arcname = file.relative_to(folder.parent)
            zf.write(file, arcname)

            progress = render_progress(file_idx, total_files, label="Files")
            print(f"\r    {progress}", end="", flush=True)

    print("", flush=True)
    return total_files, filetypes


def render_progress(
    done: int, total: int, width: int = 30, label: str = "Progress"
) -> str:
    if total <= 0:
        return f"{label}: [------------------------------] 0/0 (0%)"

    ratio = done / total
    filled = min(width, int(round(ratio * width)))
    bar = "#" * filled + "-" * (width - filled)
    percent = int(round(ratio * 100))
    return f"{label}: [{bar}] {done}/{total} ({percent}%)"


def main() -> None:
    args = parse_args()

    input_root = args.input_folder.resolve()
    output_folder = args.output_folder.resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(
            f"Input folder does not exist or is not a directory: {input_root}"
        )

    if not args.dry_run:
        output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for .hea folders under: {input_root}", flush=True)
    targets = find_folders_with_hea(input_root)

    if not targets:
        print("No target folders found (.hea in subfolders).", flush=True)
        return

    total = len(targets)
    print(f"Found {total} target folder(s).", flush=True)
    print(render_progress(0, total, label="Folders"), flush=True)

    created = 0
    skipped = 0
    total_filetypes: Counter[str] = Counter()

    for idx, folder in enumerate(targets, start=1):
        zip_path = target_zip_path(folder, input_root, output_folder)

        if args.continue_mode and zip_path.exists():
            skipped += 1
            print(f"[{idx}/{total}] Skip existing: {zip_path.name}", flush=True)
            print(render_progress(idx, total, label="Folders"), flush=True)
            continue

        if args.dry_run:
            print(f"[{idx}/{total}] Would zip: {folder} -> {zip_path.name}", flush=True)
            created += 1
            print(render_progress(idx, total, label="Folders"), flush=True)
            continue

        print(f"[{idx}/{total}] Zipping: {folder}", flush=True)
        _, filetypes = zip_folder(folder, zip_path)
        total_filetypes.update(filetypes)
        created += 1
        print(render_progress(idx, total, label="Folders"), flush=True)

    mode = "Dry run complete" if args.dry_run else "Done"
    print(f"{mode}. Created: {created}, Skipped: {skipped}, Total targets: {total}")

    if args.dry_run:
        print(
            "Filetype overview skipped in dry-run mode (no files were added).",
            flush=True,
        )
        return

    if total_filetypes:
        print("Filetype overview (added files):", flush=True)
        for ext, count in sorted(
            total_filetypes.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            print(f"  {ext}: {count}", flush=True)


if __name__ == "__main__":
    main()
