"""Incrementally zip lowest-level subfolders containing .hea files.

Usage:
    python scripts/incremental_zip.py --input_folder <in> --output_folder <out> [--continue] [--dry_run]

Behavior:
- Finds all folders under --input_folder that directly contain one or more `.hea` files.
- Keeps only the *lowest-level* such folders (i.e. excludes any folder that has a
  descendant folder that also contains `.hea` files).
- Creates one zip per selected folder in --output_folder.
- Zip names are based on the folder path relative to --input_folder, with path
  separators replaced by `-`.
- With --continue, skips folders whose target zip already exists.
- With --dry_run, prints planned actions without creating archives.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally compress lowest-level subfolders containing .hea files."
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


def is_ancestor(ancestor: Path, maybe_descendant: Path) -> bool:
    try:
        maybe_descendant.relative_to(ancestor)
        return ancestor != maybe_descendant
    except ValueError:
        return False


def filter_lowest_level_folders(folders: list[Path]) -> list[Path]:
    """Keep only folders that do not have descendants in the same set."""
    kept: list[Path] = []

    for folder in folders:
        has_descendant = any(
            is_ancestor(folder, other) for other in folders if other != folder
        )
        if not has_descendant:
            kept.append(folder)

    return sorted(kept, key=lambda p: str(p).lower())


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


def zip_folder(folder: Path, zip_path: Path) -> None:
    """Zip an entire folder, preserving that folder as top-level inside archive."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for file in folder.rglob("*"):
            if file.is_file():
                # include folder name in archive
                arcname = file.relative_to(folder.parent)
                zf.write(file, arcname)


def render_progress(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "Progress: [------------------------------] 0/0 (0%)"

    ratio = done / total
    filled = min(width, int(round(ratio * width)))
    bar = "#" * filled + "-" * (width - filled)
    percent = int(round(ratio * 100))
    return f"Progress: [{bar}] {done}/{total} ({percent}%)"


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
    all_candidates = find_folders_with_hea(input_root)
    targets = filter_lowest_level_folders(all_candidates)

    if not targets:
        print("No target folders found (.hea in lowest-level subfolders).", flush=True)
        return

    total = len(targets)
    print(f"Found {total} target folder(s).", flush=True)
    print(render_progress(0, total), flush=True)

    created = 0
    skipped = 0

    for idx, folder in enumerate(targets, start=1):
        zip_path = target_zip_path(folder, input_root, output_folder)

        if args.continue_mode and zip_path.exists():
            skipped += 1
            print(f"[{idx}/{total}] Skip existing: {zip_path.name}", flush=True)
            print(render_progress(idx, total), flush=True)
            continue

        if args.dry_run:
            print(f"[{idx}/{total}] Would zip: {folder} -> {zip_path.name}", flush=True)
            created += 1
            print(render_progress(idx, total), flush=True)
            continue

        print(f"[{idx}/{total}] Zipping: {folder}", flush=True)
        zip_folder(folder, zip_path)
        created += 1
        print(render_progress(idx, total), flush=True)

    mode = "Dry run complete" if args.dry_run else "Done"
    print(f"{mode}. Created: {created}, Skipped: {skipped}, Total targets: {total}")


if __name__ == "__main__":
    main()
