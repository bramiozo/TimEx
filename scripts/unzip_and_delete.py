"""Unzip archives recursively and delete each .zip only after successful extraction.

Usage:
    python scripts/unzip_and_delete.py --zip_folder <zip_root> [--output_folder <out_root>]

Behavior:
- Scans --zip_folder recursively for .zip files (case-insensitive).
- Extracts each archive to --output_folder while preserving relative folder structure.
- Uses a subfolder named after the zip filename stem for each extraction target.
- Deletes the source .zip only if extraction completes successfully.

If --output_folder is omitted, defaults to:
    <zip_folder>/unzipped
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import BadZipFile, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively unzip files and delete .zip only on success."
    )
    parser.add_argument(
        "--zip_folder",
        required=True,
        type=Path,
        help="Folder to scan recursively for .zip files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="",
        help=(
            "Extraction root folder. If omitted or empty, defaults to "
            "<zip_folder>/unzipped."
        ),
    )
    return parser.parse_args()


def find_zip_files(zip_root: Path) -> list[Path]:
    return sorted(
        (p for p in zip_root.rglob("*") if p.is_file() and p.suffix.lower() == ".zip"),
        key=lambda p: str(p).lower(),
    )


def extract_target(zip_path: Path, zip_root: Path, output_root: Path) -> Path:
    relative_parent = zip_path.parent.relative_to(zip_root)
    return output_root / relative_parent / zip_path.stem


def unzip_archive(zip_path: Path, target_dir: Path) -> tuple[bool, str | None]:
    try:
        with ZipFile(zip_path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                return False, f"Archive integrity check failed at member: {bad_member}"

            target_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(target_dir)
        return True, None
    except BadZipFile:
        return False, "Not a valid zip file"
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    args = parse_args()

    zip_root = args.zip_folder.resolve()
    output_value = args.output_folder.strip() if args.output_folder else ""
    output_root = (
        Path(output_value).resolve() if output_value else (zip_root / "unzipped").resolve()
    )

    if not zip_root.exists() or not zip_root.is_dir():
        raise SystemExit(f"zip_folder does not exist or is not a directory: {zip_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    zip_files = find_zip_files(zip_root)
    if not zip_files:
        print(f"No .zip files found under: {zip_root}")
        return

    print(f"Found {len(zip_files)} zip file(s) under: {zip_root}")
    print(f"Output root: {output_root}")

    extracted_ok = 0
    deleted_ok = 0
    failed_extract = 0
    failed_delete = 0

    total = len(zip_files)
    for idx, zip_path in enumerate(zip_files, start=1):
        target_dir = extract_target(zip_path, zip_root, output_root)
        print(f"[{idx}/{total}] Extracting: {zip_path} -> {target_dir}")

        success, error = unzip_archive(zip_path, target_dir)
        if not success:
            failed_extract += 1
            print(f"    Extraction failed. Keeping zip. Reason: {error}")
            continue

        extracted_ok += 1

        try:
            zip_path.unlink()
            deleted_ok += 1
            print("    Extraction successful. Zip deleted.")
        except OSError as exc:
            failed_delete += 1
            print(f"    Extraction successful, but deletion failed: {exc}")

    print("\nDone.")
    print(f"  Total zip files: {total}")
    print(f"  Extracted successfully: {extracted_ok}")
    print(f"  Failed extraction: {failed_extract}")
    print(f"  Deleted after extraction: {deleted_ok}")
    print(f"  Failed deletion after extraction: {failed_delete}")


if __name__ == "__main__":
    main()
