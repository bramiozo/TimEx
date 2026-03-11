import argparse
import os
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Iterable, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv


def make_s3(region: str, ak=None, sk=None, st=None):
    cfg = Config(
        signature_version="s3v4",
        s3={"addressing_style": "virtual", "use_arn_region": True},
        region_name=region,
        max_pool_connections=200,
    )
    return boto3.client(
        "s3",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        aws_session_token=st,
        config=cfg,
        region_name=region,
    )


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def iter_keys_from_file(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def whoami(ak=None, sk=None, st=None, region="us-east-1"):
    sts = boto3.client(
        "sts",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        aws_session_token=st,
        region_name=region,
    )
    return sts.get_caller_identity()


def download_list_mode(
    s3,
    bucket_arn: str,
    output: Path,
    prefix: Optional[str],
    workers: int = 64,
    max_in_flight: int = 512,
):
    """
    List objects with pagination and download them concurrently.

    - workers: number of download threads
    - max_in_flight: cap on pending futures to avoid huge memory use on millions of keys
    """
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {
        "Bucket": bucket_arn,
        "RequestPayer": "requester",
        "Prefix": prefix or "",
    }

    # sanity probe
    print("Probing..")
    s3.list_objects_v2(Bucket=bucket_arn, MaxKeys=1, RequestPayer="requester")
    print("---success---")
    print(
        f"Continuing with pagination... (workers={workers}, max_in_flight={max_in_flight})"
    )

    total_listed = 0
    submitted = 0
    downloaded = 0
    skipped = 0
    errors = 0

    lock = threading.Lock()
    in_flight = set()

    def _download_one(key: str, dest: Path):
        nonlocal downloaded, skipped, errors
        try:
            # double-check exists (race-safe-ish)
            if dest.exists():
                with lock:
                    skipped += 1
                return

            ensure_dir(dest)
            s3.download_file(
                bucket_arn,
                key,
                str(dest),
                ExtraArgs={"RequestPayer": "requester"},
            )
            with lock:
                downloaded += 1
        except Exception as e:
            with lock:
                errors += 1
            print(f"ERROR downloading {key}: {e}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                dest = output / key
                total_listed += 1

                # quick skip without scheduling work
                if dest.exists():
                    skipped += 1
                    continue

                # throttle: don’t let too many futures accumulate
                while len(in_flight) >= max_in_flight:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    # propagate exceptions (already handled in _download_one, but keep safe)
                    for f in done:
                        _ = f.result()

                fut = ex.submit(_download_one, key, dest)
                in_flight.add(fut)
                submitted += 1

                if submitted % 1000 == 0:
                    with lock:
                        print(
                            f"listed={total_listed} submitted={submitted} "
                            f"downloaded={downloaded} skipped={skipped} errors={errors}"
                        )

        # finish remaining
        if in_flight:
            done, _pending = wait(in_flight)
            for f in done:
                _ = f.result()

    if total_listed == 0:
        print("No objects found (check prefix/permissions).")
    else:
        print(
            f"Done. listed={total_listed} submitted={submitted} "
            f"downloaded={downloaded} skipped={skipped} errors={errors}"
        )


def download_keys_mode(s3, bucket_arn, output, keys_file, strip_prefix, workers=64):
    keys = list(iter_keys_from_file(keys_file))

    def _download(key):
        use_key = (
            key[len(strip_prefix) :]
            if strip_prefix and key.startswith(strip_prefix)
            else key
        )
        dest = output / use_key
        ensure_dir(dest)
        if not dest.exists():
            s3.download_file(
                bucket_arn,
                key,
                str(dest),
                ExtraArgs={"RequestPayer": "requester"},
            )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_download, k) for k in keys]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)


def main():
    # Load .env file if present
    load_dotenv()

    # Remove any AWS_PROFILE / AWS_DEFAULT_PROFILE so boto3 doesn't try to
    # resolve a named profile from ~/.aws/config – we supply explicit creds.
    os.environ.pop("AWS_PROFILE", None)
    os.environ.pop("AWS_DEFAULT_PROFILE", None)

    p = argparse.ArgumentParser(
        description="Download via S3 Access Point. Supports list mode and keys-file (GetObject-only) mode."
    )
    p.add_argument(
        "--bucket-arn", required=False, default=os.getenv("AWS_DEFAULT_ARN", None)
    )
    p.add_argument("--output", required=True)
    p.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--prefix", default=None, help="Prefix for list mode.")

    # keys-file mode (skips ListObjectsV2)
    p.add_argument("--keys-file", help="Path to newline-separated object keys.")
    p.add_argument(
        "--strip-prefix",
        help="If provided, strip this prefix from destination paths in keys-file mode.",
    )

    # optional assume-role (if they give you a role later)
    p.add_argument("--assume-role-arn")
    p.add_argument("--external-id")

    args = p.parse_args()

    # base creds from .env
    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")
    st = os.getenv("AWS_SESSION_TOKEN")
    if not ak or not sk:
        print(
            "ERROR: missing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in env/.env",
            file=sys.stderr,
        )
        sys.exit(2)

    # warn if caller account != access point account
    print("Getting identity for sanity check...")

    try:
        ident = whoami(ak, sk, st, args.region)
        print(f"Caller identity: {ident['Arn']} (Account {ident['Account']})")
    except Exception as e:
        print(f"ERROR verifying identity: {e}", file=sys.stderr)
        sys.exit(2)

    # optional assume role (won't work until other org sets it up)
    if args.assume_role_arn:
        sts = boto3.client("sts", region_name=args.region)
        params = {
            "RoleArn": args.assume_role_arn,
            "RoleSessionName": "s3-download-session",
        }
        if args.external_id:
            params["ExternalId"] = args.external_id
        resp = sts.assume_role(**params)
        ak = resp["Credentials"]["AccessKeyId"]
        sk = resp["Credentials"]["SecretAccessKey"]
        st = resp["Credentials"]["SessionToken"]
        ident2 = whoami(ak, sk, st, args.region)
        print(f"Assumed identity: {ident2['Arn']} (Account {ident2['Account']})")

    s3 = make_s3(args.region, ak, sk, st)
    output = Path(args.output)

    try:
        if args.keys_file:
            download_keys_mode(
                s3, args.bucket_arn, output, Path(args.keys_file), args.strip_prefix
            )
        else:
            download_list_mode(s3, args.bucket_arn, output, args.prefix)
    except ClientError as e:
        print(f"\nAWS ClientError: {e}", file=sys.stderr)
        code = e.response.get("Error", {}).get("Code")
        if code == "AccessDenied":
            print(
                "Access denied. If you can’t change the other account, use keys-file mode or request access.",
                file=sys.stderr,
            )
            print(f"Full error response: {e.response}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
