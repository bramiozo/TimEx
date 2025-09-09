import os, sys, argparse
from pathlib import Path
from typing import Optional, Iterable

from dotenv import load_dotenv
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError



def make_s3(region: str, ak=None, sk=None, st=None):
    cfg = Config(
        signature_version="s3v4",
        s3={"addressing_style": "virtual", "use_arn_region": True},
        region_name=region,
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

def download_list_mode(s3, bucket_arn: str, output: Path, prefix: Optional[str]):
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket_arn, 
              "RequestPayer": "requester",
              "Prefix": prefix or ""}
    total = 0
    # sanity probe
    print("Probing..")
    s3.list_objects_v2(Bucket=bucket_arn, MaxKeys=1, RequestPayer="requester")
    print("---success---")
    print("Continuing with pagination...")
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            dest = output / key
            ensure_dir(dest)
            print(f"Downloading: {key} -> {dest}")
            s3.download_file(bucket_arn, 
                             key, 
                             str(dest),
                             ExtraArgs={"RequestPayer": "requester"})
            total += 1
    if total == 0:
        print("No objects found (check prefix/permissions).")

def download_keys_mode(s3, bucket_arn: str, output: Path, keys_file: Path, strip_prefix: Optional[str]):
    total = 0
    for key in iter_keys_from_file(keys_file):
        use_key = key[len(strip_prefix):] if strip_prefix and key.startswith(strip_prefix) else key
        dest = output / use_key
        ensure_dir(dest)
        try:
            print(f"Downloading: {key} -> {dest}")
            s3.download_file(bucket_arn, 
                             key, 
                             str(dest),
                             ExtraArgs={"RequestPayer": "requester"})
            total += 1
        except ClientError as e:
            print(f"ERROR for {key}: {e}", file=sys.stderr)
    if total == 0:
        print("No objects downloaded. Check keys/permissions.")

def main():
    # Load .env file if present
    load_dotenv()

    p = argparse.ArgumentParser(description="Download via S3 Access Point. Supports list mode and keys-file (GetObject-only) mode.")
    p.add_argument("--bucket-arn", required=False, default=os.getenv("AWS_DEFAULT_ARN", None))
    p.add_argument("--output", required=True)
    p.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--prefix", default=None, help="Prefix for list mode.")

    # keys-file mode (skips ListObjectsV2)
    p.add_argument("--keys-file", help="Path to newline-separated object keys.")
    p.add_argument("--strip-prefix", help="If provided, strip this prefix from destination paths in keys-file mode.")

    # optional assume-role (if they give you a role later)
    p.add_argument("--assume-role-arn")
    p.add_argument("--external-id")

    args = p.parse_args()

    # base creds from .env
    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")
    st = os.getenv("AWS_SESSION_TOKEN")
    if not ak or not sk:
        print("ERROR: missing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in env/.env", file=sys.stderr)
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
        params = {"RoleArn": args.assume_role_arn, "RoleSessionName": "s3-download-session"}
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
            download_keys_mode(s3, args.bucket_arn, output, Path(args.keys_file), args.strip_prefix)
        else:
            download_list_mode(s3, args.bucket_arn, output, args.prefix)
    except ClientError as e:
        print(f"\nAWS ClientError: {e}", file=sys.stderr)
        code = e.response.get("Error", {}).get("Code")
        if code == "AccessDenied":
            print("Access denied. If you can’t change the other account, use keys-file mode or request access.", file=sys.stderr)
            print(f"Full error response: {e.response}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
