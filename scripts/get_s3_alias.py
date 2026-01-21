import boto3
import os, sys
from dotenv import load_dotenv

def whoami(ak=None, sk=None, st=None, region="us-east-1"):
    sts = boto3.client(
        "sts",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        aws_session_token=st,
        region_name=region,
    )
    return sts.get_caller_identity()

load_dotenv()
ak = os.getenv("AWS_ACCESS_KEY_ID")
sk = os.getenv("AWS_SECRET_ACCESS_KEY")
st = os.getenv("AWS_SESSION_TOKEN")
if not ak or not sk:
    print("ERROR: missing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in env/.env", file=sys.stderr)
    sys.exit(2)

# warn if caller account != access point account
print("Getting identity for sanity check...")

try:
    ident = whoami(ak, sk, st, "us-east-1")
    print(f"Caller identity: {ident['Arn']} (Account {ident['Account']})")
except Exception as e:
    print(f"ERROR verifying identity: {e}", file=sys.stderr)
    sys.exit(2)

s3c = boto3.client("s3control", region_name="us-east-1")
# bdsp-credentialed-access-point bdsp-ecg-accesspoint
resp = s3c.get_access_point(AccountId="184438910517", Name="bdsp-credentialed-access-point")
print(resp["Alias"])
