"""
data_download.py
Uploads HR_Analytics_Data.csv from the repo's data/ folder to S3.

File location in repo:
    data/HR_Analytics_Data.csv   <-- commit this file to your repo

Script location:
    src/data/data_download.py

Path resolution: __file__ -> src/data/ -> repo_root -> data/HR_Analytics_Data.csv
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError

# ── Config from environment ──────────────────────────────────────────────────
S3_BUCKET   = os.environ["S3_BUCKET"]
PROJECT     = os.environ.get("PROJECT_NAME", "hr-attrition")
RAW_S3_KEY  = f"{PROJECT}/data/raw/HR_Analytics_Data.csv"

# ── Resolve CSV path relative to THIS script ────────────────────────────────
# __file__ = /home/runner/work/<repo>/<repo>/src/data/data_download.py
# repo_root = /home/runner/work/<repo>/<repo>/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src/data/
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # .../
CSV_PATH   = os.path.join(REPO_ROOT, "data", "HR_Analytics_Data.csv")


def upload_to_s3(local_file: str, bucket: str, s3_key: str) -> None:
    if not os.path.exists(local_file):
        print(f"ERROR: Local file not found: {local_file}")
        print(f"       Make sure 'data/HR_Analytics_Data.csv' is committed to your repo.")
        sys.exit(1)

    size_mb = os.path.getsize(local_file) / (1024 * 1024)
    print(f"  File   : {local_file}  ({size_mb:.2f} MB)")
    print(f"  Target : s3://{bucket}/{s3_key}")

    s3 = boto3.client("s3")
    try:
        s3.upload_file(local_file, bucket, s3_key)
        print(f"  Upload complete.")
    except ClientError as e:
        print(f"ERROR: S3 upload failed: {e}")
        sys.exit(1)


def verify_upload(bucket: str, s3_key: str) -> None:
    s3 = boto3.client("s3")
    try:
        resp = s3.head_object(Bucket=bucket, Key=s3_key)
        size_kb = resp["ContentLength"] / 1024
        print(f"  Verified in S3: s3://{bucket}/{s3_key}  ({size_kb:.1f} KB)")
    except ClientError:
        print("ERROR: Verification failed — object not found in S3.")
        sys.exit(1)


def ensure_s3_folders(bucket: str, project: str) -> None:
    s3 = boto3.client("s3")
    for folder in ["data/raw/", "data/processed/", "models/", "evaluation/", "deployment/", "monitoring/", "code/"]:
        s3.put_object(Bucket=bucket, Key=f"{project}/{folder}")


def main():
    print("\n=== 1-Data-Download: HR Analytics ===")
    print(f"  Script dir : {SCRIPT_DIR}")
    print(f"  Repo root  : {REPO_ROOT}")
    print(f"  CSV path   : {CSV_PATH}")
    print(f"  S3 Bucket  : {S3_BUCKET}")
    print(f"  S3 Key     : {RAW_S3_KEY}")

    ensure_s3_folders(S3_BUCKET, PROJECT)
    upload_to_s3(CSV_PATH, S3_BUCKET, RAW_S3_KEY)
    verify_upload(S3_BUCKET, RAW_S3_KEY)

    print("=== Upload complete ===\n")


if __name__ == "__main__":
    main()
