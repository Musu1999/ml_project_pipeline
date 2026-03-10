"""
data_download.py
Upload HR Analytics CSV to S3 raw/ directory.

FIXED BEHAVIOUR:
- If S3 already has a CSV with MORE rows than the local repo file → keep S3 file
- If S3 is empty or local file is larger → upload local file
- This prevents the pipeline from overwriting a manually updated S3 dataset
  with the older smaller file stored in the repo.

Usage (local):
    pip install boto3
    export S3_BUCKET=term3-mousumi
    export PROJECT_NAME=mousumi-hr-attrition
    python src/data/data_download.py
"""

import io
import os
import sys
import glob
import boto3
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET  = os.environ["S3_BUCKET"]
PROJECT    = os.environ.get("PROJECT_NAME", "hr-attrition-project")
DATA_DIR   = os.environ.get("DATA_DIR", "data")
RAW_PREFIX = f"{PROJECT}/data/raw"


# ── S3 bucket / folder setup ──────────────────────────────────────────────────

def create_s3_structure(s3, bucket, region):
    """Create bucket and folder structure if they don't exist."""
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"  Bucket exists: {bucket}")
    except ClientError:
        print(f"  Creating bucket: {bucket}")
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
    for folder in ["data/raw", "data/processed", "models",
                   "evaluation", "monitoring", "deployment", "code"]:
        s3.put_object(Bucket=bucket, Key=f"{PROJECT}/{folder}/")
    print(f"  S3 structure ready at s3://{bucket}/{PROJECT}/")


# ── Local CSV finder ──────────────────────────────────────────────────────────

def find_local_csv() -> str:
    """Find HR Analytics CSV in data/ folder or repo root."""
    candidates = [
        os.path.join(DATA_DIR, "HR Analytics Data.csv"),
        "HR Analytics Data.csv",
        os.path.join(DATA_DIR, "HR_Analytics_Data.csv"),
        "HR_Analytics_Data.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    found = glob.glob(os.path.join(DATA_DIR, "*.csv")) + glob.glob("*.csv")
    if found:
        print(f"  Note: using fallback CSV: {found[0]}")
        return found[0]
    print("  ERROR: No CSV file found in data/ or repo root.")
    sys.exit(1)


# ── Row count helpers ─────────────────────────────────────────────────────────

def count_local_rows(path: str) -> int:
    """Count data rows in local CSV (excludes header)."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1          # subtract header


def count_s3_rows(s3, bucket: str, key: str) -> int:
    """Count data rows in S3 CSV (excludes header)."""
    obj  = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return sum(1 for _ in io.BytesIO(data)) - 1


def get_s3_csv_key(s3, bucket: str, prefix: str):
    """Return the S3 key of the first CSV under prefix, or None."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [o["Key"] for o in resp.get("Contents", [])
            if o["Key"].endswith(".csv")]
    return keys[0] if keys else None


# ── Upload helper ─────────────────────────────────────────────────────────────

def upload_csv(s3, local_path: str, bucket: str, prefix: str):
    filename = os.path.basename(local_path)
    s3_key   = f"{prefix}/{filename}"
    size_mb  = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  Uploading : {local_path}  ({size_mb:.2f} MB)")
    print(f"  Target    : s3://{bucket}/{s3_key}")
    s3.upload_file(local_path, bucket, s3_key)
    print(f"  Upload complete.")


def verify_upload(s3, bucket: str, prefix: str):
    resp  = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    items = resp.get("Contents", [])
    print(f"\n  Verified {len(items)} file(s) in s3://{bucket}/{prefix}/")
    for obj in items:
        size_kb = obj["Size"] / 1024
        print(f"    {obj['Key']}  ({size_kb:.1f} KB)")
    if not items:
        print("  ERROR: Upload verification failed — no files found.")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    s3     = boto3.client("s3", region_name=region)

    print("\n=== HR Analytics — Data Upload to S3 ===")
    print(f"  Bucket  : {S3_BUCKET}")
    print(f"  Project : {PROJECT}")
    print(f"  Region  : {region}")

    create_s3_structure(s3, S3_BUCKET, region)

    # ── Find local CSV and count its rows ────────────────────────────────────
    local_path  = find_local_csv()
    local_rows  = count_local_rows(local_path)
    print(f"\n  Local CSV  : {local_path}  ({local_rows:,} data rows)")

    # ── Check what is already in S3 ──────────────────────────────────────────
    s3_key = get_s3_csv_key(s3, S3_BUCKET, RAW_PREFIX)

    if s3_key:
        s3_rows = count_s3_rows(s3, S3_BUCKET, s3_key)
        print(f"  S3 CSV     : s3://{S3_BUCKET}/{s3_key}  ({s3_rows:,} data rows)")

        if s3_rows >= local_rows:
            # ── S3 has MORE (or equal) data — keep it, do NOT overwrite ──────
            print(f"\n  DECISION   : S3 has {s3_rows:,} rows vs local {local_rows:,} rows.")
            print(f"  ACTION     : Keeping S3 file — skipping upload.")
            print(f"               Pipeline will train on the larger S3 dataset.")
            verify_upload(s3, S3_BUCKET, RAW_PREFIX)
            print("=== Upload skipped — existing S3 data preserved ===\n")
            return
        else:
            # ── Local is larger — upload it ───────────────────────────────────
            print(f"\n  DECISION   : Local has {local_rows:,} rows vs S3 {s3_rows:,} rows.")
            print(f"  ACTION     : Uploading larger local file.")
    else:
        # ── Nothing in S3 yet — upload local file ────────────────────────────
        print(f"\n  DECISION   : No CSV found in S3.")
        print(f"  ACTION     : Uploading local file ({local_rows:,} rows).")

    upload_csv(s3, local_path, S3_BUCKET, RAW_PREFIX)
    verify_upload(s3, S3_BUCKET, RAW_PREFIX)
    print("=== Upload complete ===\n")


if __name__ == "__main__":
    main()








































































"""
data_download.py
Upload HR Analytics CSV to S3 raw/ directory.

FIXED BEHAVIOUR:
- If S3 already has a CSV with MORE rows than the local repo file → keep S3 file
- If S3 is empty or local file is larger → upload local file
- This prevents the pipeline from overwriting a manually updated S3 dataset
  with the older smaller file stored in the repo.

Usage (local):
    pip install boto3
    export S3_BUCKET=term3-mousumi
    export PROJECT_NAME=mousumi-hr-attrition
    python src/data/data_download.py
"""

import io
import os
import sys
import glob
import boto3
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET  = os.environ["S3_BUCKET"]
PROJECT    = os.environ.get("PROJECT_NAME", "hr-attrition-project")
DATA_DIR   = os.environ.get("DATA_DIR", "data")
RAW_PREFIX = f"{PROJECT}/data/raw"


# ── S3 bucket / folder setup ──────────────────────────────────────────────────

def create_s3_structure(s3, bucket, region):
    """Create bucket and folder structure if they don't exist."""
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"  Bucket exists: {bucket}")
    except ClientError:
        print(f"  Creating bucket: {bucket}")
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
    for folder in ["data/raw", "data/processed", "models",
                   "evaluation", "monitoring", "deployment", "code"]:
        s3.put_object(Bucket=bucket, Key=f"{PROJECT}/{folder}/")
    print(f"  S3 structure ready at s3://{bucket}/{PROJECT}/")


# ── Local CSV finder ──────────────────────────────────────────────────────────

def find_local_csv() -> str:
    """Find HR Analytics CSV in data/ folder or repo root."""
    candidates = [
        os.path.join(DATA_DIR, "HR Analytics Data.csv"),
        "HR Analytics Data.csv",
        os.path.join(DATA_DIR, "HR_Analytics_Data.csv"),
        "HR_Analytics_Data.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    found = glob.glob(os.path.join(DATA_DIR, "*.csv")) + glob.glob("*.csv")
    if found:
        print(f"  Note: using fallback CSV: {found[0]}")
        return found[0]
    print("  ERROR: No CSV file found in data/ or repo root.")
    sys.exit(1)


# ── Row count helpers ─────────────────────────────────────────────────────────

def count_local_rows(path: str) -> int:
    """Count data rows in local CSV (excludes header)."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1          # subtract header


def count_s3_rows(s3, bucket: str, key: str) -> int:
    """Count data rows in S3 CSV (excludes header)."""
    obj  = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return sum(1 for _ in io.BytesIO(data)) - 1


def get_s3_csv_key(s3, bucket: str, prefix: str):
    """Return the S3 key of the first CSV under prefix, or None."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [o["Key"] for o in resp.get("Contents", [])
            if o["Key"].endswith(".csv")]
    return keys[0] if keys else None


# ── Upload helper ─────────────────────────────────────────────────────────────

def upload_csv(s3, local_path: str, bucket: str, prefix: str):
    filename = os.path.basename(local_path)
    s3_key   = f"{prefix}/{filename}"
    size_mb  = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  Uploading : {local_path}  ({size_mb:.2f} MB)")
    print(f"  Target    : s3://{bucket}/{s3_key}")
    s3.upload_file(local_path, bucket, s3_key)
    print(f"  Upload complete.")


def verify_upload(s3, bucket: str, prefix: str):
    resp  = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    items = resp.get("Contents", [])
    print(f"\n  Verified {len(items)} file(s) in s3://{bucket}/{prefix}/")
    for obj in items:
        size_kb = obj["Size"] / 1024
        print(f"    {obj['Key']}  ({size_kb:.1f} KB)")
    if not items:
        print("  ERROR: Upload verification failed — no files found.")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    s3     = boto3.client("s3", region_name=region)

    print("\n=== HR Analytics — Data Upload to S3 ===")
    print(f"  Bucket  : {S3_BUCKET}")
    print(f"  Project : {PROJECT}")
    print(f"  Region  : {region}")

    create_s3_structure(s3, S3_BUCKET, region)

    # ── Find local CSV and count its rows ────────────────────────────────────
    local_path  = find_local_csv()
    local_rows  = count_local_rows(local_path)
    print(f"\n  Local CSV  : {local_path}  ({local_rows:,} data rows)")

    # ── Check what is already in S3 ──────────────────────────────────────────
    s3_key = get_s3_csv_key(s3, S3_BUCKET, RAW_PREFIX)

    if s3_key:
        s3_rows = count_s3_rows(s3, S3_BUCKET, s3_key)
        print(f"  S3 CSV     : s3://{S3_BUCKET}/{s3_key}  ({s3_rows:,} data rows)")

        if s3_rows >= local_rows:
            # ── S3 has MORE (or equal) data — keep it, do NOT overwrite ──────
            print(f"\n  DECISION   : S3 has {s3_rows:,} rows vs local {local_rows:,} rows.")
            print(f"  ACTION     : Keeping S3 file — skipping upload.")
            print(f"               Pipeline will train on the larger S3 dataset.")
            verify_upload(s3, S3_BUCKET, RAW_PREFIX)
            print("=== Upload skipped — existing S3 data preserved ===\n")
            return
        else:
            # ── Local is larger — upload it ───────────────────────────────────
            print(f"\n  DECISION   : Local has {local_rows:,} rows vs S3 {s3_rows:,} rows.")
            print(f"  ACTION     : Uploading larger local file.")
    else:
        # ── Nothing in S3 yet — upload local file ────────────────────────────
        print(f"\n  DECISION   : No CSV found in S3.")
        print(f"  ACTION     : Uploading local file ({local_rows:,} rows).")

    upload_csv(s3, local_path, S3_BUCKET, RAW_PREFIX)
    verify_upload(s3, S3_BUCKET, RAW_PREFIX)
    print("=== Upload complete ===\n")


if __name__ == "__main__":
    main()








































































