"""
model_deploy.py
Deploys HR Analytics XGBoost model to SageMaker endpoint.
Uses sagemaker SDK to get the correct image URI automatically.
"""

import os
import sys
import io
import json
import tarfile
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from datetime import datetime
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET          = os.environ["S3_BUCKET"]
PROJECT            = os.environ["PROJECT_NAME"]
ENDPOINT_NAME      = os.environ["ENDPOINT_NAME"]
SAGEMAKER_ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
AWS_REGION         = os.environ["AWS_REGION"]

DEPLOY_PREFIX = f"{PROJECT}/deployment"

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR        = os.path.dirname(SCRIPT_DIR)
INFERENCE_PATH = os.path.join(SRC_DIR, "inference", "inference.py")


# ── S3 helpers ────────────────────────────────────────────────────────────────

def list_bucket(s3, bucket, prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def find_key(s3, bucket, prefix, filename):
    print(f"\n  Searching for '{filename}' under s3://{bucket}/{prefix}/")
    keys = list_bucket(s3, bucket, prefix)
    matches = [k for k in keys if k.endswith(filename)]
    if not matches:
        print(f"  ERROR: '{filename}' not found. Re-run Workflow 3 first.")
        sys.exit(1)
    print(f"  Found: s3://{bucket}/{matches[0]}")
    return matches[0]


def load_bytes_s3(s3, bucket, key):
    print(f"  Downloading: s3://{bucket}/{key}")
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def upload_bytes(s3, bucket, key, data):
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    print(f"  Uploaded   : s3://{bucket}/{key}")


def read_inference_script():
    if not os.path.exists(INFERENCE_PATH):
        print(f"  ERROR: inference.py not found at {INFERENCE_PATH}")
        sys.exit(1)
    with open(INFERENCE_PATH, "rb") as f:
        data = f.read()
    print(f"  Read inference.py from repo: {INFERENCE_PATH}")
    return data


# ── Build model.tar.gz ────────────────────────────────────────────────────────

def build_model_tar(s3):
    print("\n  Building model.tar.gz ...")
    model_key    = find_key(s3, S3_BUCKET, PROJECT, "xgboost_model.pkl")
    metadata_key = find_key(s3, S3_BUCKET, PROJECT, "metadata.json")

    model_bytes     = load_bytes_s3(s3, S3_BUCKET, model_key)
    metadata_bytes  = load_bytes_s3(s3, S3_BUCKET, metadata_key)
    inference_bytes = read_inference_script()

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        for name, data in [
            ("xgboost_model.pkl", model_bytes),
            ("inference.py",      inference_bytes),
            ("metadata.json",     metadata_bytes),
        ]:
            info      = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    tar_buf.seek(0)
    tar_key = f"{DEPLOY_PREFIX}/model.tar.gz"
    upload_bytes(s3, S3_BUCKET, tar_key, tar_buf.read())
    print(f"  model.tar.gz ready: s3://{S3_BUCKET}/{tar_key}")
    return f"s3://{S3_BUCKET}/{tar_key}"


# ── Deploy using SageMaker SDK ────────────────────────────────────────────────

def deploy_endpoint(model_tar_uri):
    print(f"\n  Deploying with SageMaker SDK ...")
    print(f"  model_tar_uri : {model_tar_uri}")
    print(f"  role          : {SAGEMAKER_ROLE_ARN}")
    print(f"  endpoint      : {ENDPOINT_NAME}")

    sm_session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))

    model = SKLearnModel(
        model_data       = model_tar_uri,
        role             = SAGEMAKER_ROLE_ARN,
        entry_point      = "inference.py",
        framework_version= "1.2-1",
        py_version       = "py3",
        sagemaker_session= sm_session,
    )

    # Check if endpoint already exists — update or create
    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)
    try:
        sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        endpoint_exists = True
        print(f"  Endpoint exists — will update: {ENDPOINT_NAME}")
    except ClientError:
        endpoint_exists = False
        print(f"  Endpoint does not exist — will create: {ENDPOINT_NAME}")

    predictor = model.deploy(
        initial_instance_count = 1,
        instance_type          = "ml.t2.medium",
        endpoint_name          = ENDPOINT_NAME,
        wait                   = True,
    )

    print(f"  Endpoint InService: {ENDPOINT_NAME}")
    return predictor


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3 = boto3.client("s3", region_name=AWS_REGION)

    print("\n=== 5-Model-Deploy: HR Analytics ===")
    print(f"  Endpoint     : {ENDPOINT_NAME}")
    print(f"  Region       : {AWS_REGION}")
    print(f"  S3 Bucket    : {S3_BUCKET}")
    print(f"  Project      : {PROJECT}")
    print(f"  inference.py : {INFERENCE_PATH}")

    # Step 1: Build model.tar.gz and upload to S3
    model_tar_uri = build_model_tar(s3)

    # Step 2: Deploy using SageMaker SDK (handles image URI automatically)
    deploy_endpoint(model_tar_uri)

    # Step 3: Save deployment info
    info = {
        "endpoint_name": ENDPOINT_NAME,
        "model_tar":     model_tar_uri,
        "timestamp":     ts,
        "region":        AWS_REGION,
        "instance_type": "ml.t2.medium",
    }
    upload_bytes(s3, S3_BUCKET, f"{DEPLOY_PREFIX}/deployment_info.json",
                 json.dumps(info, indent=2).encode())

    print(f"\n  Deployment info saved.")
    print("=== Deployment complete ===\n")


if __name__ == "__main__":
    main()
