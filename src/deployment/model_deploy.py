"""
model_deploy.py
Deploys HR Analytics XGBoost model using XGBoost native container.
"""

import os
import sys
import io
import json
import tarfile
import tempfile
import boto3
import joblib
from datetime import datetime
from botocore.exceptions import ClientError, WaiterError

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET          = os.environ["S3_BUCKET"]
PROJECT            = os.environ["PROJECT_NAME"]
ENDPOINT_NAME      = os.environ["ENDPOINT_NAME"]
SAGEMAKER_ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
AWS_REGION         = os.environ["AWS_REGION"]

DEPLOY_PREFIX  = f"{PROJECT}/deployment"
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR        = os.path.dirname(SCRIPT_DIR)

# XGBoost built-in container for ap-south-1
XGB_IMAGE = f"492215442770.dkr.ecr.{AWS_REGION}.amazonaws.com/xgboost:1.7-1"


# ── S3 helpers ────────────────────────────────────────────────────────────────

def list_bucket(s3, bucket, prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def find_key(s3, bucket, prefix, filename):
    print(f"  Searching for '{filename}' in s3://{bucket}/{prefix}/")
    keys    = list_bucket(s3, bucket, prefix)
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


# ── Build model.tar.gz ────────────────────────────────────────────────────────

def build_model_tar(s3):
    print("\n  Building model.tar.gz (XGBoost native format)...")

    # Download pkl from S3
    model_key   = find_key(s3, S3_BUCKET, PROJECT, "xgboost_model.pkl")
    model_bytes = load_bytes_s3(s3, S3_BUCKET, model_key)

    # Load XGBClassifier from pkl
    model = joblib.load(io.BytesIO(model_bytes))
    print(f"  Model type: {type(model)}")

    # Save booster to a TEMP FILE (save_model needs a path, not BytesIO)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "xgboost-model")
        model.get_booster().save_model(model_path)
        print(f"  Saved XGBoost native model to: {model_path}")
        print(f"  Model file size: {os.path.getsize(model_path)} bytes")

        # Bundle into model.tar.gz
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            tar.add(model_path, arcname="xgboost-model")

    tar_buf.seek(0)
    tar_key = f"{DEPLOY_PREFIX}/model.tar.gz"
    upload_bytes(s3, S3_BUCKET, tar_key, tar_buf.read())
    print(f"  model.tar.gz ready: s3://{S3_BUCKET}/{tar_key}")
    return f"s3://{S3_BUCKET}/{tar_key}"


# ── Deploy endpoint ───────────────────────────────────────────────────────────

def deploy_endpoint(sm, model_name, config_name, model_tar_uri):
    print(f"\n  Image URI : {XGB_IMAGE}")
    print(f"  Model tar : {model_tar_uri}")

    # Create SageMaker Model
    print(f"  Creating SM model    : {model_name}")
    sm.create_model(
        ModelName        = model_name,
        ExecutionRoleArn = SAGEMAKER_ROLE_ARN,
        PrimaryContainer = {
            "Image":        XGB_IMAGE,
            "ModelDataUrl": model_tar_uri,
        },
    )

    # Create Endpoint Config
    print(f"  Creating endpoint config : {config_name}")
    sm.create_endpoint_config(
        EndpointConfigName = config_name,
        ProductionVariants = [{
            "VariantName":          "primary",
            "ModelName":            model_name,
            "InitialInstanceCount": 1,
            "InstanceType":         "ml.t2.medium",
            "InitialVariantWeight": 1.0,
        }],
    )

    # Create or update endpoint
    try:
        existing = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status   = existing["EndpointStatus"]
        print(f"  Existing endpoint status: {status}")
        if status == "Failed":
            print(f"  Deleting failed endpoint and recreating...")
            sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
            import time; time.sleep(15)
            sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        elif status == "InService":
            sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        else:
            print(f"  Endpoint is {status}, waiting...")
    except ClientError:
        print(f"  Creating endpoint : {ENDPOINT_NAME}")
        sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)

    # Wait up to 40 minutes
    print(f"\n  Waiting for InService (up to 40 min)...")
    try:
        sm.get_waiter("endpoint_in_service").wait(
            EndpointName = ENDPOINT_NAME,
            WaiterConfig = {"Delay": 30, "MaxAttempts": 80},
        )
        print(f"  Endpoint InService: {ENDPOINT_NAME} ✅")
    except WaiterError:
        resp   = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        reason = resp.get("FailureReason", "No reason provided")
        print(f"\n  Status  : {status}")
        print(f"  Reason  : {reason}")
        sys.exit(1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3 = boto3.client("s3",        region_name=AWS_REGION)
    sm = boto3.client("sagemaker", region_name=AWS_REGION)

    print("\n=== 5-Model-Deploy: HR Analytics ===")
    print(f"  Endpoint  : {ENDPOINT_NAME}")
    print(f"  Region    : {AWS_REGION}")
    print(f"  S3 Bucket : {S3_BUCKET}")
    print(f"  Project   : {PROJECT}")

    model_tar_uri = build_model_tar(s3)
    model_name    = f"{PROJECT}-model-{ts}"
    config_name   = f"{PROJECT}-config-{ts}"
    deploy_endpoint(sm, model_name, config_name, model_tar_uri)

    info = {
        "endpoint_name": ENDPOINT_NAME,
        "model_name":    model_name,
        "config_name":   config_name,
        "model_tar":     model_tar_uri,
        "timestamp":     ts,
        "region":        AWS_REGION,
        "instance_type": "ml.t2.medium",
        "container":     "xgboost-native",
    }
    upload_bytes(s3, S3_BUCKET, f"{DEPLOY_PREFIX}/deployment_info.json",
                 json.dumps(info, indent=2).encode())
    print("\n  Deployment info saved.")
    print("=== Deployment complete ===\n")


if __name__ == "__main__":
    main()
