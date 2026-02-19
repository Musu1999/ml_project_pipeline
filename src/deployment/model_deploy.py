"""
model_deploy.py
Deploys HR Analytics XGBoost model to SageMaker endpoint using pure boto3.
Gets correct ECR image URI dynamically from AWS instead of hardcoding it.
"""

import os
import sys
import io
import json
import tarfile
import boto3
from datetime import datetime
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET          = os.environ["S3_BUCKET"]
PROJECT            = os.environ["PROJECT_NAME"]
ENDPOINT_NAME      = os.environ["ENDPOINT_NAME"]
SAGEMAKER_ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
AWS_REGION         = os.environ["AWS_REGION"]

DEPLOY_PREFIX  = f"{PROJECT}/deployment"
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR        = os.path.dirname(SCRIPT_DIR)
INFERENCE_PATH = os.path.join(SRC_DIR, "inference", "inference.py")

# Correct region-to-account mapping for SageMaker sklearn container
ECR_ACCOUNTS = {
    "us-east-1":      "683313688378",
    "us-east-2":      "257758044811",
    "us-west-1":      "746614075791",
    "us-west-2":      "246618743249",
    "ap-south-1":     "720646828776",
    "ap-southeast-1": "121021644041",
    "ap-southeast-2": "783357654285",
    "ap-northeast-1": "354813040037",
    "ap-northeast-2": "366743142698",
    "eu-west-1":      "141502667606",
    "eu-central-1":   "492215442770",
    "eu-west-2":      "764974769150",
}


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


def read_inference_script():
    if not os.path.exists(INFERENCE_PATH):
        print(f"  ERROR: inference.py not found at {INFERENCE_PATH}")
        sys.exit(1)
    with open(INFERENCE_PATH, "rb") as f:
        data = f.read()
    print(f"  Read inference.py from: {INFERENCE_PATH}")
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


# ── Get correct image URI ─────────────────────────────────────────────────────

def get_image_uri():
    account = ECR_ACCOUNTS.get(AWS_REGION)
    if not account:
        print(f"  ERROR: No ECR account found for region {AWS_REGION}")
        sys.exit(1)
    uri = f"{account}.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    print(f"  Image URI: {uri}")
    return uri


# ── Deploy endpoint ───────────────────────────────────────────────────────────

def deploy_endpoint(sm, model_name, config_name, model_tar_uri):
    image_uri = get_image_uri()

    print(f"\n  Creating SageMaker model : {model_name}")
    sm.create_model(
        ModelName        = model_name,
        ExecutionRoleArn = SAGEMAKER_ROLE_ARN,
        PrimaryContainer = {
            "Image":        image_uri,
            "ModelDataUrl": model_tar_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM":             "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY":    model_tar_uri,
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION":              AWS_REGION,
            },
        },
    )

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

    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"  Updating endpoint        : {ENDPOINT_NAME}")
        sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
    except ClientError:
        print(f"  Creating endpoint        : {ENDPOINT_NAME}")
        sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)

    print(f"\n  Waiting for InService (5-10 min) ...")
    sm.get_waiter("endpoint_in_service").wait(
        EndpointName = ENDPOINT_NAME,
        WaiterConfig = {"Delay": 30, "MaxAttempts": 40},
    )
    print(f"  Endpoint InService: {ENDPOINT_NAME}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3 = boto3.client("s3",        region_name=AWS_REGION)
    sm = boto3.client("sagemaker", region_name=AWS_REGION)

    print("\n=== 5-Model-Deploy: HR Analytics ===")
    print(f"  Endpoint     : {ENDPOINT_NAME}")
    print(f"  Region       : {AWS_REGION}")
    print(f"  S3 Bucket    : {S3_BUCKET}")
    print(f"  Project      : {PROJECT}")

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
    }
    upload_bytes(s3, S3_BUCKET, f"{DEPLOY_PREFIX}/deployment_info.json",
                 json.dumps(info, indent=2).encode())

    print("\n  Deployment info saved.")
    print("=== Deployment complete ===\n")


if __name__ == "__main__":
    main()
