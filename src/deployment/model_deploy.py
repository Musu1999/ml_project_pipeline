"""
model_deploy.py
Deploys HR Analytics XGBoost model to SageMaker endpoint.
"""

import os
import sys
import io
import json
import tarfile
import boto3
from datetime import datetime
from botocore.exceptions import ClientError

S3_BUCKET          = os.environ["S3_BUCKET"]
PROJECT            = os.environ["PROJECT_NAME"]
ENDPOINT_NAME      = os.environ["ENDPOINT_NAME"]
SAGEMAKER_ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
AWS_REGION         = os.environ["AWS_REGION"]

DEPLOY_PREFIX = f"{PROJECT}/deployment"

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR        = os.path.dirname(SCRIPT_DIR)
INFERENCE_PATH = os.path.join(SRC_DIR, "inference", "inference.py")


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
    print(f"  All files found:")
    for k in keys:
        print(f"    {k}")
    matches = [k for k in keys if k.endswith(filename)]
    if not matches:
        print(f"\n  ERROR: '{filename}' not found. Re-run Workflow 3 first.")
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
    return tar_key


def get_image_uri():
    return (
        f"257758044811.dkr.ecr.{AWS_REGION}.amazonaws.com/"
        f"sagemaker-scikit-learn:1.2-1-cpu-py3"
    )


def deploy_endpoint(sm, model_name, config_name):
    model_uri = f"s3://{S3_BUCKET}/{DEPLOY_PREFIX}/model.tar.gz"
    image_uri = get_image_uri()

    print(f"\n  Creating SageMaker model : {model_name}")
    sm.create_model(
        ModelName        = model_name,
        ExecutionRoleArn = SAGEMAKER_ROLE_ARN,
        PrimaryContainer = {
            "Image":        image_uri,
            "ModelDataUrl": model_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM":             "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY":    model_uri,
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
    print(f"  Endpoint is InService: {ENDPOINT_NAME}")


def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3 = boto3.client("s3",        region_name=AWS_REGION)
    sm = boto3.client("sagemaker", region_name=AWS_REGION)

    print("\n=== 5-Model-Deploy: HR Analytics ===")
    print(f"  Endpoint     : {ENDPOINT_NAME}")
    print(f"  Region       : {AWS_REGION}")
    print(f"  S3 Bucket    : {S3_BUCKET}")
    print(f"  Project      : {PROJECT}")
    print(f"  inference.py : {INFERENCE_PATH}")

    tar_key     = build_model_tar(s3)
    model_name  = f"{PROJECT}-model-{ts}"
    config_name = f"{PROJECT}-config-{ts}"
    deploy_endpoint(sm, model_name, config_name)

    info = {
        "endpoint_name": ENDPOINT_NAME,
        "model_name":    model_name,
        "config_name":   config_name,
        "model_tar":     f"s3://{S3_BUCKET}/{tar_key}",
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
