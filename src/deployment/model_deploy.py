"""
model_deploy.py
Packages and deploys the HR Analytics XGBoost model to a SageMaker
real-time inference endpoint. All config from environment variables.
"""

import os
import sys
import io
import json
import tarfile
import boto3
from datetime import datetime
from botocore.exceptions import ClientError

# ── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET          = os.environ["S3_BUCKET"]
PROJECT            = os.environ["PROJECT_NAME"]
ENDPOINT_NAME      = os.environ["ENDPOINT_NAME"]
SAGEMAKER_ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
AWS_REGION         = os.environ["AWS_REGION"]

MODEL_PREFIX  = f"{PROJECT}/models"
CODE_PREFIX   = f"{PROJECT}/code"
DEPLOY_PREFIX = f"{PROJECT}/deployment"


# ── S3 helpers ────────────────────────────────────────────────────────────────

def load_bytes(s3, bucket: str, key: str) -> bytes:
    print(f"  Downloading: s3://{bucket}/{key}")
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def upload_bytes(s3, bucket: str, key: str, data: bytes) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    print(f"  Uploaded   : s3://{bucket}/{key}")


# ── Upload inference.py from repo ────────────────────────────────────────────

def upload_inference_script(s3) -> None:
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    repo_root   = os.path.dirname(os.path.dirname(script_dir))
    script_path = os.path.join(repo_root, "src", "inference", "inference.py")
    if os.path.exists(script_path):
        with open(script_path, "rb") as f:
            upload_bytes(s3, S3_BUCKET, f"{CODE_PREFIX}/inference.py", f.read())
    else:
        print(f"  WARNING: inference.py not found at {script_path}")


# ── Build model.tar.gz ───────────────────────────────────────────────────────

def build_model_tar(s3) -> str:
    model_bytes     = load_bytes(s3, S3_BUCKET, f"{MODEL_PREFIX}/xgboost_model.pkl")
    inference_bytes = load_bytes(s3, S3_BUCKET, f"{CODE_PREFIX}/inference.py")
    metadata_bytes  = load_bytes(s3, S3_BUCKET, f"{PROJECT}/data/processed/metadata.json")

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
    return tar_key


# ── SageMaker endpoint ────────────────────────────────────────────────────────

def sklearn_image_uri() -> str:
    return (
        f"257758044811.dkr.ecr.{AWS_REGION}.amazonaws.com/"
        f"sagemaker-scikit-learn:1.2-1-cpu-py3"
    )


def create_or_update_endpoint(sm, model_name: str, config_name: str) -> None:
    model_uri = f"s3://{S3_BUCKET}/{DEPLOY_PREFIX}/model.tar.gz"
    image_uri = sklearn_image_uri()

    print(f"\n  Creating model       : {model_name}")
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

    print(f"  Creating config      : {config_name}")
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
        print(f"  Updating endpoint    : {ENDPOINT_NAME}")
        sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
    except ClientError:
        print(f"  Creating endpoint    : {ENDPOINT_NAME}")
        sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)

    print("  Waiting for InService (5-10 min)...")
    sm.get_waiter("endpoint_in_service").wait(
        EndpointName=ENDPOINT_NAME,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},
    )
    print(f"  Endpoint InService   : {ENDPOINT_NAME}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ts  = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3  = boto3.client("s3",        region_name=AWS_REGION)
    sm  = boto3.client("sagemaker", region_name=AWS_REGION)

    print("\n=== 5-Model-Deploy: HR Analytics ===")
    print(f"  Endpoint : {ENDPOINT_NAME}")
    print(f"  Region   : {AWS_REGION}")

    upload_inference_script(s3)
    tar_key = build_model_tar(s3)

    model_name  = f"{PROJECT}-model-{ts}"
    config_name = f"{PROJECT}-config-{ts}"
    create_or_update_endpoint(sm, model_name, config_name)

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

    print(f"\n  Deployment info: s3://{S3_BUCKET}/{DEPLOY_PREFIX}/deployment_info.json")
    print("=== Deployment complete ===\n")


if __name__ == "__main__":
    main()
