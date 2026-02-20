import os
import sys
import argparse
import json
import logging
import boto3
import tarfile
import shutil
import struct
import io
import joblib
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_to_xgboost_native(pkl_bytes):
    """Convert XGBClassifier pkl to XGBoost native binary model."""
    import xgboost as xgb
    model = joblib.load(io.BytesIO(pkl_bytes))
    logger.info(f"Loaded model: {type(model).__name__}")
    booster = model.get_booster()
    tmp_path = "/tmp/xgboost-model"
    booster.save_model(tmp_path)
    with open(tmp_path, "rb") as f:
        model_bytes = f.read()
    os.remove(tmp_path)
    logger.info(f"XGBoost native model: {len(model_bytes)} bytes")
    return model_bytes


def create_sagemaker_model(bucket, prefix, role_arn, region):
    logger.info("="*70)
    logger.info("PREPARING SAGEMAKER MODEL ARTIFACT")
    logger.info("="*70)

    sm = boto3.client('sagemaker', region_name=region)
    s3 = boto3.client('s3', region_name=region)

    # 1. Find xgboost_model.pkl in S3
    logger.info(f"Searching for xgboost_model.pkl in s3://{bucket}/{prefix}/")
    all_keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            all_keys.append(obj["Key"])

    model_keys = [k for k in all_keys if k.endswith("xgboost_model.pkl")]
    if not model_keys:
        raise FileNotFoundError(f"No xgboost_model.pkl found under s3://{bucket}/{prefix}/")

    model_key = model_keys[0]
    logger.info(f"Found: s3://{bucket}/{model_key}")

    # 2. Download pkl and convert to XGBoost native format
    logger.info("Downloading xgboost_model.pkl ...")
    pkl_bytes = s3.get_object(Bucket=bucket, Key=model_key)["Body"].read()
    logger.info(f"Downloaded: {len(pkl_bytes)} bytes")

    # 3. Convert to XGBoost native binary (file MUST be named 'xgboost-model')
    logger.info("Converting to XGBoost native format ...")
    native_bytes = convert_to_xgboost_native(pkl_bytes)

    # 4. Build model.tar.gz with single file named 'xgboost-model'
    tmp_dir = "model_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    model_file = os.path.join(tmp_dir, "xgboost-model")
    with open(model_file, "wb") as f:
        f.write(native_bytes)

    new_tar = "deploy_model.tar.gz"
    with tarfile.open(new_tar, "w:gz") as tar:
        tar.add(model_file, arcname="xgboost-model")
    logger.info(f"Created model.tar.gz: {os.path.getsize(new_tar)} bytes")

    # 5. Upload to S3
    deploy_key = f"{prefix}/models/deploy/model.tar.gz"
    s3.upload_file(new_tar, bucket, deploy_key)
    deploy_model_uri = f"s3://{bucket}/{deploy_key}"
    logger.info(f"Uploaded to: {deploy_model_uri}")

    # 6. Use SageMaker built-in XGBoost container
    # This is the AWS-managed account for built-in algorithms — no ECR perms needed
    # The built-in XGBoost image is hosted by SageMaker service account directly
    xgb_containers = {
        "ap-south-1":    "720646828776",
        "us-east-1":     "683313688378",
        "us-east-2":     "257758044811",
        "us-west-2":     "246618743249",
        "eu-west-1":     "685385470294",
        "ap-southeast-1":"475088953585",
        "ap-northeast-1":"354813040037",
    }
    account = xgb_containers.get(region, "720646828776")
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1"
    logger.info(f"Container image: {image_uri}")

    timestamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"hr-attrition-model-{timestamp}"

    sm.create_model(
        ModelName        = model_name,
        PrimaryContainer = {
            "Image":        image_uri,
            "ModelDataUrl": deploy_model_uri,
        },
        ExecutionRoleArn = role_arn
    )
    logger.info(f"SageMaker Model created: {model_name}")

    # Cleanup
    shutil.rmtree(tmp_dir)
    os.remove(new_tar)

    return model_name, deploy_model_uri


def create_endpoint_config(model_name, region):
    logger.info("Creating endpoint configuration...")
    sm          = boto3.client('sagemaker', region_name=region)
    timestamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = f"hr-attrition-config-{timestamp}"

    sm.create_endpoint_config(
        EndpointConfigName = config_name,
        ProductionVariants = [{
            "VariantName":          "AllTraffic",
            "ModelName":            model_name,
            "InstanceType":         "ml.t2.medium",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 1.0
        }]
    )
    logger.info(f"Endpoint config created: {config_name}")
    return config_name


def create_or_update_endpoint(config_name, region, endpoint_name):
    logger.info("Creating/updating endpoint...")
    sm = boto3.client('sagemaker', region_name=region)

    try:
        existing = sm.describe_endpoint(EndpointName=endpoint_name)
        status   = existing["EndpointStatus"]
        logger.info(f"Existing endpoint status: {status}")
        if status == "Failed":
            logger.info("Deleting failed endpoint ...")
            sm.delete_endpoint(EndpointName=endpoint_name)
            import time; time.sleep(20)
            sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        else:
            sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    except sm.exceptions.ClientError:
        logger.info(f"Creating endpoint: {endpoint_name}")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

    return endpoint_name


def wait_for_endpoint(endpoint_name, region):
    logger.info(f"Waiting for endpoint {endpoint_name} to be InService ...")
    sm = boto3.client('sagemaker', region_name=region)
    import time

    for attempt in range(80):
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        status   = response["EndpointStatus"]
        logger.info(f"[{attempt+1}/80] Status: {status}")

        if status == "InService":
            logger.info("Endpoint is InService!")
            return

        if status in ["Failed", "RollingBack"]:
            reason = response.get("FailureReason", "Unknown")
            logger.error(f"Endpoint failed: {reason}")
            logger.error(f"CloudWatch: /aws/sagemaker/Endpoints/{endpoint_name}")
            raise Exception(f"Endpoint failed: {reason}")

        time.sleep(30)

    raise TimeoutError("Endpoint did not reach InService in 40 minutes.")


def deploy(bucket, prefix, role_arn, region, output_file, endpoint_name):
    logger.info("="*70)
    logger.info("HR ATTRITION MODEL DEPLOYMENT")
    logger.info("="*70)
    logger.info(f"Bucket        : {bucket}")
    logger.info(f"Prefix        : {prefix}")
    logger.info(f"Region        : {region}")
    logger.info(f"Endpoint Name : {endpoint_name}")

    model_name, model_uri = create_sagemaker_model(bucket, prefix, role_arn, region)
    config_name           = create_endpoint_config(model_name, region)
    endpoint_name         = create_or_update_endpoint(config_name, region, endpoint_name)

    wait_for_endpoint(endpoint_name, region)

    deployment_info = {
        "model_name":    model_name,
        "config_name":   config_name,
        "endpoint_name": endpoint_name,
        "model_uri":     model_uri,
        "timestamp":     datetime.now().isoformat(),
        "region":        region,
        "instance_type": "ml.t2.medium",
        "container":     "sagemaker-xgboost:1.7-1"
    }

    with open(output_file, "w") as f:
        json.dump(deployment_info, f, indent=2)

    # Save to S3
    s3 = boto3.client("s3", region_name=region)
    s3.put_object(
        Bucket = bucket,
        Key    = f"{prefix}/deployment/deployment_info.json",
        Body   = json.dumps(deployment_info, indent=2).encode()
    )
    logger.info(f"Deployment info saved to s3://{bucket}/{prefix}/deployment/deployment_info.json")

    logger.info("="*70)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info(f"Endpoint: {endpoint_name}")
    logger.info("="*70)

    return deployment_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy HR Attrition model to SageMaker")
    parser.add_argument("--bucket",        required=True)
    parser.add_argument("--prefix",        required=True)
    parser.add_argument("--role-arn",      required=True)
    parser.add_argument("--region",        default="ap-south-1")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--output-file",   default="deployment_info.json")
    args = parser.parse_args()

    try:
        deploy(
            bucket        = args.bucket,
            prefix        = args.prefix,
            role_arn      = args.role_arn,
            region        = args.region,
            output_file   = args.output_file,
            endpoint_name = args.endpoint_name
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
