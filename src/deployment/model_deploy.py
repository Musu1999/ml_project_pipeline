import os
import sys
import argparse
import json
import logging
import boto3
import tarfile
import shutil
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sagemaker_model(bucket, prefix, role_arn, region):
    logger.info("="*70)
    logger.info("PREPARING SAGEMAKER MODEL ARTIFACT")
    logger.info("="*70)

    sm = boto3.client('sagemaker', region_name=region)
    s3 = boto3.client('s3', region_name=region)

    # 1. Find xgboost_model.pkl in S3
    logger.info(f"Searching for xgboost_model.pkl in s3://{bucket}/{prefix}/")
    all_keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            all_keys.append(obj["Key"])

    model_keys = [k for k in all_keys if k.endswith("xgboost_model.pkl")]
    if not model_keys:
        raise FileNotFoundError(
            f"No xgboost_model.pkl found under s3://{bucket}/{prefix}/\n"
            f"Re-run Workflow 3 (Model Train) first."
        )

    model_key = model_keys[0]
    logger.info(f"Found model: s3://{bucket}/{model_key}")

    # 2. Download pkl
    logger.info("Downloading xgboost_model.pkl ...")
    pkl_bytes = s3.get_object(Bucket=bucket, Key=model_key)["Body"].read()
    logger.info(f"Downloaded: {len(pkl_bytes)} bytes")

    # 3. Build directory: xgboost_model.pkl + code/inference.py
    tmp_dir = "model_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    os.makedirs(os.path.join(tmp_dir, "code"))

    with open(os.path.join(tmp_dir, "xgboost_model.pkl"), "wb") as f:
        f.write(pkl_bytes)

    inference_script = "src/inference/inference.py"
    if not os.path.exists(inference_script):
        raise FileNotFoundError(f"Inference script not found: {inference_script}")
    shutil.copy(inference_script, os.path.join(tmp_dir, "code", "inference.py"))
    logger.info("Copied inference.py into model package")

    # 4. Create model.tar.gz
    new_tar = "deploy_model.tar.gz"
    with tarfile.open(new_tar, "w:gz") as tar:
        for item in os.listdir(tmp_dir):
            tar.add(os.path.join(tmp_dir, item), arcname=item)
    logger.info(f"Created model.tar.gz: {os.path.getsize(new_tar)} bytes")

    # 5. Upload to S3
    deploy_key = f"{prefix}/models/deploy/model.tar.gz"
    s3.upload_file(new_tar, bucket, deploy_key)
    deploy_model_uri = f"s3://{bucket}/{deploy_key}"
    logger.info(f"Uploaded to: {deploy_model_uri}")

    # 6. Create SageMaker Model
    # Account 763104351884 is the AWS Deep Learning Container account (works for student accounts)
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    logger.info(f"Container image: {image_uri}")

    timestamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"hr-attrition-model-{timestamp}"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image":        image_uri,
            "ModelDataUrl": deploy_model_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM":          "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": deploy_model_uri,
            }
        },
        ExecutionRoleArn=role_arn
    )
    logger.info(f"SageMaker Model created: {model_name}")

    # Cleanup
    shutil.rmtree(tmp_dir)
    os.remove(new_tar)

    return model_name, deploy_model_uri


def create_endpoint_config(model_name, region):
    logger.info("Creating endpoint configuration...")

    sm        = boto3.client('sagemaker', region_name=region)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = f"hr-attrition-config-{timestamp}"

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
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
            logger.info("Deleting failed endpoint and recreating...")
            sm.delete_endpoint(EndpointName=endpoint_name)
            import time; time.sleep(20)
            sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        else:
            logger.info(f"Updating endpoint: {endpoint_name}")
            sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

    except sm.exceptions.ClientError:
        logger.info(f"Creating new endpoint: {endpoint_name}")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

    return endpoint_name


def wait_for_endpoint(endpoint_name, region):
    logger.info(f"Waiting for endpoint {endpoint_name} to be InService...")

    sm = boto3.client('sagemaker', region_name=region)
    import time

    max_wait = 80   # 80 x 30s = 40 minutes
    for attempt in range(max_wait):
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        status   = response["EndpointStatus"]
        logger.info(f"[{attempt+1}/{max_wait}] Status: {status}")

        if status == "InService":
            logger.info("Endpoint is ready!")
            return

        if status in ["Failed", "RollingBack"]:
            reason = response.get("FailureReason", "Unknown")
            logger.error(f"Failure reason: {reason}")
            logger.error(f"CloudWatch logs: /aws/sagemaker/Endpoints/{endpoint_name}")
            raise Exception(f"Endpoint creation failed: {reason}")

        time.sleep(30)

    raise TimeoutError(
        f"Endpoint did not reach InService after {max_wait * 30 // 60} minutes."
    )


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
        "instance_type": "ml.t2.medium"
    }

    with open(output_file, "w") as f:
        json.dump(deployment_info, f, indent=2)

    # Also save to S3
    s3 = boto3.client("s3", region_name=region)
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/deployment/deployment_info.json",
        Body=json.dumps(deployment_info, indent=2).encode()
    )

    logger.info("="*70)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info(f"Endpoint: {endpoint_name}")
    logger.info("="*70)

    return deployment_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy HR Attrition model to SageMaker")
    parser.add_argument("--bucket",        required=True,          help="S3 bucket name")
    parser.add_argument("--prefix",        required=True,          help="S3 prefix / project name")
    parser.add_argument("--role-arn",      required=True,          help="SageMaker execution role ARN")
    parser.add_argument("--region",        default="ap-south-1",   help="AWS region")
    parser.add_argument("--endpoint-name", required=True,          help="SageMaker endpoint name")
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
