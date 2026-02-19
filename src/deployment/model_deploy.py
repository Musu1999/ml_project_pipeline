"""
model_deploy.py - HR Analytics XGBoost native container deployment.
"""
import os, sys, io, json, tarfile, tempfile, boto3, joblib
from datetime import datetime
from botocore.exceptions import ClientError, WaiterError

S3_BUCKET          = os.environ["S3_BUCKET"]
PROJECT            = os.environ["PROJECT_NAME"]
ENDPOINT_NAME      = os.environ["ENDPOINT_NAME"]
SAGEMAKER_ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
AWS_REGION         = os.environ["AWS_REGION"]
DEPLOY_PREFIX      = f"{PROJECT}/deployment"
XGB_IMAGE          = f"492215442770.dkr.ecr.{AWS_REGION}.amazonaws.com/xgboost:1.7-1"

def list_bucket(s3, bucket, prefix):
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys

def find_key(s3, bucket, prefix, filename):
    print(f"  Searching for '{filename}'...")
    matches = [k for k in list_bucket(s3, bucket, prefix) if k.endswith(filename)]
    if not matches:
        print(f"  ERROR: '{filename}' not found.")
        sys.exit(1)
    print(f"  Found: s3://{bucket}/{matches[0]}")
    return matches[0]

def load_bytes_s3(s3, bucket, key):
    print(f"  Downloading: s3://{bucket}/{key}")
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

def upload_bytes(s3, bucket, key, data):
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    print(f"  Uploaded: s3://{bucket}/{key}")

def build_model_tar(s3):
    print("\n  Building model.tar.gz...")
    model_bytes = load_bytes_s3(s3, S3_BUCKET, find_key(s3, S3_BUCKET, PROJECT, "xgboost_model.pkl"))
    model = joblib.load(io.BytesIO(model_bytes))
    print(f"  Model type: {type(model)}")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "xgboost-model")
        model.get_booster().save_model(model_path)
        print(f"  Saved to: {model_path} ({os.path.getsize(model_path)} bytes)")
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            tar.add(model_path, arcname="xgboost-model")
    tar_buf.seek(0)
    tar_key = f"{DEPLOY_PREFIX}/model.tar.gz"
    upload_bytes(s3, S3_BUCKET, tar_key, tar_buf.read())
    return f"s3://{S3_BUCKET}/{tar_key}"

def deploy_endpoint(sm, model_name, config_name, model_tar_uri):
    print(f"  Image: {XGB_IMAGE}")
    sm.create_model(ModelName=model_name, ExecutionRoleArn=SAGEMAKER_ROLE_ARN,
        PrimaryContainer={"Image": XGB_IMAGE, "ModelDataUrl": model_tar_uri})
    sm.create_endpoint_config(EndpointConfigName=config_name,
        ProductionVariants=[{"VariantName":"primary","ModelName":model_name,
            "InitialInstanceCount":1,"InstanceType":"ml.t2.medium","InitialVariantWeight":1.0}])
    try:
        status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
        print(f"  Existing endpoint: {status}")
        if status == "Failed":
            sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
            import time; time.sleep(15)
            sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        else:
            sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
    except ClientError:
        print(f"  Creating endpoint: {ENDPOINT_NAME}")
        sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
    print(f"  Waiting for InService (up to 40 min)...")
    try:
        sm.get_waiter("endpoint_in_service").wait(EndpointName=ENDPOINT_NAME,
            WaiterConfig={"Delay":30,"MaxAttempts":80})
        print(f"  Endpoint InService: {ENDPOINT_NAME} SUCCESS")
    except WaiterError:
        r = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"  Status: {r['EndpointStatus']}")
        print(f"  Reason: {r.get('FailureReason','none')}")
        sys.exit(1)

def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    print(f"\n=== 5-Model-Deploy ===")
    print(f"  Endpoint: {ENDPOINT_NAME} | Region: {AWS_REGION}")
    model_tar_uri = build_model_tar(s3)
    model_name  = f"{PROJECT}-model-{ts}"
    config_name = f"{PROJECT}-config-{ts}"
    deploy_endpoint(sm, model_name, config_name, model_tar_uri)
    upload_bytes(s3, S3_BUCKET, f"{DEPLOY_PREFIX}/deployment_info.json",
        json.dumps({"endpoint_name":ENDPOINT_NAME,"model_tar":model_tar_uri,
            "timestamp":ts,"instance_type":"ml.t2.medium"}, indent=2).encode())
    print("=== Deployment complete ===\n")

if __name__ == "__main__":
    main()
