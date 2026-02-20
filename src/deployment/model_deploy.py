"""model_deploy.py - Deploy as Lambda (no SageMaker ECR permissions needed)."""
import os, sys, io, json, zipfile, time, boto3
from datetime import datetime
from botocore.exceptions import ClientError

S3_BUCKET       = os.environ["S3_BUCKET"]
PROJECT         = os.environ["PROJECT_NAME"]
AWS_REGION      = os.environ["AWS_REGION"]
LAMBDA_ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
ENDPOINT_NAME   = os.environ["ENDPOINT_NAME"]
DEPLOY_PREFIX   = f"{PROJECT}/deployment"
LAMBDA_NAME     = ENDPOINT_NAME.replace("-endpoint", "-lambda")

HANDLER = '''
import json, os, io, boto3, joblib, numpy as np
FEATURES = ["HighTenure_Satisfied","OverTime_Yes","Age","MonthlyIncome",
            "MaritalStatus_Single","Gender_Male","YearsAtCompany","JobSatisfaction"]
CLASSES = ["No Attrition","Attrition"]
_model = None
def load_model():
    global _model
    if _model is None:
        s3  = boto3.client("s3")
        obj = s3.get_object(Bucket=os.environ["S3_BUCKET"], Key=os.environ["MODEL_KEY"])
        _model = joblib.load(io.BytesIO(obj["Body"].read()))
    return _model
def handler(event, context):
    try:
        model = load_model()
        body = json.loads(event["body"]) if isinstance(event.get("body"), str) else event.get("body", event)
        features = body.get("features", [float(body.get(f,0)) for f in FEATURES])
        X = np.array(features, dtype=np.float32).reshape(1,-1)
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()
        return {"statusCode":200,"headers":{"Content-Type":"application/json","Access-Control-Allow-Origin":"*"},
                "body":json.dumps({"prediction":pred,"predicted_class":CLASSES[pred],
                    "attrition_probability":round(float(proba[1]),5),
                    "probabilities":{CLASSES[0]:round(float(proba[0]),5),CLASSES[1]:round(float(proba[1]),5)}})}
    except Exception as e:
        return {"statusCode":500,"body":json.dumps({"error":str(e)})}
'''

def find_key(s3, prefix, filename):
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents",[]): keys.append(obj["Key"])
    matches = [k for k in keys if k.endswith(filename)]
    if not matches: print(f"ERROR: {filename} not found"); sys.exit(1)
    print(f"  Found: s3://{S3_BUCKET}/{matches[0]}"); return matches[0]

def upload_bytes(s3, key, data):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data)
    print(f"  Uploaded: s3://{S3_BUCKET}/{key}")

def deploy(lam, model_key):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf,"w",zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("handler.py", HANDLER)
    zip_buf.seek(0); zb = zip_buf.read()
    env = {"Variables":{"S3_BUCKET":S3_BUCKET,"MODEL_KEY":model_key,"AWS_REGION":AWS_REGION}}
    try:
        lam.get_function(FunctionName=LAMBDA_NAME)
        print(f"  Updating: {LAMBDA_NAME}")
        lam.update_function_code(FunctionName=LAMBDA_NAME, ZipFile=zb)
        time.sleep(8)
        lam.update_function_configuration(FunctionName=LAMBDA_NAME,Environment=env,Timeout=60,MemorySize=512)
    except ClientError:
        print(f"  Creating: {LAMBDA_NAME}")
        lam.create_function(FunctionName=LAMBDA_NAME,Runtime="python3.11",Role=LAMBDA_ROLE_ARN,
            Handler="handler.handler",Code={"ZipFile":zb},Environment=env,Timeout=60,MemorySize=512)
    for _ in range(20):
        r = lam.get_function(FunctionName=LAMBDA_NAME)
        if r["Configuration"]["State"] == "Active": break
        time.sleep(5)
    try:
        return lam.get_function_url_config(FunctionName=LAMBDA_NAME)["FunctionUrl"]
    except ClientError:
        url = lam.create_function_url_config(FunctionName=LAMBDA_NAME,AuthType="NONE")["FunctionUrl"]
        try: lam.add_permission(FunctionName=LAMBDA_NAME,StatementId="PublicAccess",
            Action="lambda:InvokeFunctionUrl",Principal="*",FunctionUrlAuthType="NONE")
        except: pass
        return url

def main():
    ts  = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3  = boto3.client("s3",     region_name=AWS_REGION)
    lam = boto3.client("lambda", region_name=AWS_REGION)
    print(f"\n=== 5-Model-Deploy: Lambda ===")
    print(f"  Lambda : {LAMBDA_NAME} | Region: {AWS_REGION}")
    model_key = find_key(s3, PROJECT, "xgboost_model.pkl")
    fn_url    = deploy(lam, model_key)
    info = {"deployment_type":"lambda","lambda_name":LAMBDA_NAME,"function_url":fn_url,
            "model_key":model_key,"timestamp":ts}
    upload_bytes(s3, f"{DEPLOY_PREFIX}/deployment_info.json", json.dumps(info,indent=2).encode())
    print(f"\n  SUCCESS! Endpoint URL:")
    print(f"  {fn_url}")
    print(f"\n  Test:")
    print(f'  curl -X POST {fn_url} -H "Content-Type: application/json" -d \'{{"features":[0,1,28,3200,1,1,2,2]}}\'')
    print("=== Done ===\n")

if __name__ == "__main__": main()
