"""
model_deploy.py
Runs batch predictions on test samples using the trained model.
Saves deployment info and predictions to S3.
No Lambda/SageMaker needed - runs directly on GitHub Actions runner.
Only requires S3 permissions (already granted).
"""
import os, sys, io, json, boto3, joblib, numpy as np
from datetime import datetime

S3_BUCKET     = os.environ["S3_BUCKET"]
PROJECT       = os.environ["PROJECT_NAME"]
AWS_REGION    = os.environ.get("AWS_REGION", "ap-south-1")
DEPLOY_PREFIX = f"{PROJECT}/deployment"

FEATURES = ["HighTenure_Satisfied","OverTime_Yes","Age","MonthlyIncome",
            "MaritalStatus_Single","Gender_Male","YearsAtCompany","JobSatisfaction"]
CLASSES  = ["No Attrition", "Attrition"]

# Sample employees to predict (covers different risk profiles)
TEST_SAMPLES = [
    {"name": "High Risk - Young Overtime",    "features": [0,1,28,3200,1,1,2,2]},
    {"name": "Low Risk - Senior Satisfied",   "features": [1,0,45,8500,0,1,15,4]},
    {"name": "Medium Risk - Mid Career",      "features": [0,0,35,5500,0,0,7,3]},
    {"name": "High Risk - Unhappy Overtime",  "features": [0,1,32,4200,1,0,4,1]},
    {"name": "Low Risk - Tenured Satisfied",  "features": [1,0,52,12000,0,1,20,4]},
]

def find_key(s3, prefix, filename):
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents",[]): keys.append(obj["Key"])
    matches = [k for k in keys if k.endswith(filename)]
    if not matches:
        print(f"ERROR: {filename} not found"); sys.exit(1)
    print(f"  Found: s3://{S3_BUCKET}/{matches[0]}"); return matches[0]

def upload_bytes(s3, key, data):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data)
    print(f"  Saved: s3://{S3_BUCKET}/{key}")

def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3 = boto3.client("s3", region_name=AWS_REGION)

    print("\n=== 5-Model-Deploy: Batch Predictions ===")
    print(f"  Project  : {PROJECT}")
    print(f"  S3 Bucket: {S3_BUCKET}")

    # Load model from S3
    model_key   = find_key(s3, PROJECT, "xgboost_model.pkl")
    print(f"  Loading model...")
    obj         = s3.get_object(Bucket=S3_BUCKET, Key=model_key)
    model       = joblib.load(io.BytesIO(obj["Body"].read()))
    print(f"  Model loaded: {type(model).__name__}")

    # Run predictions on test samples
    print(f"\n  Running predictions on {len(TEST_SAMPLES)} samples...")
    results = []
    for sample in TEST_SAMPLES:
        X    = np.array(sample["features"], dtype=np.float32).reshape(1,-1)
        pred = int(model.predict(X)[0])
        prob = model.predict_proba(X)[0].tolist()
        result = {
            "employee_profile":      sample["name"],
            "features":              dict(zip(FEATURES, sample["features"])),
            "prediction":            pred,
            "predicted_class":       CLASSES[pred],
            "attrition_probability": round(float(prob[1]), 4),
            "risk_level":            "HIGH" if prob[1] >= 0.5 else "LOW",
        }
        results.append(result)
        risk = "🔴 HIGH RISK" if prob[1] >= 0.5 else "🟢 LOW RISK"
        print(f"  {risk} | {sample['name'][:35]:35s} | Attrition prob: {prob[1]:.1%}")

    # Save predictions to S3
    predictions_json = {
        "timestamp":   ts,
        "model":       "TunedXGBoostClassifier",
        "predictions": results,
        "summary": {
            "total":     len(results),
            "high_risk": sum(1 for r in results if r["risk_level"]=="HIGH"),
            "low_risk":  sum(1 for r in results if r["risk_level"]=="LOW"),
        }
    }
    upload_bytes(s3, f"{DEPLOY_PREFIX}/predictions_{ts}.json",
                 json.dumps(predictions_json, indent=2).encode())

    # Save deployment info
    deploy_info = {
        "deployment_type": "github_actions_batch",
        "model_key":       model_key,
        "timestamp":       ts,
        "predictions_key": f"{DEPLOY_PREFIX}/predictions_{ts}.json",
        "status":          "deployed",
        "note":            "Model runs on GitHub Actions runner. Predictions saved to S3.",
        "how_to_predict":  f"Download s3://{S3_BUCKET}/{DEPLOY_PREFIX}/predictions_TIMESTAMP.json",
    }
    upload_bytes(s3, f"{DEPLOY_PREFIX}/deployment_info.json",
                 json.dumps(deploy_info, indent=2).encode())

    print(f"\n  ✅ Deployment complete!")
    print(f"  High risk employees: {predictions_json['summary']['high_risk']}/{len(results)}")
    print(f"  Results saved to: s3://{S3_BUCKET}/{DEPLOY_PREFIX}/")
    print("=== Done ===\n")

if __name__ == "__main__": main()
