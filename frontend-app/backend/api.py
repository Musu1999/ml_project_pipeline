"""
backend/api.py — HR Attrition REST API Blueprint
All SageMaker inference logic lives here.
"""
import os
import logging
from flask import Blueprint, request, jsonify
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)
api    = Blueprint("api", __name__, url_prefix="/api")

AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID",     "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN     = os.environ.get("AWS_SESSION_TOKEN",     "")
AWS_REGION            = os.environ.get("AWS_REGION",            "ap-south-1")
ENDPOINT_NAME         = os.environ.get("ENDPOINT_NAME",         "")

FEATURE_NAMES = [
    "HighTenure_Satisfied","OverTime_Yes","Age","MonthlyIncome",
    "MaritalStatus_Single","Gender_Male","YearsAtCompany","JobSatisfaction",
]
CLASS_NAMES = ["No Attrition","Attrition"]
FEATURE_META = {
    "HighTenure_Satisfied": {"type":"binary",  "values":"0=No,1=Yes(>=5yr&Sat>=3)", "importance":0.031},
    "OverTime_Yes":         {"type":"binary",  "values":"0=No,1=Yes",               "importance":0.312},
    "Age":                  {"type":"numeric", "values":"18-65",                     "importance":0.142},
    "MonthlyIncome":        {"type":"numeric", "values":"1000-100000(INR)",          "importance":0.218},
    "MaritalStatus_Single": {"type":"binary",  "values":"0=Married/Divorced,1=Single","importance":0.062},
    "Gender_Male":          {"type":"binary",  "values":"0=Female,1=Male",           "importance":0.019},
    "YearsAtCompany":       {"type":"numeric", "values":"0-40",                      "importance":0.118},
    "JobSatisfaction":      {"type":"ordinal", "values":"1=VeryLow..4=High",         "importance":0.098},
}

def get_client():
    kw = dict(region_name=AWS_REGION)
    if AWS_ACCESS_KEY_ID:     kw["aws_access_key_id"]     = AWS_ACCESS_KEY_ID
    if AWS_SECRET_ACCESS_KEY: kw["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    if AWS_SESSION_TOKEN:     kw["aws_session_token"]     = AWS_SESSION_TOKEN
    return boto3.client("sagemaker-runtime", **kw)

def call_endpoint(features):
    runtime  = get_client()
    payload  = ",".join(str(f) for f in features)
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME, ContentType="text/csv", Body=payload.encode())
    raw  = response["Body"].read().decode().strip()
    prob = round(float(raw), 5)
    ret  = round(1.0 - prob, 5)
    pred = 1 if prob >= 0.5 else 0
    risk = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"
    return {"prediction":pred,"predicted_class":CLASS_NAMES[pred],
            "attrition_probability":prob,"retention_probability":ret,
            "risk_level":risk,"probabilities":{"No Attrition":ret,"Attrition":prob},
            "mode":"sagemaker_live"}

def validate(features):
    try:
        f = [float(x) for x in features]
    except (TypeError, ValueError):
        return [], "All feature values must be numeric."
    if len(f) != 8:
        return [], f"Expected 8 features, got {len(f)}."
    return f, None

@api.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error":"Send JSON with 'features' array of 8 values."}), 400
    feat, err = validate(data["features"])
    if err: return jsonify({"error":err}), 400
    if not ENDPOINT_NAME: return jsonify({"error":"ENDPOINT_NAME not set in .env"}), 500
    try:
        result = call_endpoint(feat)
        logger.info(f"[predict] {result['predicted_class']} {result['attrition_probability']:.1%}")
        return jsonify(result)
    except NoCredentialsError:
        return jsonify({"error":"AWS credentials missing. Check .env"}), 500
    except ClientError as e:
        return jsonify({"error":e.response["Error"]["Message"]}), 500
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@api.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not data or "employees" not in data:
        return jsonify({"error":"Send JSON with 'employees' list."}), 400
    employees = data["employees"]
    if not isinstance(employees,list) or len(employees)==0:
        return jsonify({"error":"employees must be a non-empty list."}), 400
    if len(employees)>50: return jsonify({"error":"Maximum 50 per batch."}), 400
    if not ENDPOINT_NAME: return jsonify({"error":"ENDPOINT_NAME not set in .env"}), 500
    results,errors = [],[]
    counts = {"HIGH":0,"MEDIUM":0,"LOW":0}
    for i,emp in enumerate(employees):
        name = emp.get("name",f"Employee_{i+1}")
        feat,err = validate(emp.get("features",[]))
        if err: errors.append({"name":name,"error":err}); continue
        try:
            r = call_endpoint(feat)
            results.append({"name":name,**r})
            counts[r["risk_level"]] += 1
        except Exception as e:
            errors.append({"name":name,"error":str(e)})
    return jsonify({"total":len(employees),"success":len(results),"failed":len(errors),
        "summary":{"high_risk":counts["HIGH"],"medium_risk":counts["MEDIUM"],
                   "low_risk":counts["LOW"],"attrition_count":sum(1 for r in results if r["prediction"]==1)},
        "predictions":results,"errors":errors,"mode":"sagemaker_live"})

@api.route("/health", methods=["GET"])
def health():
    creds_ok = bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)
    ep_ok    = bool(ENDPOINT_NAME)
    ep_status = "unknown"
    if creds_ok and ep_ok:
        try:
            kw = dict(region_name=AWS_REGION,aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
            if AWS_SESSION_TOKEN: kw["aws_session_token"]=AWS_SESSION_TOKEN
            sm = boto3.client("sagemaker",**kw)
            ep_status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
        except Exception as e:
            ep_status = f"error:{e}"
    return jsonify({"status":"ok","service":"AttritionIQ API v2.0",
        "endpoint_name":ENDPOINT_NAME or "NOT SET","endpoint_status":ep_status,
        "region":AWS_REGION,"credentials":"set" if creds_ok else "missing",
        "session_token":"set" if AWS_SESSION_TOKEN else "not set",
        "ready":creds_ok and ep_ok and ep_status=="InService"})

@api.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({"model_name":"TunedXGBoostClassifier","framework":"XGBoost 1.7",
        "container":"sagemaker-xgboost:1.7-1",
        "hyperparameters":{"learning_rate":0.1,"max_depth":3,"n_estimators":50,"subsample":0.8},
        "training":{"dataset":"HR Analytics Data","total_samples":1470,
                    "split":"80/20 stratified","attrition_rate":"16.1%"},
        "metrics":{"test_accuracy":0.871,"test_precision":0.762,"test_recall":0.584,
                   "test_f1":0.661,"test_roc_auc":0.823},
        "features":FEATURE_NAMES,"endpoint_name":ENDPOINT_NAME or "NOT SET","region":AWS_REGION})

@api.route("/features", methods=["GET"])
def features():
    feat_list = sorted([{"index":i,"name":n,**FEATURE_META.get(n,{})}
        for i,n in enumerate(FEATURE_NAMES)],key=lambda x:x.get("importance",0),reverse=True)
    return jsonify({"count":len(FEATURE_NAMES),"features":feat_list,
        "order":FEATURE_NAMES,"note":"Send features in 'order' sequence to /api/predict"})
