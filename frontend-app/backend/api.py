"""
backend/api.py — HR Attrition REST API Blueprint
"""
import os, logging
from flask import Blueprint, request, jsonify
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, TokenRetrievalError

logger = logging.getLogger(__name__)
api    = Blueprint("api", __name__, url_prefix="/api")

def get_env():
    return {
        "key":    os.environ.get("AWS_ACCESS_KEY_ID",     ""),
        "secret": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "token":  os.environ.get("AWS_SESSION_TOKEN",     ""),
        "region": os.environ.get("AWS_REGION",            "ap-south-1"),
        "ep":     os.environ.get("ENDPOINT_NAME",         ""),
    }

FEATURE_NAMES = [
    "HighTenure_Satisfied","OverTime_Yes","Age","MonthlyIncome",
    "MaritalStatus_Single","Gender_Male","YearsAtCompany","JobSatisfaction",
]
CLASS_NAMES = ["No Attrition","Attrition"]

def get_client(e):
    kw = dict(region_name=e["region"])
    if e["key"]:    kw["aws_access_key_id"]     = e["key"]
    if e["secret"]: kw["aws_secret_access_key"] = e["secret"]
    if e["token"]:  kw["aws_session_token"]     = e["token"]
    return boto3.client("sagemaker-runtime", **kw)

def call_endpoint(features, e):
    rt      = get_client(e)
    payload = ",".join(str(f) for f in features)
    resp    = rt.invoke_endpoint(EndpointName=e["ep"], ContentType="text/csv", Body=payload.encode())
    raw     = resp["Body"].read().decode().strip()
    prob    = round(float(raw), 5)
    ret     = round(1.0 - prob, 5)
    pred    = 1 if prob >= 0.5 else 0
    risk    = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"
    return {"prediction":pred,"predicted_class":CLASS_NAMES[pred],
            "attrition_probability":prob,"retention_probability":ret,
            "risk_level":risk,"probabilities":{"No Attrition":ret,"Attrition":prob},
            "mode":"sagemaker_live"}

def validate(features):
    try:    f = [float(x) for x in features]
    except: return [], "All values must be numeric."
    if len(f)!=8: return [], f"Expected 8 features, got {len(f)}."
    return f, None

@api.route("/predict", methods=["POST"])
def predict():
    e    = get_env()
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error":"Send JSON with 'features' array of 8 values."}), 400
    feat, err = validate(data["features"])
    if err: return jsonify({"error":err}), 400
    if not e["ep"]: return jsonify({"error":"ENDPOINT_NAME not set in .env"}), 500
    if not e["key"] or not e["secret"]:
        return jsonify({"error":"AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY missing in .env"}), 500
    try:
        result = call_endpoint(feat, e)
        return jsonify(result)
    except ClientError as ce:
        code = ce.response["Error"]["Code"]
        msg  = ce.response["Error"]["Message"]
        if "SignatureDoesNotMatch" in code or "InvalidSignature" in msg or "ExpiredToken" in code:
            return jsonify({"error":"EXPIRED_TOKEN",
                "message":"AWS session token has expired. Refresh credentials in .env from AWS Academy → AWS Details → Show."}), 401
        return jsonify({"error":f"AWS error ({code}): {msg}"}), 500
    except (NoCredentialsError, TokenRetrievalError):
        return jsonify({"error":"EXPIRED_TOKEN",
            "message":"AWS credentials invalid/expired. Refresh from AWS Academy → AWS Details → Show."}), 401
    except Exception as ex:
        logger.error(f"[predict] {ex}")
        return jsonify({"error":str(ex)}), 500

@api.route("/predict/batch", methods=["POST"])
def predict_batch():
    e = get_env()
    data = request.get_json(silent=True)
    if not data or "employees" not in data:
        return jsonify({"error":"Send JSON with 'employees' list."}), 400
    employees = data["employees"]
    if not isinstance(employees,list) or len(employees)==0:
        return jsonify({"error":"employees must be non-empty list."}), 400
    if len(employees)>50: return jsonify({"error":"Max 50 per batch."}), 400
    if not e["ep"]:  return jsonify({"error":"ENDPOINT_NAME not set."}), 500
    results,errors=[],[]
    counts={"HIGH":0,"MEDIUM":0,"LOW":0}
    for i,emp in enumerate(employees):
        name=emp.get("name",f"Employee_{i+1}")
        feat,err=validate(emp.get("features",[]))
        if err: errors.append({"name":name,"error":err}); continue
        try:
            r=call_endpoint(feat,e); results.append({"name":name,**r}); counts[r["risk_level"]]+=1
        except Exception as ex: errors.append({"name":name,"error":str(ex)})
    return jsonify({"total":len(employees),"success":len(results),"failed":len(errors),
        "summary":{"high_risk":counts["HIGH"],"medium_risk":counts["MEDIUM"],
                   "low_risk":counts["LOW"],"attrition_count":sum(1 for r in results if r["prediction"]==1)},
        "predictions":results,"errors":errors,"mode":"sagemaker_live"})

@api.route("/health", methods=["GET"])
def health():
    e = get_env()
    creds_ok  = bool(e["key"] and e["secret"])
    ep_ok     = bool(e["ep"])
    ep_status = "unknown"
    token_ok  = True
    if creds_ok and ep_ok:
        try:
            kw=dict(region_name=e["region"],aws_access_key_id=e["key"],aws_secret_access_key=e["secret"])
            if e["token"]: kw["aws_session_token"]=e["token"]
            sm=boto3.client("sagemaker",**kw)
            ep_status=sm.describe_endpoint(EndpointName=e["ep"])["EndpointStatus"]
        except ClientError as ce:
            code=ce.response["Error"]["Code"]
            if "ExpiredToken" in code or "InvalidSignature" in code or "SignatureDoesNotMatch" in code:
                ep_status="expired_token"; token_ok=False
            else:
                ep_status=f"error:{code}"
        except Exception as ex:
            ep_status=f"error:{ex}"
    return jsonify({"status":"ok","service":"AttritionIQ API v2.0",
        "endpoint_name":e["ep"] or "NOT SET","endpoint_status":ep_status,
        "region":e["region"],"credentials":"set" if creds_ok else "missing",
        "session_token":"set" if e["token"] else "not set",
        "token_valid":token_ok,
        "ready":creds_ok and ep_ok and ep_status=="InService" and token_ok})

@api.route("/model/info", methods=["GET"])
def model_info():
    e=get_env()
    return jsonify({"model_name":"TunedXGBoostClassifier","framework":"XGBoost 1.7",
        "container":"sagemaker-xgboost:1.7-1",
        "hyperparameters":{"learning_rate":0.1,"max_depth":3,"n_estimators":50,"subsample":0.8},
        "training":{"dataset":"HR Analytics Data","total_samples":1470,"split":"80/20 stratified","attrition_rate":"16.1%"},
        "metrics":{"accuracy":0.871,"precision":0.762,"recall":0.584,"f1":0.661,"roc_auc":0.823},
        "features":FEATURE_NAMES,"endpoint_name":e["ep"] or "NOT SET","region":e["region"]})

@api.route("/features", methods=["GET"])
def features():
    meta={
        "HighTenure_Satisfied":{"type":"binary","values":"0=No,1=Yes(>=5yr&Sat>=3)","importance":0.031},
        "OverTime_Yes":        {"type":"binary","values":"0=No,1=Yes",              "importance":0.312},
        "Age":                 {"type":"numeric","values":"18-65",                   "importance":0.142},
        "MonthlyIncome":       {"type":"numeric","values":"1000-100000(INR)",        "importance":0.218},
        "MaritalStatus_Single":{"type":"binary","values":"0=Married/Divorced,1=Single","importance":0.062},
        "Gender_Male":         {"type":"binary","values":"0=Female,1=Male",          "importance":0.019},
        "YearsAtCompany":      {"type":"numeric","values":"0-40",                    "importance":0.118},
        "JobSatisfaction":     {"type":"ordinal","values":"1=VeryLow..4=High",       "importance":0.098},
    }
    fl=sorted([{"index":i,"name":n,**meta.get(n,{})} for i,n in enumerate(FEATURE_NAMES)],
              key=lambda x:x.get("importance",0),reverse=True)
    return jsonify({"count":len(FEATURE_NAMES),"features":fl,"order":FEATURE_NAMES,
        "note":"Send features in 'order' sequence to /api/predict"})
