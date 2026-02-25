"""
api.py — HR Attrition REST API
All SageMaker inference endpoints live here.
Registered as a Flask Blueprint on app.py via /api prefix.

Endpoints:
  POST /api/predict        — Run attrition prediction
  POST /api/predict/batch  — Run prediction on multiple employees
  GET  /api/health         — Service + endpoint health check
  GET  /api/model/info     — Model metadata
  GET  /api/features       — Feature names and descriptions
"""

import os
import logging
from flask import Blueprint, request, jsonify
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__, url_prefix="/api")

# ── Config ────────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID",     "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION            = os.environ.get("AWS_REGION",            "ap-south-1")
ENDPOINT_NAME         = os.environ.get("ENDPOINT_NAME",         "")

FEATURE_NAMES = [
    "HighTenure_Satisfied",
    "OverTime_Yes",
    "Age",
    "MonthlyIncome",
    "MaritalStatus_Single",
    "Gender_Male",
    "YearsAtCompany",
    "JobSatisfaction",
]

FEATURE_META = {
    "HighTenure_Satisfied": {"type": "binary",  "values": "0=No, 1=Yes (≥5yr & Sat≥3)",  "importance": 0.031},
    "OverTime_Yes":         {"type": "binary",  "values": "0=No, 1=Yes",                   "importance": 0.312},
    "Age":                  {"type": "numeric", "values": "18–65",                          "importance": 0.142},
    "MonthlyIncome":        {"type": "numeric", "values": "1000–100000 (₹)",               "importance": 0.218},
    "MaritalStatus_Single": {"type": "binary",  "values": "0=Married/Divorced, 1=Single",  "importance": 0.062},
    "Gender_Male":          {"type": "binary",  "values": "0=Female, 1=Male",               "importance": 0.019},
    "YearsAtCompany":       {"type": "numeric", "values": "0–40",                           "importance": 0.118},
    "JobSatisfaction":      {"type": "ordinal", "values": "1=Very Low, 2=Low, 3=Med, 4=High","importance": 0.098},
}

CLASS_NAMES = ["No Attrition", "Attrition"]


# ── Helper ────────────────────────────────────────────────────────────────────
def get_sagemaker_client():
    """Build SageMaker runtime client using env credentials."""
    kwargs = dict(region_name=AWS_REGION)
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"]     = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    return boto3.client("sagemaker-runtime", **kwargs)


def invoke_endpoint(features: list) -> dict:
    """
    Call SageMaker XGBoost endpoint with CSV payload.
    Returns parsed prediction dict.
    """
    runtime  = get_sagemaker_client()
    payload  = ",".join(str(f) for f in features)
    response = runtime.invoke_endpoint(
        EndpointName = ENDPOINT_NAME,
        ContentType  = "text/csv",
        Body         = payload.encode(),
    )
    raw            = response["Body"].read().decode().strip()
    prob_attrition = round(float(raw), 5)
    prob_retention = round(1.0 - prob_attrition, 5)
    prediction     = 1 if prob_attrition >= 0.5 else 0
    risk           = "HIGH" if prob_attrition >= 0.7 else "MEDIUM" if prob_attrition >= 0.4 else "LOW"

    return {
        "prediction":            prediction,
        "predicted_class":       CLASS_NAMES[prediction],
        "attrition_probability": prob_attrition,
        "retention_probability": prob_retention,
        "risk_level":            risk,
        "probabilities": {
            "No Attrition": prob_retention,
            "Attrition":    prob_attrition,
        },
        "mode": "sagemaker_live"
    }


def validate_features(features) -> tuple[list, str | None]:
    """Validate and coerce feature list. Returns (features, error_msg)."""
    try:
        features = [float(f) for f in features]
    except (TypeError, ValueError):
        return [], "All feature values must be numeric."
    if len(features) != 8:
        return [], f"Expected 8 features, got {len(features)}."
    return features, None


# ── Routes ────────────────────────────────────────────────────────────────────

@api.route("/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body: { "features": [HighTenure_Satisfied, OverTime_Yes, Age, MonthlyIncome,
                          MaritalStatus_Single, Gender_Male, YearsAtCompany, JobSatisfaction] }
    Returns: prediction result with probability and risk level
    """
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "Request body must be JSON with a 'features' array."}), 400

    features, err = validate_features(data["features"])
    if err:
        return jsonify({"error": err}), 400

    if not ENDPOINT_NAME:
        return jsonify({"error": "ENDPOINT_NAME not configured. Check .env file."}), 500

    try:
        result = invoke_endpoint(features)
        logger.info(f"[predict] {result['predicted_class']} ({result['attrition_probability']:.1%})")
        return jsonify(result)

    except NoCredentialsError:
        return jsonify({"error": "AWS credentials missing. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env."}), 500
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg  = e.response["Error"]["Message"]
        logger.error(f"[predict] ClientError {code}: {msg}")
        return jsonify({"error": f"AWS error ({code}): {msg}"}), 500
    except Exception as e:
        logger.error(f"[predict] Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500


@api.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    POST /api/predict/batch
    Body: { "employees": [ {"name": "...", "features": [...]}, ... ] }
    Returns: list of predictions with summary stats
    """
    data = request.get_json(silent=True)
    if not data or "employees" not in data:
        return jsonify({"error": "Body must have 'employees' list."}), 400

    employees = data["employees"]
    if not isinstance(employees, list) or len(employees) == 0:
        return jsonify({"error": "employees must be a non-empty list."}), 400
    if len(employees) > 50:
        return jsonify({"error": "Maximum 50 employees per batch."}), 400

    if not ENDPOINT_NAME:
        return jsonify({"error": "ENDPOINT_NAME not configured."}), 500

    results  = []
    errors   = []
    high_risk = 0
    med_risk  = 0
    low_risk  = 0

    for i, emp in enumerate(employees):
        name     = emp.get("name", f"Employee_{i+1}")
        features = emp.get("features", [])

        feat_list, err = validate_features(features)
        if err:
            errors.append({"name": name, "error": err})
            continue

        try:
            res = invoke_endpoint(feat_list)
            results.append({
                "name":                  name,
                "prediction":            res["prediction"],
                "predicted_class":       res["predicted_class"],
                "attrition_probability": res["attrition_probability"],
                "risk_level":            res["risk_level"],
            })
            if res["risk_level"] == "HIGH":   high_risk += 1
            elif res["risk_level"] == "MEDIUM": med_risk += 1
            else:                              low_risk += 1

        except Exception as e:
            errors.append({"name": name, "error": str(e)})

    return jsonify({
        "total":    len(employees),
        "success":  len(results),
        "failed":   len(errors),
        "summary": {
            "high_risk":   high_risk,
            "medium_risk": med_risk,
            "low_risk":    low_risk,
            "attrition_count": sum(1 for r in results if r["prediction"] == 1),
        },
        "predictions": results,
        "errors":      errors,
        "mode": "sagemaker_live"
    })


@api.route("/health", methods=["GET"])
def health():
    """
    GET /api/health
    Returns service status, endpoint status, credential status.
    """
    creds_ok    = bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)
    endpoint_ok = bool(ENDPOINT_NAME)

    # Try a quick boto3 connection check (no actual inference)
    endpoint_status = "unknown"
    if creds_ok and endpoint_ok:
        try:
            sm = boto3.client(
                "sagemaker",
                region_name            = AWS_REGION,
                aws_access_key_id      = AWS_ACCESS_KEY_ID,
                aws_secret_access_key  = AWS_SECRET_ACCESS_KEY,
            )
            resp           = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
            endpoint_status = resp["EndpointStatus"]   # InService / Creating / Failed
        except ClientError as e:
            endpoint_status = f"error: {e.response['Error']['Code']}"
        except Exception as e:
            endpoint_status = f"error: {str(e)}"

    return jsonify({
        "status":          "ok",
        "service":         "AttritionIQ API",
        "version":         "2.0.0",
        "endpoint_name":   ENDPOINT_NAME  or "NOT SET",
        "endpoint_status": endpoint_status,
        "region":          AWS_REGION,
        "credentials":     "set"     if creds_ok    else "missing",
        "ready":           creds_ok and endpoint_ok and endpoint_status == "InService"
    })


@api.route("/model/info", methods=["GET"])
def model_info():
    """
    GET /api/model/info
    Returns model metadata.
    """
    return jsonify({
        "model_name":      "TunedXGBoostClassifier",
        "model_type":      "XGBoost Binary Classifier",
        "framework":       "XGBoost 1.7",
        "container":       "sagemaker-xgboost:1.7-1",
        "hyperparameters": {
            "learning_rate": 0.1,
            "max_depth":     3,
            "n_estimators":  50,
            "subsample":     0.8,
        },
        "training": {
            "dataset":       "HR Analytics Data",
            "total_samples": 1470,
            "train_samples": 1176,
            "test_samples":  294,
            "split":         "80/20 stratified",
            "attrition_rate": "16.1%",
        },
        "metrics": {
            "test_accuracy":  0.871,
            "test_precision": 0.762,
            "test_recall":    0.584,
            "test_f1":        0.661,
            "test_roc_auc":   0.823,
        },
        "features":        FEATURE_NAMES,
        "target":          "Attrition (0=No, 1=Yes)",
        "endpoint_name":   ENDPOINT_NAME or "NOT SET",
        "region":          AWS_REGION,
    })


@api.route("/features", methods=["GET"])
def features():
    """
    GET /api/features
    Returns feature names, types, valid values and importance scores.
    """
    feat_list = []
    for i, name in enumerate(FEATURE_NAMES):
        meta = FEATURE_META.get(name, {})
        feat_list.append({
            "index":       i,
            "name":        name,
            "type":        meta.get("type", "unknown"),
            "values":      meta.get("values", ""),
            "importance":  meta.get("importance", 0),
        })

    feat_list_sorted = sorted(feat_list, key=lambda x: x["importance"], reverse=True)

    return jsonify({
        "count":    len(FEATURE_NAMES),
        "features": feat_list_sorted,
        "order":    FEATURE_NAMES,
        "note":     "Send features array in 'order' sequence to /api/predict"
    })
