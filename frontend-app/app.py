"""
app.py — HR Attrition Dashboard (Docker/Codespace)
Calls real SageMaker endpoint via AWS credentials from .env
"""
import os
import json
import logging
from flask import Flask, render_template, request, jsonify

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Config from .env ──────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID",     "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION            = os.environ.get("AWS_REGION",            "ap-south-1")
ENDPOINT_NAME         = os.environ.get("ENDPOINT_NAME",         "")

FEATURES = [
    "HighTenure_Satisfied", "OverTime_Yes", "Age", "MonthlyIncome",
    "MaritalStatus_Single", "Gender_Male", "YearsAtCompany", "JobSatisfaction"
]
CLASS_NAMES = ["No Attrition", "Attrition"]


def get_runtime_client():
    """Build SageMaker runtime client from env credentials."""
    kwargs = dict(region_name=AWS_REGION)
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"]     = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    return boto3.client("sagemaker-runtime", **kwargs)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           endpoint_name=ENDPOINT_NAME,
                           region=AWS_REGION)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing features"}), 400

    features = [float(f) for f in data["features"]]
    if len(features) != 8:
        return jsonify({"error": f"Expected 8 features, got {len(features)}"}), 400

    if not ENDPOINT_NAME:
        return jsonify({"error": "ENDPOINT_NAME not set in .env"}), 500

    try:
        runtime = get_runtime_client()
        # XGBoost built-in container accepts CSV
        payload  = ",".join(str(f) for f in features)
        response = runtime.invoke_endpoint(
            EndpointName = ENDPOINT_NAME,
            ContentType  = "text/csv",
            Body         = payload.encode(),
        )
        raw            = response["Body"].read().decode().strip()
        prob_attrition = float(raw)
        prob_retention = round(1.0 - prob_attrition, 5)
        prob_attrition = round(prob_attrition, 5)
        prediction     = 1 if prob_attrition >= 0.5 else 0

        logger.info(f"Prediction: {CLASS_NAMES[prediction]} ({prob_attrition:.1%})")

        return jsonify({
            "prediction":            prediction,
            "predicted_class":       CLASS_NAMES[prediction],
            "attrition_probability": prob_attrition,
            "probabilities": {
                "No Attrition": prob_retention,
                "Attrition":    prob_attrition,
            },
            "mode": "sagemaker_live"
        })

    except NoCredentialsError:
        return jsonify({"error": "AWS credentials not found. Check .env file."}), 500
    except ClientError as e:
        logger.error(f"SageMaker error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    creds_ok   = bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)
    endpoint_ok = bool(ENDPOINT_NAME)
    return jsonify({
        "status":        "ok",
        "endpoint_name": ENDPOINT_NAME or "NOT SET",
        "region":        AWS_REGION,
        "credentials":   "set" if creds_ok else "missing",
        "ready":         creds_ok and endpoint_ok
    })


if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  AttritionIQ — HR Attrition Dashboard")
    print("  Mousumi · SageMaker Endpoint")
    print("═"*55)
    print(f"  Endpoint : {ENDPOINT_NAME or '⚠ NOT SET'}")
    print(f"  Region   : {AWS_REGION}")
    print(f"  Creds    : {'✅ Set' if AWS_ACCESS_KEY_ID else '⚠ Missing'}")
    print(f"  URL      : http://localhost:5000")
    print("═"*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
