"""
inference.py - SageMaker sklearn container handler for HR Attrition model.
Container calls: model_fn → input_fn → predict_fn → output_fn
"""
import os
import io
import json
import joblib
import numpy as np

FEATURE_ORDER = [
    "HighTenure_Satisfied",
    "OverTime_Yes",
    "Age",
    "MonthlyIncome",
    "MaritalStatus_Single",
    "Gender_Male",
    "YearsAtCompany",
    "JobSatisfaction",
]
CLASS_NAMES = ["No Attrition", "Attrition"]


def model_fn(model_dir):
    """Load model from model_dir. Called once at container startup."""
    print(f"[model_fn] model_dir: {model_dir}")
    print(f"[model_fn] files: {os.listdir(model_dir)}")

    model_path = os.path.join(model_dir, "xgboost_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"xgboost_model.pkl not found in {model_dir}")

    model = joblib.load(model_path)
    print(f"[model_fn] Loaded: {type(model).__name__}")
    return model


def input_fn(request_body, content_type="application/json"):
    """Parse incoming request."""
    print(f"[input_fn] content_type={content_type}")
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict) and "features" in data:
            arr = np.array(data["features"], dtype=np.float32)
        elif isinstance(data, list):
            arr = np.array(data, dtype=np.float32)
        else:
            arr = np.array([float(data.get(f, 0)) for f in FEATURE_ORDER], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        print(f"[input_fn] shape={arr.shape}")
        return arr

    raise ValueError(f"Unsupported content_type: {content_type}")


def predict_fn(input_data, model):
    """Run prediction."""
    prediction  = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0].tolist()
    result = {
        "prediction":            prediction,
        "predicted_class":       CLASS_NAMES[prediction],
        "attrition_probability": round(float(probabilities[1]), 5),
        "probabilities": {
            CLASS_NAMES[0]: round(float(probabilities[0]), 5),
            CLASS_NAMES[1]: round(float(probabilities[1]), 5),
        }
    }
    print(f"[predict_fn] result={result}")
    return result


def output_fn(prediction, accept="application/json"):
    """Serialize output."""
    return json.dumps(prediction), "application/json"
