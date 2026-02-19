"""
inference.py
SageMaker inference handler for HR Analytics Attrition XGBoost model.
Loaded by the sklearn DLC container.

Input  JSON: {"features": [HighTenure_Satisfied, OverTime_Yes, Age, MonthlyIncome,
                            MaritalStatus_Single, Gender_Male, YearsAtCompany, JobSatisfaction]}
Output JSON: {"prediction": 0|1, "predicted_class": "...",
              "attrition_probability": 0.xx, "probabilities": {...}}
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


def model_fn(model_dir: str):
    model = joblib.load(os.path.join(model_dir, "xgboost_model.pkl"))
    meta_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        model._feature_names = meta.get("selected_features", FEATURE_ORDER)
        model._class_names   = meta.get("class_names",       CLASS_NAMES)
    else:
        model._feature_names = FEATURE_ORDER
        model._class_names   = CLASS_NAMES
    print(f"[inference] Model loaded. Features: {model._feature_names}")
    return model


def input_fn(request_body, content_type: str = "application/json"):
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict) and "features" in data:
            arr = np.array(data["features"], dtype=np.float32)
        elif isinstance(data, dict):
            arr = np.array([data.get(f, 0.0) for f in FEATURE_ORDER], dtype=np.float32)
        else:
            arr = np.array(data, dtype=np.float32)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr
    if content_type == "text/csv":
        rows = [list(map(float, r.split(",")))
                for r in request_body.decode().strip().split("\n")]
        return np.array(rows, dtype=np.float32)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: np.ndarray, model):
    preds  = model.predict(input_data).tolist()
    probas = model.predict_proba(input_data).tolist()
    class_names = getattr(model, "_class_names", CLASS_NAMES)
    results = [{
        "prediction":            int(p),
        "predicted_class":       class_names[int(p)],
        "attrition_probability": round(float(pr[1]), 5),
        "probabilities": {
            class_names[0]: round(float(pr[0]), 5),
            class_names[1]: round(float(pr[1]), 5),
        },
    } for p, pr in zip(preds, probas)]
    return results[0] if len(results) == 1 else results


def output_fn(prediction, accept: str = "application/json"):
    return json.dumps(prediction), "application/json"
