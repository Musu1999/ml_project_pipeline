"""
model_evaluate.py
Downloads model + test split from S3, evaluates with full metrics
(accuracy, precision, recall, F1, ROC AUC, confusion matrix, specificity),
uploads evaluation_results.json to S3.
"""

import os
import sys
import io
import json
import boto3
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

# ── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET    = os.environ["S3_BUCKET"]
PROJECT      = os.environ.get("PROJECT_NAME", "hr-attrition")
DATA_PREFIX  = f"{PROJECT}/data/processed"
MODEL_PREFIX = f"{PROJECT}/models"
EVAL_PREFIX  = f"{PROJECT}/evaluation"


# ── S3 helpers ────────────────────────────────────────────────────────────────

def load_npz(s3, bucket: str, key: str):
    print(f"  Downloading: s3://{bucket}/{key}")
    obj  = s3.get_object(Bucket=bucket, Key=key)
    data = np.load(io.BytesIO(obj["Body"].read()), allow_pickle=True)
    return data["X"].astype(np.float32), data["y"].astype(int)


def load_bytes(s3, bucket: str, key: str) -> bytes:
    print(f"  Downloading: s3://{bucket}/{key}")
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def upload_bytes(s3, bucket: str, key: str, data: bytes,
                 content_type: str = "application/json") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    print(f"  Uploaded: s3://{bucket}/{key}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n=== 4-Model-Evaluate: HR Analytics ===")
    s3 = boto3.client("s3")

    # Load test data + metadata
    X_test, y_test = load_npz(s3, S3_BUCKET, f"{DATA_PREFIX}/test.npz")
    meta           = json.loads(load_bytes(s3, S3_BUCKET, f"{DATA_PREFIX}/metadata.json"))
    class_names    = meta["class_names"]  # ["No Attrition", "Attrition"]

    # Load model
    model = joblib.load(io.BytesIO(load_bytes(s3, S3_BUCKET, f"{MODEL_PREFIX}/xgboost_model.pkl")))
    print(f"  Model   : TunedXGBoostClassifier")
    print(f"  Test set: {X_test.shape[0]} samples")

    # ── Predictions ──
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ──
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred,
                                    target_names=class_names, output_dict=True)

    # Derived metrics (from notebook train_and_evaluate function)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv         = tn / (tn + fn) if (tn + fn) else 0.0
    miss_rate   = fn / (fn + tp) if (fn + tp) else 0.0

    # ── Print ──
    print(f"\n  Results — Tuned XGBoost:")
    print(f"    Accuracy    : {acc:.4f}")
    print(f"    Precision   : {prec:.4f}")
    print(f"    Recall      : {rec:.4f}")
    print(f"    F1-Score    : {f1:.4f}")
    print(f"    ROC AUC     : {roc_auc:.4f}")
    print(f"    Specificity : {specificity:.4f}")
    print(f"    NPV         : {npv:.4f}")
    print(f"    Miss Rate   : {miss_rate:.4f}")
    print(f"    CM          : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")

    # ── Upload results ──
    results = {
        "model":               "TunedXGBoostClassifier",
        "test_accuracy":       round(float(acc),         5),
        "test_precision":      round(float(prec),        5),
        "test_recall":         round(float(rec),         5),
        "test_f1":             round(float(f1),          5),
        "test_roc_auc":        round(float(roc_auc),     5),
        "specificity":         round(float(specificity), 5),
        "npv":                 round(float(npv),         5),
        "miss_rate":           round(float(miss_rate),   5),
        "confusion_matrix":    {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": report,
        "features_used":       meta["selected_features"],
        "n_test_samples":      int(X_test.shape[0]),
    }
    upload_bytes(s3, S3_BUCKET, f"{EVAL_PREFIX}/evaluation_results.json",
                 json.dumps(results, indent=2).encode())

    print(f"  Results : s3://{S3_BUCKET}/{EVAL_PREFIX}/evaluation_results.json")
    print("=== Evaluation complete ===\n")


if __name__ == "__main__":
    main()
