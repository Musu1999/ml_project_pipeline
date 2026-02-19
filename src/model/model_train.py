"""
model_train.py
Downloads preprocessed splits from S3, trains the final Tuned XGBoost
classifier (sole model from term2 project), uploads model + metrics to S3.

Model: XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=50, subsample=0.8)
       Best parameters from notebook GridSearchCV (Cell 39/59).
"""

import os
import sys
import io
import json
import boto3
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

# ── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET    = os.environ["S3_BUCKET"]
PROJECT      = os.environ.get("PROJECT_NAME", "hr-attrition")
DATA_PREFIX  = f"{PROJECT}/data/processed"
MODEL_PREFIX = f"{PROJECT}/models"
RANDOM_STATE = 42


# ── S3 helpers ────────────────────────────────────────────────────────────────

def load_npz(s3, bucket: str, key: str):
    print(f"  Downloading: s3://{bucket}/{key}")
    obj  = s3.get_object(Bucket=bucket, Key=key)
    data = np.load(io.BytesIO(obj["Body"].read()), allow_pickle=True)
    return data["X"].astype(np.float32), data["y"].astype(int)


def load_json(s3, bucket: str, key: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def upload_bytes(s3, bucket: str, key: str, data: bytes) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    print(f"  Uploaded: s3://{bucket}/{key}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n=== 3-Model-Train: HR Analytics (Tuned XGBoost) ===")
    s3 = boto3.client("s3")

    # Load data
    X_train, y_train = load_npz(s3, S3_BUCKET, f"{DATA_PREFIX}/train.npz")
    X_test,  y_test  = load_npz(s3, S3_BUCKET, f"{DATA_PREFIX}/test.npz")
    meta             = load_json(s3, S3_BUCKET, f"{DATA_PREFIX}/metadata.json")

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Features: {meta['selected_features']}")
    print(f"  Class balance — No Attrition: {(y_train==0).sum()}  "
          f"Attrition: {(y_train==1).sum()}")

    # ── Tuned XGBoost — final model from notebook ──
    model = XGBClassifier(
        learning_rate     = 0.1,
        max_depth         = 3,
        n_estimators      = 50,
        subsample         = 0.8,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )

    print("\n  Training Tuned XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print("  Training complete.")

    # ── Evaluate on test set ──
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n  Test Performance:")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC AUC   : {roc_auc:.4f}")

    # Feature importances
    importances = dict(zip(
        meta["selected_features"],
        model.feature_importances_.tolist()
    ))
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    print(f"\n  Feature importances: {sorted_imp}")

    # ── Upload model ──
    model_buf = io.BytesIO()
    joblib.dump(model, model_buf)
    model_buf.seek(0)
    upload_bytes(s3, S3_BUCKET, f"{MODEL_PREFIX}/xgboost_model.pkl", model_buf.read())

    # ── Upload metrics ──
    metrics = {
        "model":                "TunedXGBoostClassifier",
        "test_accuracy":        round(float(acc),     5),
        "test_precision":       round(float(prec),    5),
        "test_recall":          round(float(rec),     5),
        "test_f1":              round(float(f1),      5),
        "test_roc_auc":         round(float(roc_auc), 5),
        "confusion_matrix":     cm,
        "feature_importances":  sorted_imp,
        "hyperparameters": {
            "n_estimators":  50,
            "max_depth":     3,
            "learning_rate": 0.1,
            "subsample":     0.8,
        },
    }
    upload_bytes(s3, S3_BUCKET, f"{MODEL_PREFIX}/training_metrics.json",
                 json.dumps(metrics, indent=2).encode())

    print(f"\n  Model   : s3://{S3_BUCKET}/{MODEL_PREFIX}/xgboost_model.pkl")
    print(f"  Metrics : s3://{S3_BUCKET}/{MODEL_PREFIX}/training_metrics.json")
    print("=== Training complete ===\n")


if __name__ == "__main__":
    main()
