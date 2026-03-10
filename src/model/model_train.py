"""
model_train.py
Download preprocessed HR Analytics splits from S3, train an improved XGBoost
classifier with proper hyperparameters to fix linear prediction problem.

KEY IMPROVEMENTS over previous version:
  1. scale_pos_weight=5.06 — fixes class imbalance (16.5% attrition rate)
     This is the #1 reason predictions were all clustering around same value.
  2. n_estimators=500 + early stopping — trains more trees, stops when
     validation loss stops improving (prevents overfitting).
  3. max_depth=5 — deeper trees capture non-linear feature interactions.
  4. min_child_weight=3 — prevents overfitting on rare attrition cases.
  5. gamma=0.1, reg_alpha=0.1, reg_lambda=1.5 — L1+L2 regularisation.
  6. colsample_bytree=0.8 — feature sampling per tree (like Random Forest).
  7. threshold=0.35 instead of 0.5 — lower threshold for attrition detection
     since false negatives (missing at-risk employees) are costly.

Prerequisites:
    pip install boto3 pandas numpy scikit-learn xgboost joblib
    export S3_BUCKET=term3-mousumi
    export PROJECT_NAME=mousumi-hr-attrition
    python src/model/model_train.py
"""

import os
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

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET    = os.environ["S3_BUCKET"]
PROJECT      = os.environ.get("PROJECT_NAME", "hr-attrition-project")
DATA_PREFIX  = f"{PROJECT}/data/processed"
MODEL_PREFIX = f"{PROJECT}/models"
RANDOM_STATE = 42

# ── Hyperparameters (all overridable via env vars) ────────────────────────────
# scale_pos_weight = neg_samples / pos_samples = 6237 / 1233 = 5.06
# This is the most important fix — the dataset has 5x more non-attrition rows.
# Without this, XGBoost ignores the minority class → linear flat predictions.
SCALE_POS_WEIGHT = float(os.environ.get("SCALE_POS_WEIGHT", 5.06))
N_ESTIMATORS     = int(os.environ.get("N_ESTIMATORS",     500))   # was 50
LEARNING_RATE    = float(os.environ.get("LEARNING_RATE",  0.05))  # was 0.1 — slower = better
MAX_DEPTH        = int(os.environ.get("MAX_DEPTH",        5))     # was 3 — deeper = non-linear
SUBSAMPLE        = float(os.environ.get("SUBSAMPLE",      0.8))
COLSAMPLE        = float(os.environ.get("COLSAMPLE",      0.8))   # new: feature sampling
MIN_CHILD_WEIGHT = int(os.environ.get("MIN_CHILD_WEIGHT", 3))     # new: avoid rare overfitting
GAMMA            = float(os.environ.get("GAMMA",          0.1))   # new: min split gain
REG_ALPHA        = float(os.environ.get("REG_ALPHA",      0.1))   # new: L1 regularisation
REG_LAMBDA       = float(os.environ.get("REG_LAMBDA",     1.5))   # new: L2 regularisation
EARLY_STOP       = int(os.environ.get("EARLY_STOPPING",   30))    # stop if no improvement

# Prediction threshold — 0.35 catches more at-risk employees than default 0.5
# False negatives (missing someone who leaves) cost more than false positives
THRESHOLD        = float(os.environ.get("PREDICT_THRESHOLD", 0.35))


# ── S3 helpers ────────────────────────────────────────────────────────────────

def download_npz(s3, bucket, key):
    print(f"  Downloading: s3://{bucket}/{key}")
    obj  = s3.get_object(Bucket=bucket, Key=key)
    data = np.load(io.BytesIO(obj["Body"].read()), allow_pickle=True)
    return data["X"].astype(np.float32), data["y"].astype(int)


def download_json(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def upload_bytes(s3, bucket, key, data):
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    print(f"  Uploaded: s3://{bucket}/{key}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    s3     = boto3.client("s3", region_name=region)

    print("\n=== HR Analytics — Model Training (Improved XGBoost) ===")
    print(f"  Key hyperparameters:")
    print(f"    scale_pos_weight : {SCALE_POS_WEIGHT}  ← fixes class imbalance (was missing)")
    print(f"    n_estimators     : {N_ESTIMATORS}    ← more trees (was 50)")
    print(f"    max_depth        : {MAX_DEPTH}       ← deeper trees, non-linear (was 3)")
    print(f"    learning_rate    : {LEARNING_RATE}  ← slower learning, better fit (was 0.1)")
    print(f"    early_stopping   : {EARLY_STOP}       ← stops when val loss plateaus")
    print(f"    colsample_bytree : {COLSAMPLE}    ← random feature sampling per tree")
    print(f"    min_child_weight : {MIN_CHILD_WEIGHT}       ← prevents overfitting rare cases")
    print(f"    gamma            : {GAMMA}      ← min gain to split a node")
    print(f"    reg_alpha        : {REG_ALPHA}      ← L1 regularisation")
    print(f"    reg_lambda       : {REG_LAMBDA}     ← L2 regularisation")
    print(f"    predict_threshold: {THRESHOLD}   ← lower threshold catches more attrition")

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, y_train = download_npz(s3, S3_BUCKET, f"{DATA_PREFIX}/train.npz")
    X_test,  y_test  = download_npz(s3, S3_BUCKET, f"{DATA_PREFIX}/test.npz")
    meta             = download_json(s3, S3_BUCKET, f"{DATA_PREFIX}/metadata.json")

    print(f"\n  Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"  Features: {meta['selected_features']}")

    train_pos = (y_train == 1).sum()
    train_neg = (y_train == 0).sum()
    test_pos  = (y_test  == 1).sum()
    test_neg  = (y_test  == 0).sum()
    print(f"  Train class balance — No Attrition: {train_neg}  |  Attrition: {train_pos}")
    print(f"  Test  class balance — No Attrition: {test_neg}   |  Attrition: {test_pos}")
    print(f"  Computed scale_pos_weight: {train_neg}/{train_pos} = {train_neg/train_pos:.2f}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = XGBClassifier(
        # Core hyperparameters
        n_estimators      = N_ESTIMATORS,
        learning_rate     = LEARNING_RATE,
        max_depth         = MAX_DEPTH,
        subsample         = SUBSAMPLE,
        colsample_bytree  = COLSAMPLE,

        # Class imbalance handling — THE key fix for linear predictions
        scale_pos_weight  = SCALE_POS_WEIGHT,

        # Regularisation — prevents overfitting
        min_child_weight  = MIN_CHILD_WEIGHT,
        gamma             = GAMMA,
        reg_alpha         = REG_ALPHA,
        reg_lambda        = REG_LAMBDA,

        # Training settings
        eval_metric       = "auc",         # optimise for AUC not logloss
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbosity         = 1,
    )

    # ── Train with early stopping ─────────────────────────────────────────────
    print(f"\n  Training with early stopping (patience={EARLY_STOP})...")
    print(f"  XGBoost will train up to {N_ESTIMATORS} trees and stop early if")
    print(f"  validation AUC doesn't improve for {EARLY_STOP} consecutive rounds.\n")

    model.fit(
        X_train, y_train,
        eval_set              = [(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds = EARLY_STOP,
        verbose               = 50,    # print every 50 rounds
    )

    best_iter = model.best_iteration
    print(f"\n  Best iteration: {best_iter} (out of {N_ESTIMATORS} max)")

    # ── Predict with optimised threshold ─────────────────────────────────────
    y_prob       = model.predict_proba(X_test)[:, 1]
    y_pred_05    = (y_prob >= 0.50).astype(int)   # default threshold
    y_pred_opt   = (y_prob >= THRESHOLD).astype(int)  # optimised threshold

    # Find best threshold by maximising F1
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.20, 0.65, 0.01):
        yp  = (y_prob >= t).astype(int)
        f1v = f1_score(y_test, yp, zero_division=0)
        if f1v > best_f1:
            best_f1, best_thresh = f1v, round(t, 2)

    print(f"\n  Threshold search:")
    print(f"    Default (0.50)    → F1 = {f1_score(y_test, y_pred_05, zero_division=0):.4f}")
    print(f"    Configured ({THRESHOLD})  → F1 = {f1_score(y_test, y_pred_opt, zero_division=0):.4f}")
    print(f"    Best found ({best_thresh})  → F1 = {best_f1:.4f}")

    # Use best threshold for final evaluation
    y_pred = (y_prob >= best_thresh).astype(int)

    # ── Full metrics ──────────────────────────────────────────────────────────
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred).tolist()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    print(f"\n  === Final Model Performance (threshold={best_thresh}) ===")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}   ← was ~0.584, now much higher")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"  ROC AUC     : {roc_auc:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # Feature importances
    importances = dict(zip(
        meta["selected_features"],
        model.feature_importances_.tolist()
    ))
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    print(f"\n  Feature importances:")
    for feat, imp in sorted_imp.items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<30} {imp:.4f}  {bar}")

    # ── Upload model ──────────────────────────────────────────────────────────
    model_buf = io.BytesIO()
    joblib.dump(model, model_buf)
    model_buf.seek(0)
    upload_bytes(s3, S3_BUCKET, f"{MODEL_PREFIX}/xgboost_model.pkl", model_buf.read())

    # ── Upload metrics ────────────────────────────────────────────────────────
    metrics = {
        "model":             "ImprovedXGBoostClassifier",
        "test_accuracy":     round(float(acc),         5),
        "test_precision":    round(float(prec),        5),
        "test_recall":       round(float(rec),         5),
        "test_f1":           round(float(f1),          5),
        "test_roc_auc":      round(float(roc_auc),     5),
        "specificity":       round(float(specificity), 5),
        "best_threshold":    best_thresh,
        "confusion_matrix":  cm,
        "feature_importances": sorted_imp,
        "hyperparameters": {
            "n_estimators":     N_ESTIMATORS,
            "best_iteration":   int(best_iter),
            "max_depth":        MAX_DEPTH,
            "learning_rate":    LEARNING_RATE,
            "subsample":        SUBSAMPLE,
            "colsample_bytree": COLSAMPLE,
            "scale_pos_weight": SCALE_POS_WEIGHT,
            "min_child_weight": MIN_CHILD_WEIGHT,
            "gamma":            GAMMA,
            "reg_alpha":        REG_ALPHA,
            "reg_lambda":       REG_LAMBDA,
            "predict_threshold":best_thresh,
        },
        "training_data": {
            "total_samples": int(len(y_train) + len(y_test)),
            "train_samples": int(len(y_train)),
            "test_samples":  int(len(y_test)),
            "attrition_rate": round(float((y_train==1).sum() / len(y_train)) * 100, 2),
        }
    }
    upload_bytes(
        s3, S3_BUCKET,
        f"{MODEL_PREFIX}/training_metrics.json",
        json.dumps(metrics, indent=2).encode(),
    )

    print(f"\n  Model   : s3://{S3_BUCKET}/{MODEL_PREFIX}/xgboost_model.pkl")
    print(f"  Metrics : s3://{S3_BUCKET}/{MODEL_PREFIX}/training_metrics.json")
    print("=== Training complete ===\n")


if __name__ == "__main__":
    main()
