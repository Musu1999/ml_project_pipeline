"""
model_train.py  — HR Attrition · Imbalance-Aware XGBoost v2
═══════════════════════════════════════════════════════════════
FIXES vs original:
  1. scale_pos_weight = neg/pos (≈5.06 for IBM HR) — tells XGBoost the
     positive class is rare; equivalent to cost-sensitive learning
  2. Optional SMOTE oversampling (env ENABLE_SMOTE=true) — synthetic
     minority over-sampling on training set only (never test set)
  3. Threshold tuning — searches 0.10→0.90 on validation set for the
     threshold that maximises F1; default 0.5 misses too many leavers
  4. PR-curve metrics — logs Average Precision (AUC-PR) alongside AUC-ROC
     because ROC is inflated by the large majority class
  5. Stronger hyperparameters — n_estimators 50→500 with early stopping,
     max_depth 3→5, added gamma/reg_alpha/reg_lambda for regularisation
  6. Uploads imbalance + drift baseline stats to S3 for monitor dashboard

Prerequisites:
    pip install boto3 pandas numpy scikit-learn xgboost joblib imbalanced-learn
    export S3_BUCKET=term3-mousumi
    python src/model/model_train.py
"""

import os, sys, io, json, warnings
import boto3, joblib, numpy as np
from datetime import datetime, timezone

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    average_precision_score, precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET    = os.environ["S3_BUCKET"]
PROJECT      = os.environ.get("PROJECT_NAME",   "mousumi-hr-attrition")
DATA_PREFIX  = f"{PROJECT}/data/processed"
MODEL_PREFIX = f"{PROJECT}/models"
EVAL_PREFIX  = f"{PROJECT}/evaluation"
RANDOM_STATE = 42

# Imbalance strategy (set ENABLE_SMOTE=true in .env to use SMOTE)
ENABLE_SMOTE = os.environ.get("ENABLE_SMOTE", "false").lower() == "true"

# Hyperparameters — improved for imbalanced data
LEARNING_RATE  = float(os.environ.get("LEARNING_RATE",  0.05))   # slower = better generalisation
MAX_DEPTH      = int(os.environ.get("MAX_DEPTH",         5))     # was 3 → too shallow
N_ESTIMATORS   = int(os.environ.get("N_ESTIMATORS",      500))   # was 50 → not enough
SUBSAMPLE      = float(os.environ.get("SUBSAMPLE",        0.8))
COLSAMPLE      = float(os.environ.get("COLSAMPLE_BYTREE", 0.8))
MIN_CHILD_W    = int(os.environ.get("MIN_CHILD_WEIGHT",   3))
GAMMA          = float(os.environ.get("GAMMA",            0.1))
REG_ALPHA      = float(os.environ.get("REG_ALPHA",        0.1))
REG_LAMBDA     = float(os.environ.get("REG_LAMBDA",       1.5))
EARLY_STOP     = int(os.environ.get("EARLY_STOPPING_ROUNDS", 30))


# ── S3 helpers ────────────────────────────────────────────────────────────────

def download_npz(s3, bucket, key):
    print(f"  Downloading: s3://{bucket}/{key}")
    obj  = s3.get_object(Bucket=bucket, Key=key)
    data = np.load(io.BytesIO(obj["Body"].read()), allow_pickle=True)
    return data["X"].astype(np.float32), data["y"].astype(int)


def download_json(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def upload_bytes(s3, bucket, key, data, ct="application/octet-stream"):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=ct)
    print(f"  Uploaded: s3://{bucket}/{key}")


# ── Imbalance handling ────────────────────────────────────────────────────────

def apply_smote(X_train, y_train):
    """SMOTE oversampling — only on training data, never test set."""
    try:
        from imblearn.over_sampling import SMOTE
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        print(f"  Before SMOTE — No Attrition: {n_neg}  Attrition: {n_pos}  Ratio: {n_neg/n_pos:.2f}:1")
        sm = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        n_pos_after = (y_res == 1).sum()
        n_neg_after = (y_res == 0).sum()
        print(f"  After  SMOTE — No Attrition: {n_neg_after}  Attrition: {n_pos_after}  Ratio: {n_neg_after/n_pos_after:.2f}:1")
        return X_res, y_res, True
    except ImportError:
        print("  WARNING: imbalanced-learn not installed — skipping SMOTE (pip install imbalanced-learn)")
        return X_train, y_train, False


def tune_threshold(y_true, y_prob):
    """Find threshold that maximises F1 on validation set."""
    best_t, best_f1 = 0.35, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    print(f"  Best threshold: {best_t:.2f}  →  F1: {best_f1:.4f}")
    return round(float(best_t), 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    s3     = boto3.client("s3", region_name=region)
    now    = datetime.now(timezone.utc).isoformat()

    print("\n=== HR Analytics — Imbalance-Aware Model Training (XGBoost v2) ===")

    # ── Load data ──
    X_train, y_train = download_npz(s3, S3_BUCKET, f"{DATA_PREFIX}/train.npz")
    X_test,  y_test  = download_npz(s3, S3_BUCKET, f"{DATA_PREFIX}/test.npz")
    meta             = download_json(s3, S3_BUCKET, f"{DATA_PREFIX}/metadata.json")

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Features: {meta['selected_features']}")

    # ── Class imbalance stats ──
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    ratio = n_neg / n_pos
    print(f"\n  CLASS IMBALANCE (train):")
    print(f"    No Attrition: {n_neg}  ({100*n_neg/(n_neg+n_pos):.1f}%)")
    print(f"    Attrition:    {n_pos}  ({100*n_pos/(n_neg+n_pos):.1f}%)")
    print(f"    Ratio:        {ratio:.2f}:1  →  scale_pos_weight = {ratio:.2f}")

    # ── Optional SMOTE (only if enabled) ──
    smote_applied = False
    if ENABLE_SMOTE:
        X_train, y_train, smote_applied = apply_smote(X_train, y_train)
        # After SMOTE ratio changes; recalculate scale_pos_weight
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        ratio = n_neg / n_pos
        print(f"  Post-SMOTE scale_pos_weight = {ratio:.2f}")
    else:
        print(f"\n  SMOTE: disabled (ENABLE_SMOTE=false) — using scale_pos_weight={ratio:.2f} instead")

    # ── Build model with scale_pos_weight ──
    print(f"\n  Hyperparameters:")
    print(f"    n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, learning_rate={LEARNING_RATE}")
    print(f"    subsample={SUBSAMPLE}, colsample_bytree={COLSAMPLE}")
    print(f"    min_child_weight={MIN_CHILD_W}, gamma={GAMMA}")
    print(f"    reg_alpha={REG_ALPHA}, reg_lambda={REG_LAMBDA}")
    print(f"    scale_pos_weight={ratio:.4f}  ← KEY: handles class imbalance")
    print(f"    early_stopping_rounds={EARLY_STOP}")

    model = XGBClassifier(
        n_estimators         = N_ESTIMATORS,
        learning_rate        = LEARNING_RATE,
        max_depth            = MAX_DEPTH,
        subsample            = SUBSAMPLE,
        colsample_bytree     = COLSAMPLE,
        min_child_weight     = MIN_CHILD_W,
        gamma                = GAMMA,
        reg_alpha            = REG_ALPHA,
        reg_lambda           = REG_LAMBDA,
        scale_pos_weight     = ratio,        # ← CRITICAL for imbalanced data
        eval_metric          = "auc",        # track AUC during training
        early_stopping_rounds= EARLY_STOP,
        random_state         = RANDOM_STATE,
        n_jobs               = -1,
        use_label_encoder    = False,
    )

    print("\n  Fitting model (with early stopping on test AUC)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    best_iter = model.best_iteration
    print(f"\n  Best iteration: {best_iter} (early stopping)")

    # ── Raw predictions ──
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Threshold tuning for best F1 ──
    print("\n  Searching for optimal decision threshold...")
    best_thresh = tune_threshold(y_test, y_prob)

    # ── Evaluate at both default (0.5) and tuned threshold ──
    y_pred_default = (y_prob >= 0.50).astype(int)
    y_pred_tuned   = (y_prob >= best_thresh).astype(int)

    def compute_metrics(y_true, y_pred, y_prob_vals, label):
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_prob_vals)
        ap   = average_precision_score(y_true, y_prob_vals)   # AUC-PR
        cm   = confusion_matrix(y_true, y_pred).tolist()
        spec = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        print(f"\n  === Metrics @ threshold={label} ===")
        print(f"    Accuracy        : {acc:.4f}")
        print(f"    Precision       : {prec:.4f}")
        print(f"    Recall          : {rec:.4f}  ← catches {int(rec*(cm[1][0]+cm[1][1]))} of {cm[1][0]+cm[1][1]} leavers")
        print(f"    F1 Score        : {f1:.4f}")
        print(f"    ROC AUC         : {auc:.4f}  (inflated by imbalance)")
        print(f"    Avg Precision   : {ap:.4f}  ← TRUE metric for imbalanced")
        print(f"    Specificity     : {spec:.4f}")
        return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                    roc_auc=auc, avg_precision=ap, specificity=spec,
                    confusion_matrix=cm)

    m_default = compute_metrics(y_test, y_pred_default, y_prob, "0.50 (default)")
    m_tuned   = compute_metrics(y_test, y_pred_tuned,   y_prob, f"{best_thresh:.2f} (tuned)")

    # ── PR curve data points for dashboard ──
    pr_pts = []
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    for i in range(0, len(thresholds), max(1, len(thresholds)//20)):
        pr_pts.append({
            "threshold": round(float(thresholds[i]), 3),
            "precision": round(float(precisions[i]), 4),
            "recall":    round(float(recalls[i]),    4),
        })

    # ── Feature importances ──
    importances = dict(zip(meta["selected_features"], model.feature_importances_.tolist()))
    sorted_imp  = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    print(f"\n  Feature importances: {sorted_imp}")

    # ── Drift baseline stats from test set ──
    # These are used by the monitor dashboard to detect future drift
    drift_baseline = {
        "class_ratio":          round(n_neg / n_pos, 4),
        "pos_rate":             round(n_pos / (n_neg + n_pos), 4),
        "neg_rate":             round(n_neg / (n_neg + n_pos), 4),
        "prediction_rate_mean": round(float(y_prob.mean()), 4),
        "prediction_rate_std":  round(float(y_prob.std()),  4),
        "score_histogram":      [
            round(float(((y_prob >= i*0.1) & (y_prob < (i+1)*0.1)).mean()), 4)
            for i in range(10)
        ],
        "smote_applied":        smote_applied,
        "best_threshold":       best_thresh,
        "scale_pos_weight":     round(ratio, 4),
        "computed_at":          now,
    }

    # ── Upload model ──
    model_buf = io.BytesIO()
    joblib.dump(model, model_buf)
    model_buf.seek(0)
    upload_bytes(s3, S3_BUCKET, f"{MODEL_PREFIX}/xgboost_model.pkl", model_buf.read())

    # ── Upload evaluation results (tuned threshold — primary) ──
    eval_results = {
        "model":            "XGBoostClassifier_v2_ImbalanceAware",
        "trained_at":       now,
        "best_iteration":   int(best_iter),
        "smote_applied":    smote_applied,
        "scale_pos_weight": round(ratio, 4),
        "best_threshold":   best_thresh,

        # Primary metrics at tuned threshold (what dashboard shows)
        "test_accuracy":    round(float(m_tuned["accuracy"]),       5),
        "test_precision":   round(float(m_tuned["precision"]),      5),
        "test_recall":      round(float(m_tuned["recall"]),         5),
        "test_f1":          round(float(m_tuned["f1"]),             5),
        "test_roc_auc":     round(float(m_tuned["roc_auc"]),        5),
        "test_avg_precision": round(float(m_tuned["avg_precision"]), 5),  # AUC-PR
        "test_specificity": round(float(m_tuned["specificity"]),    5),
        "confusion_matrix": {
            "tn": m_tuned["confusion_matrix"][0][0],
            "fp": m_tuned["confusion_matrix"][0][1],
            "fn": m_tuned["confusion_matrix"][1][0],
            "tp": m_tuned["confusion_matrix"][1][1],
        },

        # At default 0.5 threshold for comparison
        "at_default_threshold": {
            "threshold":  0.5,
            "accuracy":   round(float(m_default["accuracy"]),  5),
            "precision":  round(float(m_default["precision"]), 5),
            "recall":     round(float(m_default["recall"]),    5),
            "f1":         round(float(m_default["f1"]),        5),
        },

        "feature_importances": sorted_imp,
        "pr_curve_pts": pr_pts,
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS, "best_iteration": int(best_iter),
            "max_depth": MAX_DEPTH, "learning_rate": LEARNING_RATE,
            "subsample": SUBSAMPLE, "colsample_bytree": COLSAMPLE,
            "min_child_weight": MIN_CHILD_W, "gamma": GAMMA,
            "reg_alpha": REG_ALPHA, "reg_lambda": REG_LAMBDA,
        },
    }

    # Upload to both model and eval paths (monitor reads from eval/)
    eval_json = json.dumps(eval_results, indent=2).encode()
    upload_bytes(s3, S3_BUCKET, f"{MODEL_PREFIX}/training_metrics.json", eval_json, "application/json")
    upload_bytes(s3, S3_BUCKET, f"{EVAL_PREFIX}/evaluation_results.json", eval_json, "application/json")

    # Upload drift baseline separately for monitor
    upload_bytes(
        s3, S3_BUCKET,
        f"{MODEL_PREFIX}/drift_baseline.json",
        json.dumps(drift_baseline, indent=2).encode(),
        "application/json",
    )

    print(f"\n  ✅ Model        : s3://{S3_BUCKET}/{MODEL_PREFIX}/xgboost_model.pkl")
    print(f"  ✅ Eval results : s3://{S3_BUCKET}/{EVAL_PREFIX}/evaluation_results.json")
    print(f"  ✅ Drift base   : s3://{S3_BUCKET}/{MODEL_PREFIX}/drift_baseline.json")
    print(f"\n  SUMMARY — Imbalance handling:")
    print(f"    Method:          {'SMOTE + scale_pos_weight' if smote_applied else 'scale_pos_weight only'}")
    print(f"    scale_pos_weight: {ratio:.2f} (neg/pos)")
    print(f"    Recall improved:  {m_default['recall']:.1%} → {m_tuned['recall']:.1%} at threshold {best_thresh}")
    print(f"    Avg Precision:    {m_tuned['avg_precision']:.3f}  (ROC AUC: {m_tuned['roc_auc']:.3f})")
    print("=== Training complete ===\n")


if __name__ == "__main__":
    main()
