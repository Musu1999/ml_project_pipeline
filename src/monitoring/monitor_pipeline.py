"""
monitor_pipeline.py
Checks SageMaker endpoint health and evaluation metrics.
Uploads monitoring report to S3. Sets retraining flag if thresholds breached.
"""

import os
import json
import boto3
from datetime import datetime, timezone
from botocore.exceptions import ClientError

# ── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET     = os.environ["S3_BUCKET"]
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
PROJECT       = os.environ.get("PROJECT_NAME", "hr-attrition")
EVAL_PREFIX   = f"{PROJECT}/evaluation"
MON_PREFIX    = f"{PROJECT}/monitoring"

MIN_ACCURACY  = float(os.environ.get("MIN_ACCURACY", 0.80))
MIN_ROC_AUC   = float(os.environ.get("MIN_ROC_AUC",  0.75))
MIN_RECALL    = float(os.environ.get("MIN_RECALL",   0.50))


def main():
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    s3     = boto3.client("s3",        region_name=region)
    sm     = boto3.client("sagemaker", region_name=region)
    now    = datetime.now(timezone.utc).isoformat()

    print("\n=== 6-Monitor-Pipeline: HR Analytics ===")
    print(f"  Endpoint : {ENDPOINT_NAME}")
    print(f"  Time     : {now}")

    report = {"timestamp": now, "endpoint_name": ENDPOINT_NAME,
              "project": PROJECT, "checks": {}, "alerts": []}

    # ── Endpoint health ──
    try:
        resp   = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        report["checks"]["endpoint_status"]  = status
        report["checks"]["endpoint_healthy"] = (status == "InService")
        print(f"  Endpoint status: {status}")
        if status != "InService":
            report["alerts"].append(f"Endpoint not InService: {status}")
    except ClientError as e:
        report["checks"]["endpoint_status"]  = "NOT_FOUND"
        report["checks"]["endpoint_healthy"] = False
        report["alerts"].append(f"Endpoint not found: {e}")
        print(f"  WARNING: Endpoint not found — {e}")

    # ── Evaluation metrics ──
    try:
        obj     = s3.get_object(Bucket=S3_BUCKET,
                                Key=f"{EVAL_PREFIX}/evaluation_results.json")
        metrics = json.loads(obj["Body"].read())
        acc     = metrics.get("test_accuracy",  0.0)
        roc_auc = metrics.get("test_roc_auc",   0.0)
        recall  = metrics.get("test_recall",    0.0)
        f1      = metrics.get("test_f1",        0.0)

        report["last_evaluation"] = {
            "accuracy": acc, "roc_auc": roc_auc,
            "recall": recall, "f1": f1}
        report["checks"]["accuracy_ok"] = (acc     >= MIN_ACCURACY)
        report["checks"]["roc_auc_ok"]  = (roc_auc >= MIN_ROC_AUC)
        report["checks"]["recall_ok"]   = (recall  >= MIN_RECALL)

        print(f"  Accuracy : {acc:.4f}  (min {MIN_ACCURACY})")
        print(f"  ROC AUC  : {roc_auc:.4f}  (min {MIN_ROC_AUC})")
        print(f"  Recall   : {recall:.4f}  (min {MIN_RECALL})")
        print(f"  F1       : {f1:.4f}")

        for check, thresh, val in [
            ("Accuracy", MIN_ACCURACY, acc),
            ("ROC AUC",  MIN_ROC_AUC,  roc_auc),
            ("Recall",   MIN_RECALL,   recall),
        ]:
            if val < thresh:
                report["alerts"].append(
                    f"{check} below threshold: {val:.4f} < {thresh}")
    except ClientError:
        report["last_evaluation"] = None
        for k in ["accuracy_ok", "roc_auc_ok", "recall_ok"]:
            report["checks"][k] = False
        print("  WARNING: No evaluation results found in S3")

    # ── Retrain flag ──
    needs_retrain = (
        not report["checks"].get("endpoint_healthy", False)
        or not report["checks"].get("accuracy_ok",   True)
        or not report["checks"].get("roc_auc_ok",    True)
        or not report["checks"].get("recall_ok",     True)
    )
    report["retrain_recommended"] = needs_retrain

    if needs_retrain:
        print("\n  ALERT: Retraining recommended")
        for a in report["alerts"]:
            print(f"    - {a}")
        date_str = now[:10]
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=f"{PROJECT}/retraining/trigger-{date_str}.json",
            Body=json.dumps({"retrain": True, "reason": report["alerts"]}).encode(),
        )
    else:
        print("\n  All checks passed.")

    # ── Upload report ──
    report_key = f"{MON_PREFIX}/report-{now[:10]}.json"
    s3.put_object(Bucket=S3_BUCKET, Key=report_key,
                  Body=json.dumps(report, indent=2).encode(),
                  ContentType="application/json")
    print(f"\n  Report: s3://{S3_BUCKET}/{report_key}")
    print("=== Monitoring complete ===\n")


if __name__ == "__main__":
    main()
