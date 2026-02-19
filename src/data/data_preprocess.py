"""
data_preprocess.py
Downloads HR_Analytics_Data.csv from S3 raw/, applies feature engineering
from the notebook (Manual_Features.ipynb), builds the 8 manually selected
features, splits 80/20, uploads train.npz + test.npz + metadata.json to S3.

Final 8 features (from notebook Cell 7):
    HighTenure_Satisfied, OverTime_Yes, Age, MonthlyIncome,
    MaritalStatus_Single, Gender_Male, YearsAtCompany, JobSatisfaction
"""

import os
import sys
import io
import json
import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from botocore.exceptions import ClientError

# ── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET    = os.environ["S3_BUCKET"]
PROJECT      = os.environ.get("PROJECT_NAME", "hr-attrition")
RAW_S3_KEY   = f"{PROJECT}/data/raw/HR_Analytics_Data.csv"
OUT_PREFIX   = f"{PROJECT}/data/processed"
RANDOM_STATE = 42

# Final 8 manually selected features (notebook Cell 7 manual_selected)
SELECTED_FEATURES = [
    "HighTenure_Satisfied",
    "OverTime_Yes",
    "Age",
    "MonthlyIncome",
    "MaritalStatus_Single",
    "Gender_Male",
    "YearsAtCompany",
    "JobSatisfaction",
]


# ── Feature Engineering (mirrors notebook exactly) ───────────────────────────

def safe_cut(series, bins, labels):
    try:
        return pd.cut(series, bins=bins, labels=labels)
    except Exception:
        return pd.qcut(series.fillna(series.median()), q=len(labels),
                       labels=labels, duplicates="drop")


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Age", "YearsAtCompany", "JobSatisfaction", "YearsInCurrentRole"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Age groups
    if "Age" in df.columns:
        df["AgeGroup"] = safe_cut(
            df["Age"], bins=[18, 30, 40, 50, 60, 100],
            labels=["18-30", "31-40", "41-50", "51-60", "60+"])

    # Tenure buckets
    if "YearsAtCompany" in df.columns:
        df["_yac"] = df["YearsAtCompany"].fillna(0)
        df["TenureBucket"] = pd.cut(df["_yac"], bins=[-0.1, 2, 5, 10, 100],
                                    labels=["<2", "2-5", "6-10", "10+"])
        df.drop(columns=["_yac"], inplace=True)

    # JobSatisfaction × OverTime interaction
    if "JobSatisfaction" in df.columns and "OverTime" in df.columns:
        df["_js"] = df["JobSatisfaction"].fillna(df["JobSatisfaction"].median())
        df["SatisfactionOverTime"] = (
            df["_js"].astype(int).astype(str) + "_" + df["OverTime"].astype(str))
        df.drop(columns=["_js"], inplace=True)

    # Percent years in current role vs tenure
    if "YearsInCurrentRole" in df.columns and "YearsAtCompany" in df.columns:
        df["YearsInCurrentRole_Percent"] = (
            df["YearsInCurrentRole"].fillna(0) / (df["YearsAtCompany"].fillna(0) + 1))

    # HighTenure_Satisfied flag (key engineered feature)
    if "TenureBucket" in df.columns and "JobSatisfaction" in df.columns:
        df["HighTenure_Satisfied"] = (
            (df["TenureBucket"] == "10+") &
            (df["JobSatisfaction"].fillna(0) >= 3)
        ).astype(int)
    else:
        df["HighTenure_Satisfied"] = 0

    return df


def encode_features(df: pd.DataFrame):
    """One-hot encode categoricals, map Attrition → 0/1, return X, y."""
    df = df.copy()

    # Target
    df["AttritionFlag"] = (
        df["Attrition"].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0})
    )

    cat_cols = [c for c in [
        "BusinessTravel", "Department", "EducationField", "Gender",
        "JobRole", "MaritalStatus", "OverTime", "Over18",
        "AgeGroup", "TenureBucket", "SatisfactionOverTime",
    ] if c in df.columns]

    for col in cat_cols:
        df[col] = df[col].astype("category")

    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Convert bool columns to int
    for col in df_enc.select_dtypes(include="bool").columns:
        df_enc[col] = df_enc[col].astype(int)

    all_features = [c for c in df_enc.columns
                    if c not in ["Attrition", "AttritionFlag"]]
    X_full = df_enc[all_features]
    y      = df_enc["AttritionFlag"].fillna(0).astype(int)
    return X_full, y


# ── S3 helpers ────────────────────────────────────────────────────────────────

def download_csv(s3, bucket: str, key: str) -> pd.DataFrame:
    print(f"  Downloading: s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def upload_bytes(s3, bucket: str, key: str, data: bytes,
                 content_type: str = "application/octet-stream") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    print(f"  Uploaded: s3://{bucket}/{key}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n=== 2-Data-Preprocess: HR Analytics ===")
    s3 = boto3.client("s3")

    # Download raw CSV from S3
    df    = download_csv(s3, S3_BUCKET, RAW_S3_KEY)
    df_fe = feature_engineer(df)
    print(f"  New features: {[c for c in df_fe.columns if c not in df.columns]}")

    X_full, y = encode_features(df_fe)
    print(f"  Total encoded features: {X_full.shape[1]}")

    # Select the 8 manually chosen features
    available = [f for f in SELECTED_FEATURES if f in X_full.columns]
    missing   = [f for f in SELECTED_FEATURES if f not in X_full.columns]
    if missing:
        print(f"  WARNING — features not found after encoding: {missing}")
    X = X_full[available].copy()
    print(f"  Selected features ({len(available)}): {available}")

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # Upload NPZ splits
    for name, (Xs, ys) in [("train", (X_train, y_train)), ("test", (X_test, y_test))]:
        buf = io.BytesIO()
        np.savez_compressed(buf, X=Xs.astype(np.float32), y=ys.values.astype(int))
        buf.seek(0)
        upload_bytes(s3, S3_BUCKET, f"{OUT_PREFIX}/{name}.npz", buf.read())

    # Upload metadata
    meta = {
        "selected_features": available,
        "n_features": len(available),
        "n_classes": 2,
        "class_names": ["No Attrition", "Attrition"],
        "train_size": int(len(X_train)),
        "test_size":  int(len(X_test)),
        "random_state": RANDOM_STATE,
    }
    upload_bytes(s3, S3_BUCKET, f"{OUT_PREFIX}/metadata.json",
                 json.dumps(meta, indent=2).encode(), "application/json")

    print(f"\n  All files at: s3://{S3_BUCKET}/{OUT_PREFIX}/")
    print("=== Preprocessing complete ===\n")


if __name__ == "__main__":
    main()
