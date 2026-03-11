"""
data_preprocess.py  — HR Attrition · Imbalance-Aware Preprocessing v2
═══════════════════════════════════════════════════════════════════════
ADDITIONS vs original:
  1. Uploads class_weights to metadata.json (for model_train to use)
  2. Uploads drift_baseline.json to S3 with IBM HR reference statistics:
       - OverTime rate baseline (28.3%)
       - Monthly income stats (mean, median, std)
       - Attrition rate by key features
       - Age/tenure distribution stats
     → Used by monitor dashboard to detect drift in production batches
  3. Logs complete class imbalance summary to console

Prerequisites:
    pip install boto3 pandas numpy scikit-learn joblib
    export S3_BUCKET=term3-mousumi
    python src/data/data_preprocess.py
"""

import os, sys, io, json
import boto3, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET    = os.environ["S3_BUCKET"]
PROJECT      = os.environ.get("PROJECT_NAME", "mousumi-hr-attrition")
RAW_PREFIX   = f"{PROJECT}/data/raw"
OUT_PREFIX   = f"{PROJECT}/data/processed"
MODEL_PREFIX = f"{PROJECT}/models"
RANDOM_STATE = 42

MANUAL_SELECTED = [
    "HighTenure_Satisfied", "OverTime_Yes", "Age", "MonthlyIncome",
    "MaritalStatus_Single", "Gender_Male", "YearsAtCompany", "JobSatisfaction",
]


# ── Feature Engineering ───────────────────────────────────────────────────────

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

    if "Age" in df.columns:
        df["AgeGroup"] = safe_cut(df["Age"],
            bins=[18,30,40,50,60,100], labels=["18-30","31-40","41-50","51-60","60+"])

    if "YearsAtCompany" in df.columns:
        df["YearsAtCompany_filled"] = df["YearsAtCompany"].fillna(0)
        df["TenureBucket"] = pd.cut(df["YearsAtCompany_filled"],
            bins=[-0.1,2,5,10,100], labels=["<2","2-5","6-10","10+"])

    if "JobSatisfaction" in df.columns and "OverTime" in df.columns:
        df["JobSatisfaction_filled"] = df["JobSatisfaction"].fillna(df["JobSatisfaction"].median())
        df["SatisfactionOverTime"] = (
            df["JobSatisfaction_filled"].astype(int).astype(str) + "_" + df["OverTime"].astype(str))

    if "YearsInCurrentRole" in df.columns and "YearsAtCompany" in df.columns:
        df["YearsAtCompany_safe"] = df["YearsAtCompany"].fillna(0) + 1
        df["YearsInCurrentRole_Percent"] = df["YearsInCurrentRole"].fillna(0) / df["YearsAtCompany_safe"]

    if "TenureBucket" in df.columns and "JobSatisfaction" in df.columns:
        df["HighTenure_Satisfied"] = (
            (df["TenureBucket"] == "10+") & (df["JobSatisfaction"].fillna(0) >= 3)
        ).astype(int)
    else:
        df["HighTenure_Satisfied"] = 0

    for c in ["YearsAtCompany_filled", "JobSatisfaction_filled", "YearsAtCompany_safe"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    return df


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    if "Attrition" not in df.columns:
        raise KeyError("'Attrition' column not found in dataset.")

    df["AttritionFlag"] = (
        df["Attrition"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    )
    if df["AttritionFlag"].isnull().any():
        df["AttritionFlag"] = pd.to_numeric(df["Attrition"], errors="coerce").fillna(df["AttritionFlag"])

    cat_cols = [c for c in [
        "BusinessTravel","Department","EducationField","Gender",
        "JobRole","MaritalStatus","OverTime","Over18",
        "AgeGroup","TenureBucket","SatisfactionOverTime",
    ] if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    for col in df_encoded.select_dtypes(include="bool").columns:
        df_encoded[col] = df_encoded[col].astype(int)

    all_features = [c for c in df_encoded.columns if c not in ["Attrition", "AttritionFlag"]]
    X = df_encoded[all_features]
    y = df_encoded["AttritionFlag"].fillna(0).astype(int)
    return X, y, df_encoded


# ── Drift Baseline Computation ────────────────────────────────────────────────

def compute_drift_baseline(df: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compute IBM HR dataset reference statistics.
    Stored in S3 so the monitor pipeline can compare future batch
    distributions and detect feature/concept drift.
    """
    baseline = {
        "dataset":         "IBM HR Analytics",
        "n_total":         int(len(df)),
        "n_attrition":     int(y.sum()),
        "n_no_attrition":  int((y == 0).sum()),
        "attrition_rate":  round(float(y.mean()), 4),
        "class_ratio":     round(float((y==0).sum() / y.sum()), 4),  # neg/pos ≈ 5.06
    }

    # OverTime rate — key feature for drift
    if "OverTime" in df.columns:
        ot_rate = (df["OverTime"].astype(str).str.lower() == "yes").mean()
        baseline["overtime_rate"]              = round(float(ot_rate), 4)
        baseline["overtime_attrition_rate"]    = round(float(y[df["OverTime"].str.lower()=="yes"].mean()), 4)
        baseline["no_overtime_attrition_rate"] = round(float(y[df["OverTime"].str.lower()=="no"].mean()),  4)

    # Monthly income distribution
    if "MonthlyIncome" in df.columns:
        inc = df["MonthlyIncome"].dropna()
        baseline["income_mean"]   = round(float(inc.mean()),   2)
        baseline["income_median"] = round(float(inc.median()), 2)
        baseline["income_std"]    = round(float(inc.std()),    2)
        baseline["income_q25"]    = round(float(inc.quantile(0.25)), 2)
        baseline["income_q75"]    = round(float(inc.quantile(0.75)), 2)

    # Age distribution
    if "Age" in df.columns:
        age = df["Age"].dropna()
        baseline["age_mean"]   = round(float(age.mean()),   2)
        baseline["age_median"] = round(float(age.median()), 2)
        baseline["age_std"]    = round(float(age.std()),    2)

    # YearsAtCompany distribution
    if "YearsAtCompany" in df.columns:
        tenure = df["YearsAtCompany"].dropna()
        baseline["tenure_mean"]   = round(float(tenure.mean()),   2)
        baseline["tenure_median"] = round(float(tenure.median()), 2)
        baseline["tenure_std"]    = round(float(tenure.std()),    2)

    # Gender split
    if "Gender" in df.columns:
        baseline["gender_male_rate"] = round(float((df["Gender"].str.lower()=="male").mean()), 4)

    # MaritalStatus split
    if "MaritalStatus" in df.columns:
        ms = df["MaritalStatus"].str.lower()
        baseline["marital_single_rate"]  = round(float((ms=="single").mean()),  4)
        baseline["marital_married_rate"] = round(float((ms=="married").mean()), 4)
        baseline["marital_divorced_rate"]= round(float((ms=="divorced").mean()),4)

    # Feature drift thresholds (±20% of baseline = drift alert)
    baseline["drift_thresholds"] = {
        "overtime_rate":   [round(float(baseline.get("overtime_rate",0.28)*0.8), 4),
                            round(float(baseline.get("overtime_rate",0.28)*1.2), 4)],
        "attrition_rate":  [round(float(baseline["attrition_rate"]*0.7), 4),
                            round(float(baseline["attrition_rate"]*1.3), 4)],
        "income_mean":     [round(float(baseline.get("income_mean",6500)*0.85), 2),
                            round(float(baseline.get("income_mean",6500)*1.15), 2)],
        "age_mean":        [round(float(baseline.get("age_mean",36)*0.9), 2),
                            round(float(baseline.get("age_mean",36)*1.1), 2)],
    }

    return baseline


# ── S3 helpers ────────────────────────────────────────────────────────────────

def download_raw_csv(s3, bucket, prefix) -> pd.DataFrame:
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".csv")]
    if not keys:
        print(f"  ERROR: No CSV at s3://{bucket}/{prefix}/")
        sys.exit(1)
    key = sorted(keys)[0]
    print(f"  Downloading: s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def upload_bytes(s3, bucket, key, data, content_type="application/octet-stream"):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    print(f"  Uploaded: s3://{bucket}/{key}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    s3     = boto3.client("s3", region_name=region)

    print("\n=== HR Analytics — Data Preprocessing (Imbalance-Aware v2) ===")

    df    = download_raw_csv(s3, S3_BUCKET, RAW_PREFIX)
    df_fe = feature_engineer(df)
    print(f"  Engineered features: {[c for c in df_fe.columns if c not in df.columns]}")

    X_all, y_all, df_encoded = prepare_features(df_fe)
    print(f"  Total features after encoding: {X_all.shape[1]}")

    # Class imbalance summary
    n_pos = int(y_all.sum())
    n_neg = int((y_all == 0).sum())
    ratio = n_neg / n_pos
    print(f"\n  CLASS IMBALANCE SUMMARY:")
    print(f"    Total employees   : {len(y_all)}")
    print(f"    No Attrition (0)  : {n_neg}  ({100*n_neg/len(y_all):.1f}%)")
    print(f"    Attrition    (1)  : {n_pos}  ({100*n_pos/len(y_all):.1f}%)")
    print(f"    Ratio (neg/pos)   : {ratio:.2f}:1  →  use scale_pos_weight={ratio:.2f} in XGBoost")
    print(f"    Recommended SMOTE : {'YES (set ENABLE_SMOTE=true)' if ratio > 5 else 'OPTIONAL'}")

    # Build feature subset
    available = [f for f in MANUAL_SELECTED if f in X_all.columns]
    missing   = [f for f in MANUAL_SELECTED if f not in X_all.columns]
    if missing:
        print(f"  Warning: Features not found after encoding: {missing}")
    X_selected = X_all[available].copy()
    print(f"  Selected feature matrix: {X_selected.shape}")

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_STATE
    )
    print(f"  Split → train: {len(X_train)}  test: {len(X_test)}")
    print(f"    Train pos rate: {y_train.mean():.3f}  Test pos rate: {y_test.mean():.3f}")

    # Upload splits as NPZ
    for split_name, (X, y) in [("train", (X_train, y_train)), ("test", (X_test, y_test))]:
        buf = io.BytesIO()
        np.savez_compressed(buf, X=X, y=y)
        buf.seek(0)
        upload_bytes(s3, S3_BUCKET, f"{OUT_PREFIX}/{split_name}.npz", buf.read())

    # Compute drift baseline from full dataset (before split)
    print("\n  Computing drift baseline statistics...")
    drift_baseline = compute_drift_baseline(df, y_all)
    print(f"    OverTime rate      : {drift_baseline.get('overtime_rate', 'N/A'):.3f}")
    print(f"    Income mean        : {drift_baseline.get('income_mean', 'N/A'):.0f}")
    print(f"    Age mean           : {drift_baseline.get('age_mean', 'N/A'):.1f}")

    # Upload metadata with imbalance info
    meta = {
        "selected_features": available,
        "n_features":        int(X_selected.shape[1]),
        "n_classes":         2,
        "class_names":       ["No Attrition", "Attrition"],
        "train_size":        int(len(X_train)),
        "test_size":         int(len(X_test)),
        "target_col":        "AttritionFlag",
        "random_state":      RANDOM_STATE,
        # Imbalance info for model_train.py
        "n_pos":             n_pos,
        "n_neg":             n_neg,
        "class_ratio":       round(ratio, 4),
        "pos_rate":          round(n_pos / len(y_all), 4),
        "scale_pos_weight":  round(ratio, 4),
        "imbalance_note":    f"Dataset is {100*n_pos/len(y_all):.1f}% positive. Use scale_pos_weight={ratio:.2f}",
    }
    upload_bytes(s3, S3_BUCKET, f"{OUT_PREFIX}/metadata.json",
                 json.dumps(meta, indent=2).encode(), "application/json")

    # Upload drift baseline for monitor pipeline
    upload_bytes(s3, S3_BUCKET, f"{MODEL_PREFIX}/drift_baseline.json",
                 json.dumps(drift_baseline, indent=2).encode(), "application/json")

    print(f"\n  ✅ All artifacts uploaded to s3://{S3_BUCKET}/{OUT_PREFIX}/")
    print(f"  ✅ Drift baseline  : s3://{S3_BUCKET}/{MODEL_PREFIX}/drift_baseline.json")
    print("=== Preprocessing complete ===\n")


if __name__ == "__main__":
    main()
