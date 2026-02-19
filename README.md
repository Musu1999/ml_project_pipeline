# HR Analytics — Attrition Prediction Pipeline
**GitHub Actions + AWS SageMaker | Term 2 Project**

Predicts employee attrition using **Tuned XGBoost** on 8 manually selected HR features.

---

## ⚠️ IMPORTANT — Before You Push

The CSV **must be committed to the repo** at `data/HR_Analytics_Data.csv`.

```bash
# In your local repo folder:
git add data/HR_Analytics_Data.csv
git add .
git commit -m "Add HR dataset and pipeline files"
git push origin main
```

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       ├── 1-data-download.yml       # Upload CSV → S3 raw/
│       ├── 2-data-preprocess.yml     # Feature engineering + split → S3
│       ├── 3-model-train.yml         # Train XGBoost → S3
│       ├── 4-model-evaluate.yml      # Evaluate on test set → S3
│       ├── 5-model-deploy.yml        # Deploy SageMaker endpoint
│       └── 6-monitor-pipeline.yml    # Health check every 6h
├── src/
│   ├── data/
│   │   ├── data_download.py          # Uploads data/HR_Analytics_Data.csv → S3
│   │   └── data_preprocess.py        # Feature engineering → train/test NPZ
│   ├── model/
│   │   └── model_train.py            # Tuned XGBoost training
│   ├── evaluation/
│   │   └── model_evaluate.py         # Full metrics evaluation
│   ├── deployment/
│   │   └── model_deploy.py           # SageMaker endpoint deployment
│   ├── inference/
│   │   └── inference.py              # SageMaker inference handler
│   └── monitoring/
│       └── monitor_pipeline.py       # Endpoint health + metrics check
└── data/
    └── HR_Analytics_Data.csv         # ← COMMIT THIS FILE
```

---

## GitHub Variables & Secrets

**Settings → Secrets and variables → Actions**

| Type | Name | Value |
|---|---|---|
| Variable | `AWS_REGION` | `ap-south-1` |
| Variable | `S3_BUCKET` | `term3-<username>` |
| Variable | `PROJECT_NAME` | `<username>-ml-crp-project-name` |
| Variable | `ENDPOINT_NAME` | `<username>-ml-crp-project-name-endpoint` |
| Secret | `AWS_ROLE_ARN` | `arn:aws:iam::251198450921:role/GitHubActionS3Role` |
| Secret | `SAGEMAKER_ROLE_ARN` | `arn:aws:iam::251198450921:role/SageMakerStudentExecutionRole` |

---

## Workflow Sequence

| Step | Workflow | Triggers | Action |
|---|---|---|---|
| 1 | `1-Data-Download` | manual / push to main | Reads `data/HR_Analytics_Data.csv` from repo → uploads to S3 `raw/` |
| 2 | `2-Data-Preprocess` | auto after step 1 | Feature engineering → 8-feature NPZ splits → S3 `processed/` |
| 3 | `3-Model-Train` | auto after step 2 | Trains Tuned XGBoost → uploads model PKL + metrics → S3 `models/` |
| 4 | `4-Model-Evaluate` | auto after step 3 | Full evaluation → uploads results → S3 `evaluation/` |
| 5 | `5-Model-Deploy` | auto after step 4 | Packages model.tar.gz → creates SageMaker endpoint (`ml.t2.medium`) |
| 6 | `6-Monitor-Pipeline` | auto after step 5 + every 6h | Endpoint health + metric threshold check → S3 `monitoring/` |

---

## Model Details

**Final model (only):** Tuned XGBoost  
`learning_rate=0.1, max_depth=3, n_estimators=50, subsample=0.8`  
Selected via GridSearchCV in the term2 notebook.

**8 manually selected features:**
1. `HighTenure_Satisfied` — engineered: tenure ≥ 10yr AND satisfaction ≥ 3
2. `OverTime_Yes` — works overtime
3. `Age`
4. `MonthlyIncome`
5. `MaritalStatus_Single`
6. `Gender_Male`
7. `YearsAtCompany`
8. `JobSatisfaction`

---

## S3 Layout

```
s3://<S3_BUCKET>/<PROJECT_NAME>/
├── data/raw/                   HR_Analytics_Data.csv
├── data/processed/             train.npz, test.npz, metadata.json
├── models/                     xgboost_model.pkl, training_metrics.json
├── evaluation/                 evaluation_results.json
├── deployment/                 model.tar.gz, deployment_info.json
├── monitoring/                 report-YYYY-MM-DD.json
└── code/                       inference.py
```
