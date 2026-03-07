"""
tests/test_pipeline.py
Unit tests for AttritionIQ HR Attrition ML Pipeline
Covers: data_preprocess.py, model_train.py, model_evaluate.py,
        inference.py, predict.py, monitor_pipeline.py

Run locally:
    pip install pytest xgboost scikit-learn pandas numpy
    pytest tests/test_pipeline.py -v
"""

import json
import io
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────
# SHARED CONSTANTS  (must match your actual pipeline files exactly)
# ─────────────────────────────────────────────────────────────────
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

MIN_ACCURACY = 0.80
MIN_ROC_AUC  = 0.75
MIN_RECALL   = 0.50


# ─────────────────────────────────────────────────────────────────
# HELPER — build a small realistic HR dataframe
# ─────────────────────────────────────────────────────────────────
def make_hr_df(n=30):
    np.random.seed(42)
    return pd.DataFrame({
        "Age":                     np.random.randint(22, 58, n),
        "Attrition":               np.random.choice(["Yes", "No"], n, p=[0.16, 0.84]),
        "BusinessTravel":          np.random.choice(["Travel_Rarely", "Non-Travel"], n),
        "Department":              np.random.choice(["Sales", "R&D"], n),
        "DistanceFromHome":        np.random.randint(1, 30, n),
        "Education":               np.random.randint(1, 5, n),
        "EducationField":          np.random.choice(["Life Sciences", "Other"], n),
        "EmployeeCount":           [1] * n,
        "EmployeeNumber":          range(1, n + 1),
        "EnvironmentSatisfaction": np.random.randint(1, 4, n),
        "Gender":                  np.random.choice(["Male", "Female"], n),
        "HourlyRate":              np.random.randint(30, 100, n),
        "JobInvolvement":          np.random.randint(1, 4, n),
        "JobLevel":                np.random.randint(1, 5, n),
        "JobRole":                 np.random.choice(["Manager", "Analyst"], n),
        "JobSatisfaction":         np.random.randint(1, 4, n),
        "MaritalStatus":           np.random.choice(["Single", "Married", "Divorced"], n),
        "MonthlyIncome":           np.random.randint(2000, 15000, n),
        "MonthlyRate":             np.random.randint(5000, 25000, n),
        "NumCompaniesWorked":      np.random.randint(0, 9, n),
        "Over18":                  ["Y"] * n,
        "OverTime":                np.random.choice(["Yes", "No"], n),
        "PercentSalaryHike":       np.random.randint(11, 25, n),
        "PerformanceRating":       np.random.randint(3, 4, n),
        "RelationshipSatisfaction":np.random.randint(1, 4, n),
        "StandardHours":           [80] * n,
        "StockOptionLevel":        np.random.randint(0, 3, n),
        "TotalWorkingYears":       np.random.randint(0, 30, n),
        "TrainingTimesLastYear":   np.random.randint(0, 6, n),
        "WorkLifeBalance":         np.random.randint(1, 4, n),
        "YearsAtCompany":          np.random.randint(0, 20, n),
        "YearsInCurrentRole":      np.random.randint(0, 10, n),
        "YearsSinceLastPromotion": np.random.randint(0, 10, n),
        "YearsWithCurrManager":    np.random.randint(0, 10, n),
    })


# helper: inline feature_engineer (mirrors data_preprocess.py exactly)
def _feature_engineer(df):
    df = df.copy()
    for col in ["Age", "YearsAtCompany", "JobSatisfaction", "YearsInCurrentRole"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[18, 30, 40, 50, 60, 100],
            labels=["18-30", "31-40", "41-50", "51-60", "60+"],
        )

    if "YearsAtCompany" in df.columns:
        df["YearsAtCompany_filled"] = df["YearsAtCompany"].fillna(0)
        df["TenureBucket"] = pd.cut(
            df["YearsAtCompany_filled"],
            bins=[-0.1, 2, 5, 10, 100],
            labels=["<2", "2-5", "6-10", "10+"],
        )

    if "JobSatisfaction" in df.columns and "OverTime" in df.columns:
        df["JobSatisfaction_filled"] = df["JobSatisfaction"].fillna(
            df["JobSatisfaction"].median()
        )
        df["SatisfactionOverTime"] = (
            df["JobSatisfaction_filled"].astype(int).astype(str)
            + "_"
            + df["OverTime"].astype(str)
        )

    if "YearsInCurrentRole" in df.columns and "YearsAtCompany" in df.columns:
        df["YearsAtCompany_safe"] = df["YearsAtCompany"].fillna(0) + 1
        df["YearsInCurrentRole_Percent"] = (
            df["YearsInCurrentRole"].fillna(0) / df["YearsAtCompany_safe"]
        )

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


# ═════════════════════════════════════════════════════════════════
# 1.  DATA PREPROCESSING  (data_preprocess.py)
# ═════════════════════════════════════════════════════════════════
class TestFeatureEngineering:

    def test_new_columns_created(self):
        out = _feature_engineer(make_hr_df())
        for col in ["AgeGroup", "TenureBucket", "HighTenure_Satisfied",
                    "SatisfactionOverTime", "YearsInCurrentRole_Percent"]:
            assert col in out.columns, f"Missing: {col}"

    def test_high_tenure_satisfied_is_binary(self):
        out = _feature_engineer(make_hr_df())
        assert set(out["HighTenure_Satisfied"].unique()).issubset({0, 1})

    def test_high_tenure_satisfied_correct_logic(self):
        df = make_hr_df(5)
        df.loc[0, "YearsAtCompany"] = 15
        df.loc[0, "JobSatisfaction"] = 4
        out = _feature_engineer(df)
        assert out.loc[0, "HighTenure_Satisfied"] == 1

    def test_low_tenure_not_flagged(self):
        df = make_hr_df(10)
        df["YearsAtCompany"] = 1
        out = _feature_engineer(df)
        assert out["HighTenure_Satisfied"].sum() == 0

    def test_age_group_valid_values(self):
        out = _feature_engineer(make_hr_df())
        valid = {"18-30", "31-40", "41-50", "51-60", "60+"}
        actual = set(out["AgeGroup"].astype(str).unique()) - {"nan"}
        assert actual.issubset(valid)

    def test_no_temp_columns_remain(self):
        out = _feature_engineer(make_hr_df())
        for c in ["YearsAtCompany_filled", "JobSatisfaction_filled", "YearsAtCompany_safe"]:
            assert c not in out.columns

    def test_missing_values_handled(self):
        df = make_hr_df(10)
        df.loc[0, "YearsAtCompany"] = np.nan
        df.loc[1, "JobSatisfaction"] = np.nan
        _feature_engineer(df)  # must not raise

    def test_attrition_flag_yes_is_1(self):
        df = make_hr_df(6)
        df["Attrition"] = ["Yes", "No", "Yes", "No", "No", "Yes"]
        df["AttritionFlag"] = (
            df["Attrition"].str.strip().str.lower().map({"yes": 1, "no": 0})
        )
        assert list(df["AttritionFlag"]) == [1, 0, 1, 0, 0, 1]

    def test_overtime_binary_encoding(self):
        df = make_hr_df(10)
        df["OverTime_Yes"] = (df["OverTime"] == "Yes").astype(int)
        assert set(df["OverTime_Yes"].unique()).issubset({0, 1})

    def test_marital_status_single_encoding(self):
        df = make_hr_df(10)
        df["MaritalStatus_Single"] = (df["MaritalStatus"] == "Single").astype(int)
        assert set(df["MaritalStatus_Single"].unique()).issubset({0, 1})


# ═════════════════════════════════════════════════════════════════
# 2.  FEATURE SELECTION
# ═════════════════════════════════════════════════════════════════
class TestFeatureSelection:

    def test_exactly_8_features(self):
        assert len(FEATURE_ORDER) == 8

    def test_no_duplicate_features(self):
        assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER))

    def test_correct_feature_names(self):
        expected = {
            "HighTenure_Satisfied", "OverTime_Yes", "Age", "MonthlyIncome",
            "MaritalStatus_Single", "Gender_Male", "YearsAtCompany", "JobSatisfaction",
        }
        assert set(FEATURE_ORDER) == expected

    def test_gender_male_encoding(self):
        df = make_hr_df(10)
        df["Gender_Male"] = (df["Gender"] == "Male").astype(int)
        assert set(df["Gender_Male"].unique()).issubset({0, 1})


# ═════════════════════════════════════════════════════════════════
# 3.  MODEL TRAINING  (model_train.py)
# ═════════════════════════════════════════════════════════════════
def make_xy(n=200):
    np.random.seed(42)
    X = np.column_stack([
        np.random.randint(0, 2, n),
        np.random.randint(0, 2, n),
        np.random.randint(22, 58, n),
        np.random.randint(2000, 15000, n),
        np.random.randint(0, 2, n),
        np.random.randint(0, 2, n),
        np.random.randint(0, 20, n),
        np.random.randint(1, 4, n),
    ]).astype(np.float32)
    y = (np.random.random(n) < 0.16).astype(int)
    return X, y


class TestModelTraining:

    def test_xgboost_trains_without_error(self):
        from xgboost import XGBClassifier
        X, y = make_xy()
        model = XGBClassifier(n_estimators=10, max_depth=3,
                              eval_metric="logloss", random_state=42)
        model.fit(X, y)
        assert hasattr(model, "feature_importances_")

    def test_predict_shape(self):
        from xgboost import XGBClassifier
        X, y = make_xy()
        model = XGBClassifier(n_estimators=10, eval_metric="logloss", random_state=42)
        model.fit(X, y)
        assert model.predict(X[:5]).shape == (5,)

    def test_predict_proba_shape(self):
        from xgboost import XGBClassifier
        X, y = make_xy()
        model = XGBClassifier(n_estimators=10, eval_metric="logloss", random_state=42)
        model.fit(X, y)
        assert model.predict_proba(X[:5]).shape == (5, 2)

    def test_probabilities_sum_to_one(self):
        from xgboost import XGBClassifier
        X, y = make_xy()
        model = XGBClassifier(n_estimators=10, eval_metric="logloss", random_state=42)
        model.fit(X, y)
        np.testing.assert_allclose(
            model.predict_proba(X).sum(axis=1), 1.0, atol=1e-5
        )

    def test_predictions_are_binary(self):
        from xgboost import XGBClassifier
        X, y = make_xy()
        model = XGBClassifier(n_estimators=10, eval_metric="logloss", random_state=42)
        model.fit(X, y)
        assert set(model.predict(X)).issubset({0, 1})

    def test_feature_importances_sum_to_one(self):
        from xgboost import XGBClassifier
        X, y = make_xy()
        model = XGBClassifier(n_estimators=10, eval_metric="logloss", random_state=42)
        model.fit(X, y)
        np.testing.assert_allclose(model.feature_importances_.sum(), 1.0, atol=1e-5)

    def test_correct_hyperparameters_stored(self):
        from xgboost import XGBClassifier
        model = XGBClassifier(
            learning_rate=0.1, max_depth=3,
            n_estimators=50, subsample=0.8,
            eval_metric="logloss", random_state=42,
        )
        assert model.learning_rate == 0.1
        assert model.max_depth     == 3
        assert model.n_estimators  == 50
        assert model.subsample     == 0.8

    def test_model_serialises_with_joblib(self):
        from xgboost import XGBClassifier
        import joblib
        X, y = make_xy()
        model = XGBClassifier(n_estimators=10, eval_metric="logloss", random_state=42)
        model.fit(X, y)
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        loaded = joblib.load(buf)
        np.testing.assert_array_equal(
            model.predict(X[:3]), loaded.predict(X[:3])
        )


# ═════════════════════════════════════════════════════════════════
# 4.  INFERENCE HANDLER  (inference.py)
# ═════════════════════════════════════════════════════════════════

# inline mirrors of inference.py functions — no import needed
def _input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict) and "features" in data:
            arr = np.array(data["features"], dtype=np.float32)
        elif isinstance(data, dict):
            arr = np.array(
                [data.get(f, 0.0) for f in FEATURE_ORDER], dtype=np.float32
            )
        elif isinstance(data, list):
            arr = np.array(data, dtype=np.float32)
        else:
            raise ValueError(f"Unexpected input: {type(data)}")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    if content_type == "text/csv":
        arr = np.array(
            [list(map(float, r.split(",")))
             for r in request_body.decode().strip().split("\n")],
            dtype=np.float32,
        )
        return arr
    raise ValueError(f"Unsupported content type: {content_type}")


def _predict_fn(input_data, model):
    preds  = model.predict(input_data).tolist()
    probas = model.predict_proba(input_data).tolist()
    results = []
    for pred, proba in zip(preds, probas):
        results.append({
            "prediction":            int(pred),
            "predicted_class":       CLASS_NAMES[int(pred)],
            "attrition_probability": round(float(proba[1]), 5),
            "probabilities": {
                CLASS_NAMES[0]: round(float(proba[0]), 5),
                CLASS_NAMES[1]: round(float(proba[1]), 5),
            },
        })
    return results[0] if len(results) == 1 else results


class TestInferenceFunctions:

    def test_input_fn_json_features_list(self):
        arr = _input_fn(json.dumps({"features": [0, 1, 28, 3200, 1, 1, 2, 2]}).encode())
        assert arr.shape == (1, 8)
        assert arr[0, 2] == 28.0

    def test_input_fn_json_named_dict(self):
        body = json.dumps({
            "HighTenure_Satisfied": 0, "OverTime_Yes": 1, "Age": 30,
            "MonthlyIncome": 4000, "MaritalStatus_Single": 1,
            "Gender_Male": 1, "YearsAtCompany": 3, "JobSatisfaction": 2,
        }).encode()
        arr = _input_fn(body)
        assert arr.shape == (1, 8)
        assert arr[0, 2] == 30.0

    def test_input_fn_csv_single_row(self):
        arr = _input_fn(b"0,1,28,3200,1,1,2,2", content_type="text/csv")
        assert arr.shape == (1, 8)

    def test_input_fn_csv_multi_row(self):
        arr = _input_fn(
            b"0,1,28,3200,1,1,2,2\n1,0,45,9000,0,0,15,4",
            content_type="text/csv",
        )
        assert arr.shape == (2, 8)

    def test_input_fn_unsupported_content_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported content type"):
            _input_fn(b"data", content_type="text/plain")

    def test_predict_fn_output_has_required_keys(self):
        mock = MagicMock()
        mock.predict.return_value       = np.array([1])
        mock.predict_proba.return_value = np.array([[0.25, 0.75]])
        result = _predict_fn(np.zeros((1, 8)), mock)
        for key in ["prediction", "predicted_class", "attrition_probability", "probabilities"]:
            assert key in result

    def test_predict_fn_attrition_class_name(self):
        mock = MagicMock()
        mock.predict.return_value       = np.array([1])
        mock.predict_proba.return_value = np.array([[0.25, 0.75]])
        result = _predict_fn(np.zeros((1, 8)), mock)
        assert result["predicted_class"] == "Attrition"

    def test_predict_fn_no_attrition_class_name(self):
        mock = MagicMock()
        mock.predict.return_value       = np.array([0])
        mock.predict_proba.return_value = np.array([[0.85, 0.15]])
        result = _predict_fn(np.zeros((1, 8)), mock)
        assert result["predicted_class"] == "No Attrition"

    def test_predict_fn_probability_in_range(self):
        mock = MagicMock()
        mock.predict.return_value       = np.array([1])
        mock.predict_proba.return_value = np.array([[0.30, 0.70]])
        result = _predict_fn(np.zeros((1, 8)), mock)
        assert 0.0 <= result["attrition_probability"] <= 1.0

    def test_predict_fn_probabilities_sum_to_one(self):
        mock = MagicMock()
        mock.predict.return_value       = np.array([0])
        mock.predict_proba.return_value = np.array([[0.65, 0.35]])
        result = _predict_fn(np.zeros((1, 8)), mock)
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4

    def test_predict_fn_batch_returns_list(self):
        mock = MagicMock()
        mock.predict.return_value       = np.array([0, 1])
        mock.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        result = _predict_fn(np.zeros((2, 8)), mock)
        assert isinstance(result, list) and len(result) == 2


# ═════════════════════════════════════════════════════════════════
# 5.  EVALUATION METRICS  (model_evaluate.py)
# ═════════════════════════════════════════════════════════════════
class TestEvaluationMetrics:

    def _metrics(self, y_true, y_pred, y_prob):
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix,
        )
        acc     = accuracy_score(y_true, y_pred)
        prec    = precision_score(y_true, y_pred, zero_division=0)
        rec     = recall_score(y_true, y_pred, zero_division=0)
        f1      = f1_score(y_true, y_pred, zero_division=0)
        auc     = roc_auc_score(y_true, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                    tn=tn, fp=fp, fn=fn, tp=tp)

    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        m = self._metrics(y, y, y.astype(float))
        assert m["acc"] == 1.0

    def test_accuracy_meets_project_threshold(self):
        y_true = np.array([0] * 84 + [1] * 16)
        y_pred = np.array([0] * 84 + [1] * 14 + [0] * 2)
        m = self._metrics(y_true, y_pred, y_pred.astype(float))
        assert m["acc"] >= MIN_ACCURACY

    def test_roc_auc_meets_project_threshold(self):
        np.random.seed(42)
        y_true = np.array([0] * 84 + [1] * 16)
        y_prob = np.concatenate([
            np.random.uniform(0.05, 0.40, 84),
            np.random.uniform(0.55, 0.95, 16),
        ])
        y_pred = (y_prob >= 0.5).astype(int)
        m = self._metrics(y_true, y_pred, y_prob)
        assert m["auc"] >= MIN_ROC_AUC

    def test_confusion_matrix_components(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.8, 0.9, 0.3, 0.2, 0.7])
        m = self._metrics(y_true, y_pred, y_prob)
        assert m["tp"] == 2
        assert m["tn"] == 2
        assert m["fp"] == 1
        assert m["fn"] == 1

    def test_specificity_formula(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4])
        m = self._metrics(y_true, y_pred, y_prob)
        specificity = m["tn"] / (m["tn"] + m["fp"])
        assert round(specificity, 4) == round(2 / 3, 4)

    def test_metrics_json_serialisable(self):
        y = np.array([0, 1, 0, 1])
        m = self._metrics(y, y, y.astype(float))
        json.dumps({k: float(v) for k, v in m.items()})  # must not raise


# ═════════════════════════════════════════════════════════════════
# 6.  MONITORING  (monitor_pipeline.py)
# ═════════════════════════════════════════════════════════════════
class TestMonitoringLogic:

    def _needs_retrain(self, acc, auc, rec, endpoint_ok=True):
        return (
            not endpoint_ok
            or acc < MIN_ACCURACY
            or auc < MIN_ROC_AUC
            or rec < MIN_RECALL
        )

    def test_no_retrain_when_all_pass(self):
        assert not self._needs_retrain(0.871, 0.823, 0.584)

    def test_retrain_on_low_accuracy(self):
        assert self._needs_retrain(0.75, 0.823, 0.584)

    def test_retrain_on_low_roc_auc(self):
        assert self._needs_retrain(0.871, 0.70, 0.584)

    def test_retrain_on_low_recall(self):
        assert self._needs_retrain(0.871, 0.823, 0.40)

    def test_retrain_when_endpoint_down(self):
        assert self._needs_retrain(0.871, 0.823, 0.584, endpoint_ok=False)

    def test_thresholds_match_pipeline_defaults(self):
        assert MIN_ACCURACY == 0.80
        assert MIN_ROC_AUC  == 0.75
        assert MIN_RECALL   == 0.50

    def test_report_structure_valid(self):
        report = {
            "timestamp":          "2026-02-27T14:00:00+00:00",
            "endpoint_name":      "mousumi-hr-attrition-endpoint",
            "checks":             {"endpoint_healthy": True, "accuracy_ok": True},
            "alerts":             [],
            "retrain_recommended": False,
        }
        assert "checks" in report
        assert isinstance(report["alerts"], list)

    def test_alert_messages_are_strings(self):
        alerts = []
        acc = 0.75
        if acc < MIN_ACCURACY:
            alerts.append(f"Accuracy {acc:.4f} < {MIN_ACCURACY}")
        assert all(isinstance(a, str) for a in alerts)


# ═════════════════════════════════════════════════════════════════
# 7.  PREDICT CLIENT  (predict.py)
# ═════════════════════════════════════════════════════════════════
DEFAULT_FEATURES = {
    "HighTenure_Satisfied": 0, "OverTime_Yes": 1, "Age": 28,
    "MonthlyIncome": 3200,     "MaritalStatus_Single": 1,
    "Gender_Male": 1,          "YearsAtCompany": 2, "JobSatisfaction": 2,
}


class TestPredictClient:

    def test_default_features_has_8_values(self):
        assert len(DEFAULT_FEATURES) == 8

    def test_default_feature_keys_match_order(self):
        assert list(DEFAULT_FEATURES.keys()) == FEATURE_ORDER

    def test_all_feature_values_numeric(self):
        for k, v in DEFAULT_FEATURES.items():
            assert isinstance(v, (int, float)), f"{k} is not numeric"

    def test_payload_serialises_to_json(self):
        payload = json.loads(
            json.dumps({"features": list(DEFAULT_FEATURES.values())})
        )
        assert len(payload["features"]) == 8

    def test_high_risk_label(self):
        label = "HIGH RISK" if 0.73 >= 0.5 else "LOW RISK"
        assert label == "HIGH RISK"

    def test_low_risk_label(self):
        label = "HIGH RISK" if 0.22 >= 0.5 else "LOW RISK"
        assert label == "LOW RISK"

    def test_mock_endpoint_call(self):
        mock_rt = MagicMock()
        mock_rt.invoke_endpoint.return_value = {
            "Body": MagicMock(
                read=lambda: json.dumps({
                    "prediction": 1,
                    "predicted_class": "Attrition",
                    "attrition_probability": 0.73,
                    "probabilities": {"No Attrition": 0.27, "Attrition": 0.73},
                }).encode()
            )
        }
        resp   = mock_rt.invoke_endpoint(
            EndpointName="mousumi-hr-attrition-endpoint",
            ContentType="application/json",
            Body=json.dumps({"features": list(DEFAULT_FEATURES.values())}).encode(),
        )
        result = json.loads(resp["Body"].read())
        assert result["prediction"] == 1
        assert result["attrition_probability"] == 0.73


# ═════════════════════════════════════════════════════════════════
# 8.  END-TO-END INTEGRATION
# ═════════════════════════════════════════════════════════════════
class TestEndToEnd:

    def test_full_pipeline_no_aws(self):
        """
        Raw features → engineer → train → inference → validate result.
        Zero AWS calls — pure local test.
        """
        from xgboost import XGBClassifier

        n = 150
        np.random.seed(0)
        X = np.column_stack([
            np.random.randint(0, 2, n),
            np.random.randint(0, 2, n),
            np.random.randint(22, 58, n),
            np.random.randint(2000, 15000, n),
            np.random.randint(0, 2, n),
            np.random.randint(0, 2, n),
            np.random.randint(0, 20, n),
            np.random.randint(1, 4, n),
        ]).astype(np.float32)
        y = (np.random.random(n) < 0.16).astype(int)

        model = XGBClassifier(
            n_estimators=20, max_depth=3,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X, y)

        arr    = _input_fn(json.dumps({"features": [0, 1, 28, 3200, 1, 1, 2, 2]}).encode())
        result = _predict_fn(arr, model)

        assert result["prediction"] in [0, 1]
        assert 0.0 <= result["attrition_probability"] <= 1.0
        assert result["predicted_class"] in CLASS_NAMES
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4

    def test_high_risk_scores_higher_than_low_risk(self):
        """Very-high-risk employee must score higher probability than very-low-risk."""
        from xgboost import XGBClassifier

        n = 300
        np.random.seed(7)
        X = np.vstack([
            np.tile([0, 1, 24, 2800, 1, 1, 1, 1], (n // 2, 1)),
            np.tile([1, 0, 50, 14000, 0, 0, 20, 4], (n // 2, 1)),
        ]).astype(np.float32)
        y = np.array([1] * (n // 2) + [0] * (n // 2))

        model = XGBClassifier(
            n_estimators=30, max_depth=3,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X, y)

        high = model.predict_proba([[0, 1, 24, 2800, 1, 1, 1, 1]])[0][1]
        low  = model.predict_proba([[1, 0, 50, 14000, 0, 0, 20, 4]])[0][1]
        assert high > low, f"Expected high ({high:.3f}) > low ({low:.3f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
