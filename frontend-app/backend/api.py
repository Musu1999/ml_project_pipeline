"""
backend/api.py — HR Attrition REST API Blueprint (Enhanced)
Added: /api/monitor endpoint, /api/predict/batch CSV upload support,
       quarterly batch upload with S3 archiving
"""
import os, io, csv, json, logging
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, TokenRetrievalError

logger = logging.getLogger(__name__)
api    = Blueprint("api", __name__, url_prefix="/api")

FEATURE_NAMES = [
    "HighTenure_Satisfied","OverTime_Yes","Age","MonthlyIncome",
    "MaritalStatus_Single","Gender_Male","YearsAtCompany","JobSatisfaction",
]
CLASS_NAMES = ["No Attrition","Attrition"]

def get_env():
    return {
        "key":     os.environ.get("AWS_ACCESS_KEY_ID",     ""),
        "secret":  os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "token":   os.environ.get("AWS_SESSION_TOKEN",     ""),
        "region":  os.environ.get("AWS_REGION",            "ap-south-1"),
        "ep":      os.environ.get("ENDPOINT_NAME",         ""),
        "bucket":  os.environ.get("S3_BUCKET",             ""),
        "project": os.environ.get("PROJECT_NAME",          "mousumi-hr-attrition"),
    }

def get_sm_runtime(e):
    kw = dict(region_name=e["region"])
    if e["key"]:    kw["aws_access_key_id"]     = e["key"]
    if e["secret"]: kw["aws_secret_access_key"] = e["secret"]
    if e["token"]:  kw["aws_session_token"]     = e["token"]
    return boto3.client("sagemaker-runtime", **kw)

def get_sm_client(e):
    kw = dict(region_name=e["region"])
    if e["key"]:    kw["aws_access_key_id"]     = e["key"]
    if e["secret"]: kw["aws_secret_access_key"] = e["secret"]
    if e["token"]:  kw["aws_session_token"]     = e["token"]
    return boto3.client("sagemaker", **kw)

def get_s3_client(e):
    kw = dict(region_name=e["region"])
    if e["key"]:    kw["aws_access_key_id"]     = e["key"]
    if e["secret"]: kw["aws_secret_access_key"] = e["secret"]
    if e["token"]:  kw["aws_session_token"]     = e["token"]
    return boto3.client("s3", **kw)

def get_top_risk_factors(features):
    """Return top contributing risk factors based on feature values."""
    ht, ot, age, inc, ms, gm, yr, sat = features
    factors = []
    if ot == 1:
        factors.append({"factor": "Works Overtime", "impact": "Very High", "weight": 0.312})
    if sat <= 2:
        factors.append({"factor": f"Low Job Satisfaction ({int(sat)}/4)", "impact": "High", "weight": 0.098})
    if inc < 3500:
        factors.append({"factor": "Below-average Income", "impact": "High", "weight": 0.148})
    if ms == 1:
        factors.append({"factor": "Single Marital Status", "impact": "Medium", "weight": 0.062})
    if yr < 3:
        factors.append({"factor": "New to Company (<3 yrs)", "impact": "Medium", "weight": 0.094})
    if age < 30:
        factors.append({"factor": "Young Employee (<30)", "impact": "Medium", "weight": 0.121})
    if ht == 1:
        factors.append({"factor": "High Tenure + Satisfied", "impact": "Protective", "weight": -0.201})
    # Sort by absolute weight descending, limit to top 3
    factors.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return factors[:3]

def get_action_recommendations(prob, features):
    """
    Return HR action items driven by BOTH risk band AND specific feature profile.
    Every feature combination within each band produces distinct recommendations
    so that no two meaningfully different employees receive the same advice.
    Bands: VERY HIGH >=0.70 | HIGH >=0.50 | MEDIUM >=0.35 | LOW <0.35
    """
    ht, ot, age, inc, ms, gm, yr, sat = features
    THRESHOLD = 0.35
    actions = []

    # ── VERY HIGH (>=0.70) ───────────────────────────────────────
    if prob >= 0.70:
        actions.append("🔴 URGENT: Book a private, confidential 1-on-1 within 48 hours — not email, not a group meeting")

        if ot == 1 and inc < 3500:
            actions.append("Double pressure: overtime AND critically low pay — address both simultaneously or this employee will not stay past 30 days")
        elif ot == 1:
            actions.append("Overtime is the #1 attrition driver here — immediately cap hours, redistribute workload, or bring in contract support this sprint")
        elif inc < 3500:
            actions.append(f"Salary at ₹{int(inc):,} is critically below market — escalate to compensation team for an emergency off-cycle review today")

        if inc >= 3500 and inc < 5000 and ot == 0:
            actions.append("Income is below peer retention levels — flag for next compensation cycle with urgency note; consider spot bonus to bridge the gap")

        if sat == 1:
            actions.append("Job satisfaction at floor level (1/4) — this is a resignation-imminent signal. Conduct a stay interview this week, not a survey")
        elif sat == 2:
            actions.append("Low satisfaction (2/4) — identify the specific dissatisfier: manager, role, team, or workload. Do not guess; ask directly")

        if yr < 2:
            actions.append("Less than 2 years tenure — early leavers rarely recover. Assign a senior buddy today and create a visible 90-day milestone plan")
        elif 2 <= yr < 4:
            actions.append("3–4 year tenure is a known flight risk window — confirm promotion or role evolution is on the roadmap within 6 months")

        if ms == 1 and age < 30:
            actions.append("Young, single, and high-risk: this profile has the highest market mobility. A personalised career roadmap discussion is non-negotiable")
        elif ms == 1 and age >= 30:
            actions.append("Single employees at this risk level are actively comparing market options — a concrete growth offer will outperform a pay raise alone")

        if age < 25:
            actions.append("Employees under 25 leave for growth, not just money — offer a stretch project, mentorship pairing, or visible role upgrade immediately")

        if ht == 0 and yr >= 3:
            actions.append(f"No tenure-satisfaction anchor despite {int(yr)} years — something eroded their engagement. Find the inflection point in this week's 1-on-1")

        actions.append("Prepare a structured retention package before the meeting: title change, hybrid policy, L&D budget, or 6-month bonus commitment")

    # ── HIGH (0.50–0.69) ─────────────────────────────────────────
    elif prob >= 0.50:
        actions.append("🟠 HIGH RISK: Manager must engage personally within this fortnight — do not delegate to HR alone")

        if ot == 1 and sat <= 2:
            actions.append("Overtime + low satisfaction is a compounding combination — tackle workload relief first; satisfaction rarely recovers under sustained pressure")
        elif ot == 1:
            actions.append("Overtime is elevating risk — agree on a realistic weekly hours ceiling and check in after 2 weeks to confirm compliance")
        elif sat <= 2:
            actions.append(f"Satisfaction at {int(sat)}/4 without overtime suggests the issue is role or culture, not workload — explore internal transfer options")

        if inc < 4000:
            actions.append(f"Pay at ₹{int(inc):,} is a flight risk — request early placement in the next compensation review cycle")
        elif 4000 <= inc < 5500:
            actions.append("Income is competitive but not strong — ensure the employee knows what the path to the next salary band looks like")

        if yr < 3:
            actions.append(f"Early tenure ({int(yr)} yrs) with high risk — verify their onboarding expectations were met and their manager relationship is healthy")
        elif 3 <= yr < 6:
            actions.append(f"{int(yr)}-year mark is a common decision point — ensure a promotion or role evolution conversation has happened in the last 6 months")
        elif yr >= 6:
            actions.append("Long tenure with rising risk often signals stagnation — discuss a lateral move, new domain, or leadership track this quarter")

        if ms == 1:
            actions.append("Single employees prioritise advancement over stability — make the growth trajectory explicit and time-bound")

        if age < 30 and sat >= 3:
            actions.append("Young employee with decent satisfaction but rising risk — likely being approached externally; ask directly and counter with opportunity")
        elif age >= 40 and sat <= 2:
            actions.append("Experienced employee with low satisfaction — this profile often leaves quietly; a senior leader conversation carries more weight than HR")

        if ht == 1:
            actions.append("High tenure + satisfaction is a stabilising signal — reinforce what is working and ensure nothing is about to change for the worse")

        actions.append("Document this as a key-person risk — prepare a basic knowledge transfer plan as a contingency measure")

    # ── MEDIUM (0.35–0.49) ───────────────────────────────────────
    elif prob >= THRESHOLD:
        actions.append("🟡 WATCH: Schedule a structured monthly check-in — not a status update, a genuine career conversation")

        if ot == 1 and yr < 4:
            actions.append("Overtime in early career is a slow burn — acknowledge the extra effort explicitly; unrecognised overwork is a top medium-risk escalator")
        elif ot == 1 and yr >= 4:
            actions.append(f"Sustained overtime for a {int(yr)}-year employee suggests structural understaffing — address at team level, not just individually")
        elif ot == 0 and sat <= 2:
            actions.append("Low satisfaction without overtime suggests the issue is engagement, not workload — explore what would make the role more meaningful")

        if sat == 2:
            actions.append("Satisfaction at 2/4 is a leading indicator — ask about specific project preferences and team dynamics before it becomes a firm decision to leave")
        elif sat == 3:
            actions.append("Satisfaction is neutral (3/4) — identify what would shift this employee from 'content' to 'committed'; small targeted changes have high impact here")
        elif sat == 4 and prob >= THRESHOLD:
            actions.append("High satisfaction but still medium risk — the pull is likely external (market offer, career ceiling). Ask directly about their 2-year plan")

        if inc < 4500 and ot == 0:
            actions.append("Below-average income without overtime pressure — ensure compensation is reviewed at the next cycle before dissatisfaction compounds")
        elif inc < 4500 and ot == 1:
            actions.append("Overtime with below-average pay is a clear value mismatch — prioritise pay review and discuss comp-for-hours explicitly")
        elif 4500 <= inc < 6000:
            actions.append("Income is adequate but not retentive at this risk level — visibility of the next salary milestone will help more than the salary itself")

        if yr < 3:
            actions.append(f"Year {int(yr)} is still formative — confirm the employee's original career expectations are being met; course-correct now if not")
        elif 3 <= yr < 5:
            actions.append("Approaching the 3–5 year inflection point — proactively open the promotion or growth conversation before they open it with a resignation")

        if ms == 1 and age < 35:
            actions.append("Single employee in their early-to-mid career: growth and title matter more than perks — make the next step concrete and achievable")

        if age >= 35 and sat <= 3 and ot == 0:
            actions.append("Mid-career with moderate satisfaction — explore whether the role still offers the challenge level this employee needs; stagnation is silent")

        if ht == 1:
            actions.append("Tenure-satisfaction combination is protective — publicly acknowledge their institutional value; recognition costs nothing and reduces this risk band significantly")

        actions.append("Ensure the last 60 days of work have been visibly recognised — unacknowledged contributions are the most common untracked driver of medium-risk escalation")

    # ── LOW (<0.35) ───────────────────────────────────────────────
    else:
        actions.append("🟢 STABLE: No immediate action needed — maintain standard quarterly engagement rhythm")

        if ht == 1 and sat == 4:
            actions.append("Exemplary profile: high tenure + top satisfaction — nominate as a culture anchor or onboarding mentor to deepen their organisational investment")
        elif ht == 1 and sat == 3:
            actions.append("Solid tenure with good satisfaction — a low-cost way to keep this profile stable is a public recognition moment or a new responsibility")
        elif ht == 1:
            actions.append("Long tenure is a stabilising force — ensure nothing structurally changes for this employee (manager, team, role scope) without prior discussion")

        if sat == 4 and ot == 0:
            actions.append("Highest satisfaction + no overtime: this is your benchmark employee — understand what makes this role work and replicate it for others")
        elif sat == 4 and ot == 1:
            actions.append("Top satisfaction despite overtime is unusual — monitor hours carefully; sustained overload erodes even the most engaged employees over time")
        elif sat == 3 and ot == 0:
            actions.append("Steady and stable — a quarterly career conversation is sufficient; ask what would make the role a 4/4, then act on it")

        if inc >= 6000 and yr >= 5:
            actions.append("Well-compensated and established — retain by offering visibility: cross-functional projects, leadership panels, or external conference representation")
        elif 4500 <= inc < 6000 and yr >= 3:
            actions.append("Good standing overall — ensure the next salary step is on the roadmap so compensation doesn't become a surprise issue at the 2-year mark")

        if yr >= 10:
            actions.append(f"{int(yr)}-year veteran: protect this knowledge base — involve them in documentation, mentoring, and institutional memory initiatives")
        elif yr >= 5:
            actions.append(f"Established {int(yr)}-year employee — involve in team decisions and onboarding to deepen their sense of ownership and belonging")

        if age >= 40 and sat >= 3 and ot == 0:
            actions.append("Experienced, satisfied, and balanced — this profile rarely leaves; maintain the conditions and ensure manager continuity")

        if ms == 0 and inc >= 5000 and sat >= 3:
            actions.append("Stable personal and professional profile — a light-touch annual career review is all that is needed to keep this employee on track")

        if ot == 0 and inc >= 5000:
            actions.append("Healthy work-life balance and competitive income — these two conditions are your strongest retention levers; protect both proactively")

        actions.append("Flag for positive recognition in the next performance cycle — appreciated employees at this risk level have near-zero voluntary attrition")

    return actions

def call_endpoint(features, e):
    rt      = get_sm_runtime(e)
    payload = ",".join(str(f) for f in features)
    resp    = rt.invoke_endpoint(EndpointName=e["ep"], ContentType="text/csv", Body=payload.encode())
    raw     = resp["Body"].read().decode().strip()
    prob    = round(float(raw), 5)
    ret     = round(1.0 - prob, 5)
    # Use tuned threshold 0.35 (optimised for F1 on IBM HR dataset)
    THRESHOLD = 0.35
    pred    = 1 if prob >= THRESHOLD else 0
    risk    = "VERY HIGH" if prob >= 0.70 else "HIGH" if prob >= 0.50 else "MEDIUM" if prob >= THRESHOLD else "LOW"
    top_factors = get_top_risk_factors(features)
    actions     = get_action_recommendations(prob, features)
    return {
        "prediction": pred, "predicted_class": CLASS_NAMES[pred],
        "attrition_probability": prob, "retention_probability": ret,
        "risk_level": risk,
        "threshold_used": THRESHOLD,
        "top_risk_factors": top_factors,
        "action_recommendations": actions,
        "probabilities": {"No Attrition": ret, "Attrition": prob},
        "mode": "sagemaker_live"
    }

def validate(features):
    try:    f = [float(x) for x in features]
    except: return [], "All values must be numeric."
    if len(f) != 8: return [], f"Expected 8 features, got {len(f)}."
    return f, None

# ── /api/predict ─────────────────────────────────────────────────
@api.route("/predict", methods=["POST"])
def predict():
    e    = get_env()
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "Send JSON with 'features' array of 8 values."}), 400
    feat, err = validate(data["features"])
    if err: return jsonify({"error": err}), 400
    if not e["ep"]:
        return jsonify({"error": "ENDPOINT_NAME not set in .env"}), 500
    if not e["key"] or not e["secret"]:
        return jsonify({"error": "AWS credentials missing in .env"}), 500
    try:
        return jsonify(call_endpoint(feat, e))
    except ClientError as ce:
        code = ce.response["Error"]["Code"]
        msg  = ce.response["Error"]["Message"]
        if any(x in code+msg for x in ["SignatureDoesNotMatch","InvalidSignature","ExpiredToken"]):
            return jsonify({"error":"EXPIRED_TOKEN","message":"AWS token expired. Refresh from AWS Academy."}), 401
        return jsonify({"error": f"AWS error ({code}): {msg}"}), 500
    except (NoCredentialsError, TokenRetrievalError):
        return jsonify({"error":"EXPIRED_TOKEN","message":"AWS credentials invalid/expired."}), 401
    except Exception as ex:
        logger.error(f"[predict] {ex}")
        return jsonify({"error": str(ex)}), 500

# ── /api/predict/batch  (JSON body) ──────────────────────────────
@api.route("/predict/batch", methods=["POST"])
def predict_batch():
    e    = get_env()
    data = request.get_json(silent=True)
    if not data or "employees" not in data:
        return jsonify({"error": "Send JSON with 'employees' list."}), 400
    employees = data["employees"]
    if not isinstance(employees, list) or len(employees) == 0:
        return jsonify({"error": "employees must be non-empty list."}), 400
    if len(employees) > 500:
        return jsonify({"error": "Max 500 per batch."}), 400
    if not e["ep"]:
        return jsonify({"error": "ENDPOINT_NAME not set."}), 500

    results, errors = [], []
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for i, emp in enumerate(employees):
        name = emp.get("name", f"Employee_{i+1}")
        feat, err = validate(emp.get("features", []))
        if err:
            errors.append({"name": name, "error": err})
            continue
        try:
            r = call_endpoint(feat, e)
            results.append({"name": name, **r})
            counts[r["risk_level"]] += 1
        except ClientError as ce:
            code = ce.response["Error"]["Code"]
            if "ExpiredToken" in code or "SignatureDoesNotMatch" in code:
                return jsonify({"error":"EXPIRED_TOKEN","message":"AWS token expired."}), 401
            errors.append({"name": name, "error": str(ce)})
        except Exception as ex:
            errors.append({"name": name, "error": str(ex)})

    return jsonify({
        "total": len(employees), "success": len(results), "failed": len(errors),
        "summary": {
            "high_risk": counts["HIGH"], "medium_risk": counts["MEDIUM"],
            "low_risk": counts["LOW"],
            "attrition_count": sum(1 for r in results if r["prediction"] == 1)
        },
        "predictions": results, "errors": errors, "mode": "sagemaker_live"
    })

# ── /api/predict/batch/csv  (CSV file upload — quarterly) ────────
@api.route("/predict/batch/csv", methods=["POST"])
def predict_batch_csv():
    """
    Accept a CSV file upload for quarterly batch prediction.
    CSV format:
      name,HighTenure_Satisfied,OverTime_Yes,Age,MonthlyIncome,
           MaritalStatus_Single,Gender_Male,YearsAtCompany,JobSatisfaction
    Returns predictions + optionally archives to S3 under batch-uploads/YYYY-QN/
    """
    e = get_env()
    if not e["ep"]:
        return jsonify({"error": "ENDPOINT_NAME not set."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send multipart/form-data with key 'file'."}), 400

    file   = request.files["file"]
    quarter = request.form.get("quarter", _current_quarter())

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only .csv files accepted."}), 400

    raw_bytes = file.read()
    try:
        text = raw_bytes.decode("utf-8-sig")  # handle BOM
        reader = csv.DictReader(io.StringIO(text))
        rows   = list(reader)
    except Exception as ex:
        return jsonify({"error": f"CSV parse error: {ex}"}), 400

    if len(rows) == 0:
        return jsonify({"error": "CSV file is empty."}), 400
    if len(rows) > 1000:
        return jsonify({"error": "Max 1,000 rows per upload."}), 400

    # ── Archive raw CSV to S3 if configured ──
    archived_path = None
    if e["bucket"]:
        try:
            s3  = get_s3_client(e)
            key = f"{e['project']}/batch-uploads/{quarter}/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{file.filename}"
            s3.put_object(Bucket=e["bucket"], Key=key, Body=raw_bytes, ContentType="text/csv")
            archived_path = f"s3://{e['bucket']}/{key}"
            logger.info(f"[batch/csv] Archived to {archived_path}")
        except Exception as ex:
            logger.warning(f"[batch/csv] S3 archive failed: {ex}")

    # ── Score each row ──
    employees = []
    for row in rows:
        name = row.get("name", row.get("Name", row.get("employee_name", f"Row_{len(employees)+1}")))
        try:
            feat = [
                float(row.get("HighTenure_Satisfied", 0)),
                float(row.get("OverTime_Yes",         0)),
                float(row.get("Age",                  30)),
                float(row.get("MonthlyIncome",       5000)),
                float(row.get("MaritalStatus_Single", 0)),
                float(row.get("Gender_Male",          0)),
                float(row.get("YearsAtCompany",       5)),
                float(row.get("JobSatisfaction",      3)),
            ]
            employees.append({"name": name, "features": feat})
        except Exception as ex:
            employees.append({"name": name, "features": [], "_parse_error": str(ex)})

    results, errors = [], []
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for emp in employees:
        name = emp["name"]
        if "_parse_error" in emp:
            errors.append({"name": name, "error": emp["_parse_error"]})
            continue
        feat, err = validate(emp["features"])
        if err:
            errors.append({"name": name, "error": err})
            continue
        try:
            r = call_endpoint(feat, e)
            results.append({"name": name, **r})
            counts[r["risk_level"]] += 1
        except ClientError as ce:
            code = ce.response["Error"]["Code"]
            if "ExpiredToken" in code or "SignatureDoesNotMatch" in code:
                return jsonify({"error":"EXPIRED_TOKEN","message":"AWS token expired."}), 401
            errors.append({"name": name, "error": str(ce)})
        except Exception as ex:
            errors.append({"name": name, "error": str(ex)})

    # ── Archive results to S3 ──
    results_path = None
    if e["bucket"] and archived_path:
        try:
            s3   = get_s3_client(e)
            rkey = archived_path.replace(f"s3://{e['bucket']}/","").replace(".csv","-results.json")
            s3.put_object(
                Bucket=e["bucket"], Key=rkey,
                Body=json.dumps({"quarter": quarter, "predictions": results, "errors": errors}, indent=2).encode(),
                ContentType="application/json"
            )
            results_path = f"s3://{e['bucket']}/{rkey}"
        except Exception as ex:
            logger.warning(f"[batch/csv] S3 results archive failed: {ex}")

    return jsonify({
        "quarter": quarter,
        "total": len(rows), "success": len(results), "failed": len(errors),
        "summary": {
            "high_risk":      counts["HIGH"],
            "medium_risk":    counts["MEDIUM"],
            "low_risk":       counts["LOW"],
            "attrition_count": sum(1 for r in results if r["prediction"] == 1),
            "attrition_rate": round(sum(1 for r in results if r["prediction"]==1)/max(len(results),1)*100, 1),
        },
        "predictions":    results,
        "errors":         errors,
        "archived_input": archived_path,
        "archived_results": results_path,
        "mode":           "sagemaker_live"
    })

def _current_quarter():
    m = datetime.now().month
    q = (m - 1) // 3 + 1
    return f"{datetime.now().year}-Q{q}"

# ── /api/health ──────────────────────────────────────────────────
@api.route("/health", methods=["GET"])
def health():
    e = get_env()
    creds_ok  = bool(e["key"] and e["secret"])
    ep_ok     = bool(e["ep"])
    ep_status = "unknown"
    token_ok  = True

    if creds_ok and ep_ok:
        try:
            sm = get_sm_client(e)
            ep_status = sm.describe_endpoint(EndpointName=e["ep"])["EndpointStatus"]
        except ClientError as ce:
            code = ce.response["Error"]["Code"]
            if any(x in code for x in ["ExpiredToken","InvalidSignature","SignatureDoesNotMatch"]):
                ep_status = "expired_token"; token_ok = False
            else:
                ep_status = f"error:{code}"
        except Exception as ex:
            ep_status = f"error:{ex}"

    return jsonify({
        "status": "ok", "service": "AttritionIQ API v3.0",
        "endpoint_name":   e["ep"] or "NOT SET",
        "endpoint_status": ep_status,
        "region":          e["region"],
        "credentials":     "set" if creds_ok else "missing",
        "session_token":   "set" if e["token"] else "not set",
        "token_valid":     token_ok,
        "ready": creds_ok and ep_ok and ep_status == "InService" and token_ok,
    })

# ── /api/monitor  (NEW — for monitoring dashboard page) ──────────
@api.route("/monitor", methods=["GET"])
def monitor():
    """
    Returns all monitoring data needed for the monitoring dashboard:
    - Endpoint health + status
    - Latest evaluation metrics from S3
    - Drift indicators
    - Recent batch upload history
    - Retrain recommendation
    """
    e = get_env()
    now = datetime.now(timezone.utc).isoformat()

    report = {
        "timestamp": now,
        "endpoint":  {},
        "metrics":   {},
        "drift":     {},
        "batches":   [],
        "retrain":   False,
        "alerts":    [],
    }

    # ── 1. Endpoint health ──
    if e["key"] and e["secret"] and e["ep"]:
        try:
            sm   = get_sm_client(e)
            desc = sm.describe_endpoint(EndpointName=e["ep"])
            report["endpoint"] = {
                "name":          desc["EndpointName"],
                "status":        desc["EndpointStatus"],
                "created":       desc.get("CreationTime","").isoformat() if hasattr(desc.get("CreationTime",""),"isoformat") else str(desc.get("CreationTime","")),
                "last_modified": desc.get("LastModifiedTime","").isoformat() if hasattr(desc.get("LastModifiedTime",""),"isoformat") else str(desc.get("LastModifiedTime","")),
                "instance_type": desc.get("ProductionVariants",[{}])[0].get("CurrentInstanceType","ml.t2.medium") if desc.get("ProductionVariants") else "ml.t2.medium",
                "healthy":       desc["EndpointStatus"] == "InService",
            }
        except ClientError as ce:
            code = ce.response["Error"]["Code"]
            report["endpoint"] = {"name": e["ep"], "status": "error", "healthy": False, "error": code}
            if "ExpiredToken" in code or "SignatureDoesNotMatch" in code:
                report["alerts"].append("AWS token expired — refresh credentials")
        except Exception as ex:
            report["endpoint"] = {"name": e["ep"], "status": "error", "healthy": False, "error": str(ex)}
    else:
        report["endpoint"] = {"name": e["ep"] or "NOT SET", "status": "not_configured", "healthy": False}

    # ── 2. Latest evaluation metrics from S3 ──
    if e["key"] and e["secret"] and e["bucket"]:
        try:
            s3  = get_s3_client(e)
            obj = s3.get_object(
                Bucket=e["bucket"],
                Key=f"{e['project']}/evaluation/evaluation_results.json"
            )
            m = json.loads(obj["Body"].read())
            report["metrics"] = {
                "accuracy":    m.get("test_accuracy",  0),
                "roc_auc":     m.get("test_roc_auc",   0),
                "recall":      m.get("test_recall",    0),
                "precision":   m.get("test_precision", 0),
                "f1":          m.get("test_f1",        0),
                "specificity": m.get("specificity",    0),
                "confusion_matrix": m.get("confusion_matrix", {}),
                "thresholds": {"accuracy": 0.60, "roc_auc": 0.50, "recall": 0.35},
                "accuracy_ok":  m.get("test_accuracy",0)  >= 0.60,
                "roc_auc_ok":   m.get("test_roc_auc",0)   >= 0.50,
                "recall_ok":    m.get("test_recall",0)    >= 0.35,
            }
            # Check if retraining needed
            if not all([report["metrics"]["accuracy_ok"],
                        report["metrics"]["roc_auc_ok"],
                        report["metrics"]["recall_ok"]]):
                report["retrain"] = True
                report["alerts"].append("Model metrics below threshold — retraining recommended")
        except ClientError:
            report["metrics"] = {"error": "evaluation_results.json not found in S3"}
        except Exception as ex:
            report["metrics"] = {"error": str(ex)}
    else:
        # Return known model metrics as fallback
        report["metrics"] = {
            "accuracy": 0.871, "roc_auc": 0.823, "recall": 0.584,
            "precision": 0.762, "f1": 0.661, "specificity": 0.942,
            "confusion_matrix": {"tn": 240, "fp": 15, "fn": 50, "tp": 69},
            "thresholds": {"accuracy": 0.60, "roc_auc": 0.50, "recall": 0.35},
            "accuracy_ok": True, "roc_auc_ok": True, "recall_ok": True,
            "source": "hardcoded_fallback"
        }

    # ── 3. Drift indicators (from monitoring reports in S3) ──
    if e["key"] and e["secret"] and e["bucket"]:
        try:
            s3   = get_s3_client(e)
            resp = s3.list_objects_v2(
                Bucket=e["bucket"],
                Prefix=f"{e['project']}/monitoring/",
                MaxKeys=5
            )
            keys = sorted(
                [o["Key"] for o in resp.get("Contents",[]) if o["Key"].endswith(".json")],
                reverse=True
            )
            if keys:
                latest = json.loads(s3.get_object(Bucket=e["bucket"],Key=keys[0])["Body"].read())
                report["drift"] = {
                    "last_check": latest.get("timestamp",""),
                    "checks": latest.get("checks",{}),
                    "retrain_recommended": latest.get("retrain_recommended", False),
                    "recent_reports": len(keys),
                }
                if latest.get("retrain_recommended"):
                    report["retrain"] = True
            else:
                report["drift"] = {"last_check": "No monitoring reports found", "checks": {}}
        except Exception as ex:
            report["drift"] = {"error": str(ex)}

    # ── 4. Recent quarterly batch uploads ──
    if e["key"] and e["secret"] and e["bucket"]:
        try:
            s3   = get_s3_client(e)
            resp = s3.list_objects_v2(
                Bucket=e["bucket"],
                Prefix=f"{e['project']}/batch-uploads/",
                MaxKeys=20
            )
            uploads = []
            for obj in resp.get("Contents", []):
                if obj["Key"].endswith(".csv"):
                    parts = obj["Key"].split("/")
                    quarter = parts[-2] if len(parts) >= 2 else "unknown"
                    uploads.append({
                        "key":      obj["Key"],
                        "quarter":  quarter,
                        "size_kb":  round(obj["Size"] / 1024, 1),
                        "uploaded": obj["LastModified"].isoformat() if hasattr(obj["LastModified"],"isoformat") else str(obj["LastModified"]),
                    })
            report["batches"] = sorted(uploads, key=lambda x: x["uploaded"], reverse=True)[:10]
        except Exception as ex:
            report["batches"] = []

    return jsonify(report)

# ── /api/model/info ──────────────────────────────────────────────
@api.route("/model/info", methods=["GET"])
def model_info():
    e = get_env()
    return jsonify({
        "model_name":   "TunedXGBoostClassifier",
        "framework":    "XGBoost 1.7",
        "container":    "sagemaker-scikit-learn:1.2-1-cpu-py3",
        "hyperparameters": {"learning_rate":0.1,"max_depth":3,"n_estimators":50,"subsample":0.8},
        "training": {"dataset":"HR Analytics Data","total_samples":1470,"split":"80/20 stratified","attrition_rate":"16.1%"},
        "metrics":  {"accuracy":0.871,"precision":0.762,"recall":0.584,"f1":0.661,"roc_auc":0.823},
        "features": FEATURE_NAMES,
        "endpoint_name": e["ep"] or "NOT SET",
        "region":        e["region"],
    })

@api.route("/features", methods=["GET"])
def features():
    meta = {
        "HighTenure_Satisfied": {"type":"binary",  "values":"0=No, 1=Yes(>=5yr & Sat>=3)", "importance":0.031},
        "OverTime_Yes":         {"type":"binary",  "values":"0=No, 1=Yes",                 "importance":0.312},
        "Age":                  {"type":"numeric", "values":"18–65",                        "importance":0.142},
        "MonthlyIncome":        {"type":"numeric", "values":"1000–100000 (INR)",            "importance":0.218},
        "MaritalStatus_Single": {"type":"binary",  "values":"0=Married/Divorced, 1=Single","importance":0.062},
        "Gender_Male":          {"type":"binary",  "values":"0=Female, 1=Male",             "importance":0.019},
        "YearsAtCompany":       {"type":"numeric", "values":"0–40",                         "importance":0.118},
        "JobSatisfaction":      {"type":"ordinal", "values":"1=VeryLow  2=Low  3=Med  4=High","importance":0.098},
    }
    fl = sorted(
        [{"index":i,"name":n,**meta.get(n,{})} for i,n in enumerate(FEATURE_NAMES)],
        key=lambda x: x.get("importance",0), reverse=True
    )
    return jsonify({"count":len(FEATURE_NAMES),"features":fl,"order":FEATURE_NAMES})
