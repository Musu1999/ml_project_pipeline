"""
predict.py - Test HR Attrition SageMaker endpoint (XGBoost native container).
XGBoost built-in container accepts CSV input.

Feature order (8 features):
  HighTenure_Satisfied, OverTime_Yes, Age, MonthlyIncome,
  MaritalStatus_Single, Gender_Male, YearsAtCompany, JobSatisfaction

Usage:
  python src/inference/predict.py \
    --endpoint-name mousumi-hr-attrition-endpoint \
    --region ap-south-1 \
    --features "[0, 1, 28, 3200, 1, 1, 2, 2]"
"""
import argparse
import json
import boto3

CLASS_NAMES     = ["No Attrition", "Attrition"]
DEFAULT_FEATURES = [0, 1, 28, 3200, 1, 1, 2, 2]


def predict(endpoint_name, region, features):
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    # XGBoost built-in container expects CSV
    payload  = ",".join(str(f) for f in features)
    response = runtime.invoke_endpoint(
        EndpointName = endpoint_name,
        ContentType  = "text/csv",
        Body         = payload.encode(),
    )
    raw = response["Body"].read().decode().strip()
    logger_print(f"Raw response: {raw}")

    # XGBoost binary returns probability of class 1
    prob_attrition    = float(raw)
    prob_no_attrition = 1.0 - prob_attrition
    prediction        = 1 if prob_attrition >= 0.5 else 0

    return {
        "prediction":            prediction,
        "predicted_class":       CLASS_NAMES[prediction],
        "attrition_probability": round(prob_attrition, 5),
        "probabilities": {
            "No Attrition": round(prob_no_attrition, 5),
            "Attrition":    round(prob_attrition, 5),
        }
    }


def logger_print(msg):
    print(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--region",        default="ap-south-1")
    parser.add_argument("--features",      default=None)
    args = parser.parse_args()

    features = json.loads(args.features) if args.features else DEFAULT_FEATURES
    features = [float(f) for f in features]

    print("\n=== HR Attrition Prediction ===")
    print(f"Endpoint : {args.endpoint_name}")
    print(f"Features : {features}")

    result = predict(args.endpoint_name, args.region, features)
    print(json.dumps(result, indent=2))

    prob = result["attrition_probability"]
    print(f"\n→ {result['predicted_class']} | {'HIGH RISK' if prob>=0.5 else 'LOW RISK'} | {prob:.1%}")


if __name__ == "__main__":
    main()
