"""
predict.py - Test the HR Attrition SageMaker endpoint.

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

CLASS_NAMES = ["No Attrition", "Attrition"]
DEFAULT_FEATURES = [0, 1, 28, 3200, 1, 1, 2, 2]   # high-risk profile


def predict(endpoint_name, region, features):
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    payload  = json.dumps({"features": features})
    response = runtime.invoke_endpoint(
        EndpointName = endpoint_name,
        ContentType  = "application/json",
        Body         = payload.encode(),
    )
    result = json.loads(response["Body"].read().decode())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--region",        default="ap-south-1")
    parser.add_argument("--features",      default=None,
        help="JSON list of 8 features: [HighTenure_Satisfied, OverTime_Yes, Age, "
             "MonthlyIncome, MaritalStatus_Single, Gender_Male, YearsAtCompany, JobSatisfaction]")
    args = parser.parse_args()

    features = json.loads(args.features) if args.features else DEFAULT_FEATURES
    features = [float(f) for f in features]

    print("\n=== HR Attrition Prediction ===")
    print(f"Endpoint : {args.endpoint_name}")
    print(f"Region   : {args.region}")
    print(f"Features : {features}")

    result = predict(args.endpoint_name, args.region, features)

    print(f"\nResult:")
    print(json.dumps(result, indent=2))

    prob = result.get("attrition_probability", 0)
    cls  = result.get("predicted_class", "Unknown")
    risk = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    print(f"\n→ {cls} | {risk} | Probability: {prob:.1%}")


if __name__ == "__main__":
    main()
