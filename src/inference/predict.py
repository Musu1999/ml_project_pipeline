"""
predict.py
Test the deployed HR Analytics SageMaker endpoint.

Feature order:
    [HighTenure_Satisfied, OverTime_Yes, Age, MonthlyIncome,
     MaritalStatus_Single, Gender_Male, YearsAtCompany, JobSatisfaction]

Usage:
    python src/inference/predict.py \
        --endpoint-name yourname-hr-attrition-endpoint \
        --region ap-south-1
"""

import argparse
import json
import boto3

# Default test sample — young, single, overtime employee (likely attrition)
DEFAULT_FEATURES = {
    "HighTenure_Satisfied": 0,
    "OverTime_Yes":         1,
    "Age":                  28,
    "MonthlyIncome":        3200,
    "MaritalStatus_Single": 1,
    "Gender_Male":          1,
    "YearsAtCompany":       2,
    "JobSatisfaction":      2,
}


def predict(endpoint_name: str, region: str, features: dict) -> dict:
    runtime  = boto3.client("sagemaker-runtime", region_name=region)
    payload  = json.dumps({"features": list(features.values())})
    response = runtime.invoke_endpoint(
        EndpointName = endpoint_name,
        ContentType  = "application/json",
        Body         = payload.encode(),
    )
    return json.loads(response["Body"].read().decode())


def main():
    parser = argparse.ArgumentParser(description="HR Attrition Prediction")
    parser.add_argument("--endpoint-name", required=True,
                        help="SageMaker endpoint name")
    parser.add_argument("--region", default="ap-south-1",
                        help="AWS region")
    parser.add_argument("--features", default=None,
                        help="JSON list: [HighTenure_Satisfied, OverTime_Yes, "
                             "Age, MonthlyIncome, MaritalStatus_Single, "
                             "Gender_Male, YearsAtCompany, JobSatisfaction]")
    args = parser.parse_args()

    if args.features:
        vals     = json.loads(args.features)
        features = dict(zip(DEFAULT_FEATURES.keys(), vals))
    else:
        features = DEFAULT_FEATURES

    print("\n=== HR Attrition Prediction ===")
    print("Input features:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    result = predict(args.endpoint_name, args.region, features)

    print("\nPrediction Result:")
    print(json.dumps(result, indent=2))

    prob  = result.get("attrition_probability", 0)
    cls   = result.get("predicted_class", "?")
    risk  = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    print(f"\n→ {cls}  |  {risk}  |  Attrition probability: {prob:.1%}")
    print("================================\n")


if __name__ == "__main__":
    main()
