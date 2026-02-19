"""
predict.py - Test the HR Analytics endpoint (XGBoost native container).
XGBoost native container accepts CSV input and returns raw probabilities.

Feature order:
  [HighTenure_Satisfied, OverTime_Yes, Age, MonthlyIncome,
   MaritalStatus_Single, Gender_Male, YearsAtCompany, JobSatisfaction]

Usage:
  python src/inference/predict.py --endpoint-name mousumi-hr-attrition-endpoint
"""

import argparse
import json
import boto3

CLASS_NAMES = ["No Attrition", "Attrition"]

# Default: young, overtime, single employee = likely attrition
DEFAULT = [0, 1, 28, 3200, 1, 1, 2, 2]


def predict(endpoint_name, region, features):
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    # XGBoost native container expects CSV
    payload  = ",".join(str(f) for f in features)
    response = runtime.invoke_endpoint(
        EndpointName = endpoint_name,
        ContentType  = "text/csv",
        Body         = payload.encode(),
    )
    raw = response["Body"].read().decode().strip()
    print(f"\nRaw response: {raw}")

    # XGBoost binary classifier returns single probability of class 1
    prob_attrition = float(raw)
    prob_no_attrition = 1.0 - prob_attrition
    prediction = 1 if prob_attrition >= 0.5 else 0

    result = {
        "prediction":            prediction,
        "predicted_class":       CLASS_NAMES[prediction],
        "attrition_probability": round(prob_attrition, 5),
        "probabilities": {
            "No Attrition": round(prob_no_attrition, 5),
            "Attrition":    round(prob_attrition, 5),
        }
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--region",        default="ap-south-1")
    parser.add_argument("--features",      default=None,
                        help="Comma-separated: HighTenure_Satisfied,OverTime_Yes,"
                             "Age,MonthlyIncome,MaritalStatus_Single,"
                             "Gender_Male,YearsAtCompany,JobSatisfaction")
    args = parser.parse_args()

    features = [float(x) for x in args.features.split(",")] \
               if args.features else DEFAULT

    print("\n=== HR Attrition Prediction ===")
    print(f"Features: {features}")

    result = predict(args.endpoint_name, args.region, features)
    print(f"\nResult:")
    print(json.dumps(result, indent=2))
    prob = result["attrition_probability"]
    print(f"\n→ {result['predicted_class']} | "
          f"{'HIGH RISK' if prob >= 0.5 else 'LOW RISK'} | "
          f"Probability: {prob:.1%}")


if __name__ == "__main__":
    main()
