import boto3
import json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
endpoint_name = "petrol-price-endpoint-v4"

payload = {
    "country": "South Africa",
    "region": "Western Cape",
    "income_level": "High",
    "subsidy_level": "Low",
    "diesel_usd_liter": 1.20,
    "lpg_usd_liter": 0.80,
    "brent_crude_usd": 75.0,
    "tax_percentage": 0.15
}

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload)
)

print("✅ Prediction result:", response["Body"].read().decode("utf-8"))
