import pandas as pd
import xgboost as xgb
import boto3
import os
import sagemaker

# --- Configuration ---
bucket = "sagemaker-us-east-1-266088941084"   # your S3 bucket
prefix = "fuel-price-demo/output"             # folder for outputs
s3_input = f"s3://{bucket}/fuel_prices_clean.csv"

session = sagemaker.Session()

# --- Load dataset from S3 ---
print("Downloading dataset from S3...")
s3 = boto3.client("s3")
local_csv = "fuel_prices_clean.csv"
s3.download_file(bucket, "fuel_prices_clean.csv", local_csv)

df = pd.read_csv(local_csv)

# --- Preprocess categorical features ---
print("Preprocessing dataset...")
categorical_cols = ["country", "region", "income_level", "subsidy_level"]
df = pd.get_dummies(df, columns=categorical_cols)

# Separate features and target (predict petrol price)
X = df.drop("petrol_usd_liter", axis=1)
y = df["petrol_usd_liter"]

# Convert any remaining object/string columns to numeric codes
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category').cat.codes

# Drop date column if present
if 'date' in X.columns:
    X = X.drop('date', axis=1)

# --- Train XGBoost model (Scikit-Learn API) ---
print("Training XGBoost model...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    eval_metric="rmse"
)

model.fit(X, y)

# --- Save booster in SageMaker-compatible format ---
bst_local_path = "xgboost-model.json"
print(f"Saving model locally to {bst_local_path}...")
model.get_booster().save_model(bst_local_path)

# Upload to S3
print("Uploading model to S3...")
session.upload_data(bst_local_path, bucket=bucket, key_prefix=prefix)
print("✅ Training complete. Model uploaded to S3.")
