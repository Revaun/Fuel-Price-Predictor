import pandas as pd
import boto3
import os

# --- Config ---
RAW_FILE = "../data/fuel_prices_raw.csv"
CLEAN_FILE = "../data/fuel_prices_clean.csv"
S3_BUCKET = "petrol-price-data-demo"   # replace with your bucket name
S3_KEY = "fuel_prices_clean.csv"

def clean_data():
    # Load raw CSV
    df = pd.read_csv(RAW_FILE)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop rows with missing values
    df = df.dropna()

    # Optional: keep only relevant columns (date + price)
    if "date" in df.columns and "south_africa" in df["country"].unique():
        df = df[df["country"] == "South Africa"]
        # Example: keep date + first numeric column as target
        df = df[["date", df.columns[5]]]
        df.rename(columns={df.columns[1]: "petrol_price"}, inplace=True)

    # Save cleaned CSV locally
    df.to_csv(CLEAN_FILE, index=False)
    print(f"Cleaned data saved to {CLEAN_FILE}")

    return CLEAN_FILE

def upload_to_s3(file_path):
    s3 = boto3.client("s3")
    s3.upload_file(file_path, S3_BUCKET, S3_KEY)
    print(f"Uploaded {file_path} to s3://{S3_BUCKET}/{S3_KEY}")

if __name__ == "__main__":
    cleaned_file = clean_data()
    upload_to_s3(cleaned_file)
