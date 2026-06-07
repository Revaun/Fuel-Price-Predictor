import pandas as pd
from sklearn.datasets import dump_svmlight_file

# Load your CSV
df = pd.read_csv("fuel_prices_clean.csv")

# Choose your target column (example: petrol_usd_liter)
target = df["petrol_usd_liter"]

# Drop the target from features
features = df.drop(columns=["petrol_usd_liter"])

# Convert to LIBSVM format
dump_svmlight_file(features, target, "fuel_prices_libsvm.txt", zero_based=True)

print("Saved LIBSVM file: fuel_prices_libsvm.txt")
