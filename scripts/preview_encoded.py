import pandas as pd

df = pd.read_csv("C:/Users/Revaun/Projects/sagemaker-petrol-price-demo/data/fuel_prices_clean.csv")

# Apply the same preprocessing as train.py
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

if 'date' in df.columns:
    df = df.drop('date', axis=1)

print(df.head())
