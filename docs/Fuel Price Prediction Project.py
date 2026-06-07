# Title: Fuel Price Prediction Project

# Set environment variables for sagemaker_studio imports

import os
os.environ['DataZoneProjectId'] = '3hyk3729i7fr1j'
os.environ['DataZoneDomainId'] = 'dzd-3juu8npjx0gxd3'
os.environ['DataZoneEnvironmentId'] = 'at096q53fw8i3r'
os.environ['DataZoneDomainRegion'] = 'us-east-1'

# create both a function and variable for metadata access
_resource_metadata = None

def _get_resource_metadata():
    global _resource_metadata
    if _resource_metadata is None:
        _resource_metadata = {
            "AdditionalMetadata": {
                "DataZoneProjectId": "3hyk3729i7fr1j",
                "DataZoneDomainId": "dzd-3juu8npjx0gxd3",
                "DataZoneEnvironmentId": "at096q53fw8i3r",
                "DataZoneDomainRegion": "us-east-1",
            }
        }
    return _resource_metadata
metadata = _get_resource_metadata()

"""
Logging Configuration

Purpose:
--------
This sets up the logging framework for code executed in the user namespace.
"""

from typing import Optional


def _set_logging(log_dir: str, log_file: str, log_name: Optional[str] = None):
    import os
    import logging
    from logging.handlers import RotatingFileHandler

    level = logging.INFO
    max_bytes = 5 * 1024 * 1024
    backup_count = 5

    # fallback to /tmp dir on access, helpful for local dev setup
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = "/tmp/kernels/"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger() if not log_name else logging.getLogger(log_name)
    logger.handlers = []
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Rotating file handler
    fh = RotatingFileHandler(filename=log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging initialized for {log_name}.")


_set_logging("/var/log/computeEnvironments/kernel/", "kernel.log")
_set_logging("/var/log/studio/data-notebook-kernel-server/", "metrics.log", "metrics")

import logging
from sagemaker_studio import ClientConfig, sqlutils, sparkutils, dataframeutils

logger = logging.getLogger(__name__)
logger.info("Initializing sparkutils")
spark = sparkutils.init()
logger.info("Finished initializing sparkutils")

def _reset_os_path():
    """
    Reset the process's working directory to handle mount timing issues.
    
    This function resolves a race condition where the Python process starts
    before the filesystem mount is complete, causing the process to reference
    old mount paths and inodes. By explicitly changing to the mounted directory
    (/home/sagemaker-user), we ensure the process uses the correct, up-to-date
    mount point.
    
    The function logs stat information (device ID and inode) before and after
    the directory change to verify that the working directory is properly
    updated to reference the new mount.
    
    Note:
        This is executed at module import time to ensure the fix is applied
        as early as possible in the kernel initialization process.
    """
    try:
        import os
        import logging

        logger = logging.getLogger(__name__)
        logger.info("---------Before------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)

        os.chdir("/home/sagemaker-user")

        logger.info("---------After------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)
    except Exception as e:
        logger.exception(f"Failed to reset working directory: {e}")

_reset_os_path()

# MARKDOWN CELL djik
# # 🚀 Petrol Price Prediction Project
# 
# This notebook demonstrates a complete machine learning workflow:
# - **Data preparation and cleaning**
# - **Exploratory Data Analysis (EDA)**
# - **Model training and evaluation**
# - **Insights and business impact**
# 
# **Goal:** Predict petrol prices using global fuel data and showcase technical + analytical skills.

import pandas as pd

df = pd.read_csv("fuel_prices_clean.csv")
print("✅ Data loaded successfully!")
df.head()

# MARKDOWN CELL drbg
# ## 🔍 Data Exploration
# 
# We examine the dataset structure, sample rows, and correlations.  
# Key features include **crude oil price**, **tax percentage**, and **region**.

df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    if col != "date":
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
df_encoded = df_encoded.drop("date", axis=1)

X = df_encoded.drop("petrol_usd_liter", axis=1)
y = df_encoded["petrol_usd_liter"]

# MARKDOWN CELL 58ik
# ## ⚙️ Model Training
# 
# We train three models:
# - **Linear Regression**
# - **Random Forest**
# - **Gradient Boosting**
# 
# Each model offers different strengths in capturing relationships.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

lr_pred, rf_pred, gb_pred = lr.predict(X_test), rf.predict(X_test), gb.predict(X_test)

# MARKDOWN CELL 4n1y
# ## 📊 Model Evaluation
# 
# We compare model performance using:
# - **Mean Squared Error (MSE)**
# - **R² Score**
# 
# This highlights which model best predicts petrol prices.

metrics = {
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
    "MSE": [
        mean_squared_error(y_test, lr_pred),
        mean_squared_error(y_test, rf_pred),
        mean_squared_error(y_test, gb_pred)
    ],
    "R2 Score": [
        r2_score(y_test, lr_pred),
        r2_score(y_test, rf_pred),
        r2_score(y_test, gb_pred)
    ]
}
comparison_df = pd.DataFrame(metrics)
print("📊 Model Performance Comparison")
comparison_df

# MARKDOWN CELL chjn
# ## 🎯 Predicted vs Actual Comparison
# 
# To visually assess model performance, we plot predicted petrol prices against actual values.
# 
# - Points close to the diagonal line indicate accurate predictions.  
# - This chart highlights how well each model fits the data.

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

# Scatter plots for each model
plt.scatter(y_test, lr_pred, alpha=0.6, label="Linear Regression")
plt.scatter(y_test, rf_pred, alpha=0.6, label="Random Forest")
plt.scatter(y_test, gb_pred, alpha=0.6, label="Gradient Boosting")

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'k--', lw=2, label="Perfect Fit")

plt.xlabel("Actual Petrol Price (USD/Liter)")
plt.ylabel("Predicted Petrol Price (USD/Liter)")
plt.title("Predicted vs Actual Petrol Prices")
plt.legend()
plt.show()

# MARKDOWN CELL b6yq
# ## 🌟 Random Forest Feature Importance
# 
# To interpret the model, we examine which features most strongly influence petrol price predictions.  
# - **High importance** → feature has strong predictive power.  
# - **Low importance** → feature contributes less to the model.  
# 
# This helps connect the machine learning results to real-world drivers such as **crude oil price** and **tax percentage**.

import matplotlib.pyplot as plt

importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.show()

# MARKDOWN CELL d4t0
# ## 💡 Insights
# 
# Feature importance shows **crude oil price** and **tax percentage** as key drivers.  
# This provides interpretability and connects predictions to real-world factors.

# MARKDOWN CELL c4va
# # ✅ Conclusion
# 
# **Random Forest** and **Gradient Boosting** outperform Linear Regression.  
# R² scores ≈ **0.999**, showing near-perfect predictive power.
# 
# ### Business Impact
# - Logistics companies can forecast costs and optimize fleet operations.  
# - Policymakers can plan subsidies and taxation strategies more effectively.  
# - Consumers and businesses can anticipate price changes and budget accordingly.  
# 
# This notebook demonstrates a **full data science workflow** — recruiter‑ready and industry‑relevant.