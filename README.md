# ⛽ Fuel Price Predictor 🚀

![Banner](assets/optimus-style-banner.png) 
*A professional dashboard for forecasting fuel prices, built with AWS SageMaker + Streamlit.*

---

## 📊 Features
- **Machine Learning Model**: Trained XGBoost on historical fuel price data  
- **Interactive Dashboard**:  
  - Forecast chart for future predictions  
  - Scatter plot with RMSE/MAE error metrics  
  - Downloadable predictions (`predictions.csv`)  
- **Professional UI**: Banner, collapsible sections, and author signature  

---

## 🛠 Tech Stack
- **AWS SageMaker** → training, model hosting, cost‑aware cloud ops  
- **Streamlit** → interactive app deployment  
- **Python** → pandas, numpy, scikit‑learn, xgboost, altair  

---

## 🚀 Live Demo
👉 [Fuel Price Predictor on Streamlit Cloud](https://fuel-price-predictor.streamlit.app)  
*(Link will be active after deployment)*

---

## 📸 Screenshots
*(Replace placeholders with your new snapshots once deployed)*

- Landing page with banner + signature  
- Forecast chart with predictions  
- Scatter plot with RMSE/MAE  

---
## ⚙️ Run Locally
Clone the repo and install dependencies:

bash
git clone https://github.com/Revaun/Fuel-Price-Predictor.git
cd Fuel-Price-Predictor
pip install -r requirements.txt
streamlit run fuel_app.py

📂 Project Structure 
data/              # Raw and cleaned datasets
output/            # Model artifacts and scripts
scripts/           # Training, preprocessing, deployment scripts
docs/              # Proof snapshots and notebooks
fuel_app.py        # Streamlit app
requirements.txt   # Dependencies
README.md          # Project overview



🛠 Issues Faced & Resolutions (SageMaker Project)
ImportError: Transformer not found in SageMaker SDK  
Issue: Tried to use Batch Transform, but the sagemaker package didn’t expose the Transformer class.
Resolution: Upgraded the SageMaker Python SDK. As a cost‑saving alternative, ran predictions locally in Jupyter using the exported XGBoost model artifact (xgboost_model.json).

Free Tier Credits Exhausted  
Issue: Fast instance types quickly consumed AWS free tier credits.
Resolution: Shut down endpoints and notebook instances immediately after use. Switched to local inference with XGBoost to avoid further charges.

Model Artifact Handling  
Issue: SageMaker training produced model.tar.gz containing a raw xgboost-model file. Needed to extract and load it manually.
Resolution: Extracted the artifact, loaded with xgb.Booster(), and used DMatrix for predictions locally.

Evaluation & Visualization  
Issue: Needed to validate predictions without SageMaker endpoints.
Resolution: Computed RMSE (~1.5 USD/Liter) against test data locally and visualized results with a scatter plot + diagonal “perfect prediction” line.

AWS Resource Cleanup  
Issue: Risk of incurring hidden charges from leftover endpoints, EC2 instances, or EBS volumes.
Resolution: Wrote and ran a PowerShell cleanup script to terminate all SageMaker endpoints, stop notebooks, kill EC2 instances, and delete unattached volumes.

📸 Screenshots
(Replace placeholders with fresh snapshots once deployed)

Banner landing page

Forecast chart

Scatter plot with RMSE/MAE

✨ Author
Built by Revaun — showcasing ML engineering, cloud ops, and app deployment skills.

🔧 Notes
The banner uses a royalty‑free Unsplash image (energy/city theme). You can swap the link for any other professional photo (fuel, transport, skyline, technology).

Once you deploy to Streamlit Cloud, capture fresh screenshots and drop them into the README under “📸 Screenshots.”

📌 Summary
This project demonstrates end‑to‑end ML engineering with AWS SageMaker and Streamlit, plus real DevOps problem‑solving (SDK issues, cost management, artifact handling, evaluation, and cleanup). It’s polished, recruiter‑ready, and visually appealing.