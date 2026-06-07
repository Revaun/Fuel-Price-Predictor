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

📁 Project Structure
- data/              # Raw and cleaned datasets
- output/            # Model artifacts and scripts
- scripts/           # Training, preprocessing, deployment scripts
- docs/              # Proof snapshots and notebooks
- fuel_app.py        # Streamlit app
- requirements.txt   # Dependencies
- README.md          # Project overview

🧩 Issues Faced & Resolutions
- SDK ImportError → Upgraded SageMaker SDK; used local inference as fallback
- Free Tier Credits Exhausted → Shut down endpoints quickly; switched to local predictions
- Model Artifact Handling → Extracted model.tar.gz, loaded with xgb.Booster()
- Evaluation & Visualization → Computed RMSE (~1.5 USD/Liter), plotted scatter with diagonal line
- AWS Resource Cleanup → Wrote PowerShell script to terminate endpoints and delete unused volumes

📸 Screenshots
(Replace placeholders with fresh snapshots once deployed)
- Banner landing page
- Forecast chart
- Scatter plot with RMSE/MAE

✨ Author
Built by Revaun — showcasing ML engineering, cloud ops, and app deployment skills.

🔧 Notes
- Banner uses a custom futuristic truck design.
- After deployment, update screenshots in the “📸 Screenshots” section.

📌 Summary
This project demonstrates ML engineering + DevOps awareness:
- Training and deployment on SageMaker
- Cost management and cleanup scripts
- Interactive app delivery with Streamlit
