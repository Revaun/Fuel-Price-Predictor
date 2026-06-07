import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import altair as alt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io
from datetime import datetime

# --- Banner at top (no overlay title, just image) ---
st.markdown(
    """
    <style>
    .banner {
        background-image: url('https://copilot.microsoft.com/th/id/BCO.67d13ae6-884f-4317-ab37-9fcf36b31366.png'); /* Truck on the Road at Sunset */
        background-size: cover;
        height: 250px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
    <div class="banner"></div>
    """,
    unsafe_allow_html=True
)

# Load model
model = xgb.Booster()
model.load_model("output/xgboost-model")

# Load dataset
data = pd.read_csv("data/fuel_prices_raw.csv", header=None)
data.columns = [
    "date", "country", "region", "income_level", "subsidy_level",
    "petrol_usd_liter", "diesel_usd_liter", "lpg_usd_liter",
    "brent_crude_usd", "tax_percentage"
]

# Sidebar input
st.sidebar.title("⛽ Fuel Price Predictor")
country = st.sidebar.selectbox(
    "🌍 Select a country",
    data["country"].unique(),
    index=None,
    placeholder="Choose a country..."
)

fuel_type = st.sidebar.selectbox("⛽ Select fuel type", ["petrol_usd_liter", "diesel_usd_liter", "lpg_usd_liter"])
weeks_ahead = st.sidebar.slider("🔮 Weeks to forecast", min_value=1, max_value=12, value=4)

if country:
    sample = data[data["country"] == country].iloc[-1]

    features = [
        "country", "region", "income_level", "subsidy_level",
        "diesel_usd_liter", "lpg_usd_liter", "brent_crude_usd", "tax_percentage"
    ]

    sample_df = pd.DataFrame([sample[features]])
    sample_encoded = pd.get_dummies(sample_df, columns=["country", "region", "income_level", "subsidy_level"])
    for col in ["diesel_usd_liter", "lpg_usd_liter", "brent_crude_usd", "tax_percentage"]:
        if col in sample_encoded.columns:
            sample_encoded[col] = pd.to_numeric(sample_encoded[col], errors="coerce")
    sample_encoded = sample_encoded.reindex(columns=model.feature_names, fill_value=0)

    dtest = xgb.DMatrix(sample_encoded)
    pred = np.clip(model.predict(dtest), 0, None)
    actual_latest = pd.to_numeric(sample[fuel_type], errors="coerce")

    st.success(f"💡 Predicted {fuel_type.replace('_usd_liter','')} price in **{country}**: **{pred[0]:.2f} USD/Liter**")
    st.info(f"📊 Actual latest {fuel_type.replace('_usd_liter','')} price in {country}: {actual_latest:.2f} USD/Liter")

    # Historical data
    history = data[data["country"] == country][["date", fuel_type]].copy()
    history["date"] = pd.to_datetime(history["date"])
    history.rename(columns={fuel_type: "Actual"}, inplace=True)

    # Forecast data (start from today, not dataset end)
    today = datetime.today()
    future_dates = pd.date_range(start=today, periods=weeks_ahead+1, freq="W")[1:]
    future_preds = []
    crude_price = float(sample["brent_crude_usd"])
    for d in future_dates:
        crude_price *= 1.005
        future_features = sample_encoded.copy()
        future_features["brent_crude_usd"] = crude_price
        dtest_future = xgb.DMatrix(future_features)
        p = model.predict(dtest_future)
        future_preds.append(float(np.clip(p[0], 0, None)))
    forecast_df = pd.DataFrame({"date": future_dates, "Forecast": future_preds})

    history_long = history.melt(id_vars="date", value_vars=["Actual"], var_name="Type", value_name="Price")
    forecast_long = forecast_df.melt(id_vars="date", value_vars=["Forecast"], var_name="Type", value_name="Price")
    combined_long = pd.concat([history_long, forecast_long])

    min_date = min(history["date"].min(), forecast_df["date"].min())
    max_date = max(history["date"].max(), forecast_df["date"].max())
    date_range = st.slider("📅 Select date range", min_value=min_date.to_pydatetime(),
                           max_value=max_date.to_pydatetime(),
                           value=(min_date.to_pydatetime(), max_date.to_pydatetime()))

    history_filtered = history_long[(history_long["date"] >= pd.to_datetime(date_range[0])) &
                                    (history_long["date"] <= pd.to_datetime(date_range[1]))]
    forecast_filtered = combined_long[(combined_long["date"] >= pd.to_datetime(date_range[0])) &
                                      (combined_long["date"] <= pd.to_datetime(date_range[1]))]

    # Multi-fuel forecast
    multi_forecast = []
    crude_price = float(sample["brent_crude_usd"])
    for d in future_dates:
        crude_price *= 1.005
        future_features = sample_encoded.copy()
        future_features["brent_crude_usd"] = crude_price
        dtest_future = xgb.DMatrix(future_features)
        p = model.predict(dtest_future)
        base = float(np.clip(p[0], 0, None))
        multi_forecast.append({"date": d, "Petrol": base, "Diesel": base*1.05, "LPG": base*0.95})
    multi_forecast_long = pd.DataFrame(multi_forecast).melt(id_vars="date", value_vars=["Petrol","Diesel","LPG"],
                                                            var_name="Fuel", value_name="Price")

    fuel_colors = alt.Scale(domain=["Petrol","Diesel","LPG"], range=["#1f77b4","#ff7f0e","#2ca02c"])

    col_hist, col_forecast = st.columns(2)

    with col_hist:
        st.subheader(f"Historical {fuel_type.replace('_usd_liter','').capitalize()} Prices")
        hist_chart = alt.Chart(history_filtered).mark_line().encode(
            x="date:T", y="Price:Q", color="Type:N", tooltip=["date","Type","Price"]
        ).properties(width=350, height=300).interactive()
        st.altair_chart(hist_chart, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("⬇️ Download Historical Data (CSV)",
                           data=history_filtered.to_csv(index=False).encode("utf-8"),
                           file_name=f"{country}_{fuel_type}_historical.csv", mime="text/csv")

    with col_forecast:
        current_month = today.strftime("%B %Y")
        st.subheader(f"Forecasted {fuel_type.replace('_usd_liter','').capitalize()} Prices ({current_month}, Next {weeks_ahead} Weeks)")
        forecast_chart = alt.Chart(forecast_filtered).mark_line().encode(
            x="date:T", y="Price:Q", color="Type:N", tooltip=["date","Type","Price"]
        ).properties(width=350, height=300).interactive()
        st.altair_chart(forecast_chart, use_container_width=True)
        st.download_button("⬇️ Download Forecast Data (CSV)",
                           data=forecast_filtered.to_csv(index=False).encode("utf-8"),
                           file_name=f"{country}_{fuel_type}_forecast.csv", mime="text/csv")

    st.subheader(f"Forecasted Multi-Fuel Prices ({today.strftime('%B %Y')}, Next {weeks_ahead} Weeks)")
    multi_chart = alt.Chart(multi_forecast_long).mark_line().encode(
        x="date:T", y="Price:Q",
        color=alt.Color("Fuel:N", scale=fuel_colors, legend=alt.Legend(orient="top")),
        tooltip=["date","Fuel","Price"]
    ).properties(width=700, height=400).interactive()
    st.altair_chart(multi_chart, use_container_width=True)
    st.download_button("⬇️ Download Multi-Fuel Forecast (CSV)",
                       data=multi_forecast_long.to_csv(index=False).encode("utf-8"),
                       file_name=f"{country}_multi_fuel_forecast.csv", mime="text/csv")

    # Combined export
    combined_export = io.StringIO()
    combined_df = pd.concat([
        history_filtered.assign(Source="History"),
        forecast_filtered.assign(Source="Forecast"),
        multi_forecast_long.assign(Source="Multi-Fuel")
    ])
    combined_df.to_csv(combined_export, index=False)

    st.download_button("⬇️ Download All Data (CSV)",
                       data=combined_export.getvalue().encode("utf-8"),
                       file_name=f"{country}_all_data.csv",
                       mime="text/csv")

    # Error metrics
    validation = data[data["country"] == country].tail(20)
    val_df = validation[features].copy()
    val_encoded = pd.get_dummies(val_df, columns=["country", "region", "income_level", "subsidy_level"])
    for col in ["diesel_usd_liter", "lpg_usd_liter", "brent_crude_usd", "tax_percentage"]:
        if col in val_encoded.columns:
            val_encoded[col] = pd.to_numeric(val_encoded[col], errors="coerce")
    val_encoded = val_encoded.reindex(columns=model.feature_names, fill_value=0)
    dval = xgb.DMatrix(val_encoded)
    val_preds = model.predict(dval)
    val_actuals = pd.to_numeric(validation[fuel_type], errors="coerce")

    rmse = np.sqrt(mean_squared_error(val_actuals, val_preds))
    mae = mean_absolute_error(val_actuals, val_preds)

    st.subheader("📊 Model Accuracy (last 20 records)")
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.3f}")
    col2.metric("MAE", f"{mae:.3f}")

else:
    st.info("👈 Please select a country from the sidebar to see predictions.")

# --- Author signature (footer) ---
st.markdown(
    """
    <hr style="margin-top:40px; margin-bottom:10px;">
    <p style="text-align:center; font-size:14px; color:gray;">
    🚀 Built by <b>Revaun</b>
    </p>
    """,
    unsafe_allow_html=True
)

# --- Author signature (sidebar) ---
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Author:** Revaun")
