import os
import json
import pandas as pd
import xgboost as xgb

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "xgboost-model.bst")
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster

def input_fn(request_body, content_type="text/csv"):
    if content_type == "text/csv":
        values = request_body.strip().split(",")

        # Columns must match training order
        columns = [
            "country", "region", "income_level", "subsidy_level",
            "feature1", "feature2"  # replace with your actual numeric features
        ]
        df = pd.DataFrame([values], columns=columns)

        # Apply same encoding as training
        categorical_cols = ["country", "region", "income_level", "subsidy_level"]
        df = pd.get_dummies(df, columns=categorical_cols)

        # Ensure all expected dummy columns exist
        expected_cols = [
            # fill in the full list of dummy columns from training
            # e.g. "country_United States", "region_North America", ...
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training
        df = df[expected_cols]

        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    dmatrix = xgb.DMatrix(input_data)
    prediction = model.predict(dmatrix)
    return prediction.tolist()

def output_fn(prediction, accept="application/json"):
    if accept == "application/json":
        return json.dumps({"prediction": prediction})
    else:
        return str(prediction)
