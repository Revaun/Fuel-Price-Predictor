from sagemaker.image_uris import retrieve
from sagemaker.model import Model

# Execution role ARN
role = "arn:aws:iam::266088941084:role/service-role/AmazonSageMaker-ExecutionRole-20260501T182812"

# XGBoost container image for us-east-1
xgb_image = retrieve("xgboost", region="us-east-1", version="1.2-1")

# Register the model
model = Model(
    image_uri=xgb_image,
    model_data="s3://sagemaker-us-east-1-266088941084/model.tar.gz",
    role=role
)

# Deploy endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="petrol-price-endpoint"
)

# Test prediction (example input)
result = predictor.predict([[1.8, 2026, 5]])
print("Prediction:", result)

