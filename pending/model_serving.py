# model_serving.py (using Flask)
import pandas as pd
import joblib
import os
import boto3
from botocore.client import Config
from flask import Flask, request, jsonify
import json
from datetime import datetime

app = Flask(__name__)

# MinIO configuration
MINIO_ENDPOINT = os.environ.get('MINIO_ENDPOINT', "http://minio:9000") # Use 'minio' as hostname in Docker network
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', "minioadmin")
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', "minioadmin")

S3_CLIENT = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

MODEL = None
TRAINING_FEATURE_COLUMNS = None
STATIC_METADATA = {} # For item/store features

def download_from_minio(bucket_name, object_name, local_path):
    """Downloads a file from MinIO."""
    try:
        S3_CLIENT.download_file(bucket_name, object_name, local_path)
        print(f"Successfully downloaded {bucket_name}/{object_name} to {local_path}")
    except Exception as e:
        print(f"Error downloading {bucket_name}/{object_name}: {e}")
        raise

def load_model_and_features():
    """Loads the trained model and feature columns."""
    global MODEL, TRAINING_FEATURE_COLUMNS, STATIC_METADATA
    
    model_path = "./models/forecast_model.joblib"
    features_path = "./models/model_features.json"
    
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./data", exist_ok=True) # Ensure local data dir for static metadata
    
    try:
        download_from_minio("curated", "models/forecast_model.joblib", model_path)
        MODEL = joblib.load(model_path)
        print("Model loaded.")

        download_from_minio("curated", "models/model_features.json", features_path)
        with open(features_path, "r") as f:
            TRAINING_FEATURE_COLUMNS = json.load(f)
        print("Training feature columns loaded.")

        # Load static catalog and store data for feature generation from 'static' prefix
        download_from_minio("raw", "static/catalog.csv", "./data/catalog.csv")
        download_from_minio("raw", "static/stores.csv", "./data/stores.csv")
        STATIC_METADATA['catalog'] = pd.read_csv("./data/catalog.csv")
        STATIC_METADATA['stores'] = pd.read_csv("./data/stores.csv")
        
        # Load markdowns and discounts for basic promo checks (not full history)
        download_from_minio("raw", "static/markdowns.csv", "./data/markdowns.csv")
        download_from_minio("raw", "static/discounts_history.csv", "./data/discounts_history.csv")
        STATIC_METADATA['markdowns'] = pd.read_csv("./data/markdowns.csv", parse_dates=['date'])
        STATIC_METADATA['discounts'] = pd.read_csv("./data/discounts_history.csv", parse_dates=['date'])
        STATIC_METADATA['price_history'] = pd.read_csv("./data/price_history.csv", parse_dates=['date']) # For future price

        print("Static metadata loaded.")

    except Exception as e:
        print(f"Failed to load model or features: {e}")
        MODEL = None
        TRAINING_FEATURE_COLUMNS = None
        STATIC_METADATA = {}

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None or TRAINING_FEATURE_COLUMNS is None:
        return jsonify({'error': 'Model not loaded. Please try again later.'}), 503

    try:
        data = request.get_json(force=True)
        
        # Required inputs for ad-hoc prediction
        item_id = data.get('item_id')
        store_id = data.get('store_id')
        prediction_date_str = data.get('date') # YYYY-MM-DD
        
        # Optional inputs (pricing/promotion related) - if not provided, treated as 0 or null
        # These are expected to be *future planned* values
        markdown_quantity = data.get('markdown_quantity', 0)
        markdown_price = data.get('markdown_price', 0)
        promo_type_code = data.get('promo_type_code') # e.g., 'Discount'
        promo_price = data.get('promo_price', 0)
        avg_base_price = data.get('avg_base_price', 0) # Base price for the day

        if not all([item_id, store_id, prediction_date_str]):
            return jsonify({'error': 'Missing required fields: item_id, store_id, date'}), 400

        prediction_date = pd.to_datetime(prediction_date_str)

        # Build input DataFrame for prediction
        input_df = pd.DataFrame([{
            'date': prediction_date,
            'item_id': item_id,
            'store_id': store_id,
            'markdown_quantity': markdown_quantity,
            'markdown_price': markdown_price,
            'promo_type_code': promo_type_code,
            'promo_price': promo_price,
            'avg_base_price': avg_base_price,
            # Lagged and rolling features are NOT dynamically computed here for ad-hoc prediction.
            # They will be filled with 0s for this PoC as their computation requires historical sales data context.
            'lag_7_quantity': 0, 'lag_30_quantity': 0, 'lag_90_quantity': 0, 'lag_365_quantity': 0,
            'rolling_mean_7_quantity': 0, 'rolling_mean_30_quantity': 0
        }])

        # Feature Engineering for ad-hoc prediction (static joins & time features only)
        # Join with catalog and stores data
        input_df = input_df.merge(STATIC_METADATA['catalog'], on='item_id', how='left')
        input_df = input_df.merge(STATIC_METADATA['stores'], on='store_id', how='left')

        # Time-based features
        input_df['day_of_week'] = input_df['date'].dt.dayofweek + 1 # Monday=1, Sunday=7 for Spark
        input_df['month'] = input_df['date'].dt.month
        input_df['year'] = input_df['date'].dt.year
        input_df['week_of_year'] = input_df['date'].dt.isocalendar().week.astype(int)
        input_df['is_weekend'] = ((input_df['day_of_week'] == 1) | (input_df['day_of_week'] == 7)).astype(int)

        # Promotional indicator
        input_df['on_promotion'] = ((input_df['promo_type_code'].notna()) | (input_df['markdown_quantity'] > 0)).astype(int)

        # Fill NaNs for features (especially for new items/stores or missing metadata)
        numeric_cols_to_fill = [
            "avg_base_price", "markdown_quantity", "markdown_price", "promo_price",
            "lag_7_quantity", "lag_30_quantity", "lag_90_quantity", "lag_365_quantity",
            "rolling_mean_7_quantity", "rolling_mean_30_quantity"
        ]
        input_df[numeric_cols_to_fill] = input_df[numeric_cols_to_fill].fillna(0)

        categorical_features = ["dept_name", "class_name", "subclass_name", "item_type",
                                "division", "format", "city", "area"]
        for col_name in categorical_features:
            if col_name in input_df.columns:
                input_df[col_name] = input_df[col_name].fillna("unknown").astype(str)

        # One-hot encode categorical features
        input_df = pd.get_dummies(input_df, columns=categorical_features, dummy_na=False)

        # IMPORTANT: Align columns with training data
        missing_cols = set(TRAINING_FEATURE_COLUMNS) - set(input_df.columns)
        for c in missing_cols:
            input_df[c] = 0
        
        extra_cols = set(input_df.columns) - set(TRAINING_FEATURE_COLUMNS)
        input_df = input_df.drop(columns=list(extra_cols))
        
        # Ensure the order of columns is the same as training
        X_inference = input_df[TRAINING_FEATURE_COLUMNS]

        # Predict
        forecast = MODEL.predict(X_inference)
        predicted_quantity = max(0, int(forecast[0].round())) # Ensure non-negative integer

        return jsonify({
            'item_id': item_id,
            'store_id': store_id,
            'date': prediction_date_str,
            'predicted_quantity': predicted_quantity,
            'note': 'Lagged and rolling features were set to 0 for this ad-hoc prediction, which may impact accuracy. For accurate predictions, use the batch forecasting pipeline.'
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model_and_features() # Load model on startup
    app.run(host='0.0.0.0', port=5000)
