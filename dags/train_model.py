# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import joblib
import os
import boto3
from botocore.client import Config
from datetime import date, timedelta
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lit
import argparse

# MinIO configuration
MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

S3_CLIENT = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

def upload_to_minio(file_path, bucket_name, object_name):
    """Uploads a file to MinIO."""
    try:
        S3_CLIENT.upload_file(file_path, bucket_name, object_name)
        print(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")
        raise

def create_spark_session_for_training():
    """Creates and configures a SparkSession for MinIO integration."""
    spark = SparkSession.builder \
        .appName("ModelTrainingDataLoader") \
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT) \
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY) \
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
        .config("spark.hadoop.ipc.client.connection.maxidletime",    "60000") \
        .config("spark.hadoop.fs.s3a.connection.timeout",            "60000") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout",  "60000") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.4.1") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def train_forecasting_model(training_end_date_str, transformed_data_path="s3a://transformed/forecasting_features/", model_output_dir="./models"):
    """
    Loads feature-engineered data up to training_end_date, trains a forecasting model,
    and saves the model and its feature columns.
    """
    training_end_date = date.fromisoformat(training_end_date_str)
    print(f"Loading data from {transformed_data_path} for training up to {training_end_date_str}...", flush=True)
    
    spark = create_spark_session_for_training()
    try:
        # Read partitioned Parquet data for the training period up to training_end_date
        df_spark = spark.read.parquet(transformed_data_path)
        df_spark = df_spark.filter(col("date") <= to_date(lit(training_end_date_str), "yyyy-MM-dd")) 
        
        # Convert to Pandas DataFrame for Scikit-learn
        pandas_df = df_spark.toPandas()
        pandas_df['date'] = pd.to_datetime(pandas_df['date']).dt.date
        spark.stop()
    except Exception as e:
        if spark: spark.stop()
        print(f"Error loading data for training: {e}", flush=True)
        raise

    if pandas_df.empty:
        print(f"No data available for training up to {training_end_date_str}. Exiting.", flush=True)
        return

    print(f"Loaded {len(pandas_df)} rows for training.", flush=True)

    # Prepare data for modeling - total_sales_amount is removed as a feature
    features = [
        "item_id", "store_id", "avg_base_price", "markdown_quantity", "markdown_price", "markdown_normal_price",
        "promo_price", "discount_price_before_promo",
        "on_promotion", "day_of_week", "month", "year", "week_of_year", "is_weekend", "day",
        "lag_7_quantity", "lag_30_quantity", "lag_90_quantity", "lag_365_quantity",
        "rolling_mean_7_quantity", "rolling_mean_30_quantity"
    ]
    categorical_features = ["dept_name", "class_name", "subclass_name", "item_type",
                            "division", "format", "city", "area"]

    # One-hot encode categorical features
    pandas_df = pd.get_dummies(pandas_df, columns=categorical_features, dummy_na=False)

    # Filter features that exist in the dataframe after one-hot encoding
    final_features_list = [col for col in features if col in pandas_df.columns]
    final_features_list.extend([col for col in pandas_df.columns if any(col.startswith(f"{cat}_") for cat in categorical_features)])

    X = pandas_df[final_features_list]
    y = pandas_df['total_quantity']

    # Handle potential NaNs introduced by lag features at the start of the time series
    X = X.fillna(0) # Simple fill for PoC
    y = y.fillna(0)

    # Time-based split for validation (e.g., last 15 days of the training data)
    val_split_date = training_end_date - timedelta(days=14) 
    
    X_train = X[pandas_df['date'] <= val_split_date]
    y_train = y[pandas_df['date'] <= val_split_date]
    X_val = X[pandas_df['date'] > val_split_date]
    y_val = y[pandas_df['date'] > val_split_date]

    print(f"Training data size: {len(X_train)}", flush=True)
    print(f"Validation data size: {len(X_val)}", flush=True)

    if len(X_train) == 0:
        print("Not enough training data. Exiting model training.", flush=True)
        return

    # Train a simple RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    print("Training model...", flush=True)
    model.fit(X_train, y_train)
    print("Model training completed.", flush=True)

    # Evaluate model
    if not X_val.empty:
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        print(f"Validation RMSE: {rmse:.2f}", flush=True)
        print(f"Validation MAPE: {mape:.2f}", flush=True)
    else:
        print("No validation data available for the split.", flush=True)

    # Save the model
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, "forecast_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}", flush=True)

    # Save the feature columns used for training
    features_file_path = os.path.join(model_output_dir, "model_features.json")
    with open(features_file_path, "w") as f:
        json.dump(X_train.columns.tolist(), f)
    print(f"Model features saved to {features_file_path}", flush=True)

    # Upload model and features to MinIO
    upload_to_minio(model_path, "forecasts", "models/forecast_model.joblib")
    upload_to_minio(features_file_path, "forecasts", "models/model_features.json")
    print("Model and features uploaded to MinIO 'forecasts' bucket.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Forecasting Model.")
    parser.add_argument("--training_end_date", type=str, help="End date for training data (YYYY-MM-DD).")
    args = parser.parse_args()

    if not args.training_end_date:
        print("Please provide a training end date using --training_end_date (YYYY-MM-DD).")
        exit(1)

    os.makedirs("./models", exist_ok=True)
    train_forecasting_model(args.training_end_date)
