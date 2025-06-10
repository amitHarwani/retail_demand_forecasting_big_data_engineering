# predict_daily.py
import pandas as pd
import joblib
import os
import boto3
from botocore.client import Config
from datetime import date, timedelta
import argparse
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, dayofweek, month, year, weekofyear, lag, avg, expr, lit
from pyspark.sql.window import Window
from sqlalchemy import create_engine

# MinIO configuration
MINIO_ENDPOINT = os.environ.get('MINIO_ENDPOINT', "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', "minioadmin")
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', "minioadmin")

S3_CLIENT = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

# PostgreSQL connection (for Superset)
PG_CONN_STR = os.environ.get('PG_CONN_STR', "postgresql://airflow:airflow@postgres:5432/airflow")
PG_ENGINE = create_engine(PG_CONN_STR)

def download_from_minio(bucket_name, object_name, local_path):
    """Downloads a file from MinIO."""
    try:
        S3_CLIENT.download_file(bucket_name, object_name, local_path)
        print(f"Successfully downloaded {bucket_name}/{object_name} to {local_path}")
    except Exception as e:
        print(f"Error downloading {bucket_name}/{object_name}: {e}")
        raise

def upload_to_minio(file_path, bucket_name, object_name):
    """Uploads a file to MinIO."""
    try:
        S3_CLIENT.upload_file(file_path, bucket_name, object_name)
        print(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")
        raise

def create_spark_session_for_prediction():
    """Creates and configures a SparkSession for MinIO integration."""
    spark = SparkSession.builder \
        .appName("DailyPredictionSpark") \
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT) \
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY) \
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1,com.amazonaws:aws-java-sdk-bundle:1.11.901") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def generate_daily_forecasts(forecast_start_date_str, prediction_horizon_days=7):
    """
    Generates forecasts for a range of dates across all active products and stores
    using the trained model and latest refined data.
    """
    forecast_start_date = date.fromisoformat(forecast_start_date_str)
    
    print(f"Generating forecasts for {prediction_horizon_days} days starting from {forecast_start_date_str}...")
    
    # 1. Load the model and feature columns
    local_model_path = "./temp_forecast_model.joblib"
    local_features_path = "./temp_model_features.json"
    
    download_from_minio("curated", "models/forecast_model.joblib", local_model_path)
    download_from_minio("curated", "models/model_features.json", local_features_path)
    
    model = joblib.load(local_model_path)
    with open(local_features_path, "r") as f:
        training_feature_columns = json.load(f)
    print("Model and features loaded.")

    # 2. Prepare Spark session and load necessary data
    spark = create_spark_session_for_prediction()
    
    try:
        # Load static catalog and stores to get all combinations for prediction
        catalog_df = spark.read.csv(f"s3a://raw/static/catalog.csv", header=True, inferSchema=True)
        stores_df = spark.read.csv(f"s3a://raw/static/stores.csv", header=True, inferSchema=True)
        
        # Load future markdown/discount/price data from static files
        # These represent *planned* promotions/prices for future dates
        markdowns_static_df = spark.read.csv(f"s3a://raw/static/markdowns.csv", header=True, inferSchema=True)
        discounts_static_df = spark.read.csv(f"s3a://raw/static/discounts_history.csv", header=True, inferSchema=True)
        price_history_static_df = spark.read.csv(f"s3a://raw/static/price_history.csv", header=True, inferSchema=True)

        markdowns_static_df = markdowns_static_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
        discounts_static_df = discounts_static_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
        price_history_static_df = price_history_static_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))


        # Determine the date range for loading historical refined data for lag/rolling calculations
        # Need up to (forecast_start_date - 1) day, and max 365 days back from there
        historical_look_back_end_date = forecast_start_date - timedelta(days=1)
        historical_look_back_start_date = historical_look_back_end_date - timedelta(days=365) # Max lag

        # Construct paths for the required historical refined data
        historical_paths = []
        current_date_hist = historical_look_back_start_date
        while current_date_hist <= historical_look_back_end_date:
            historical_paths.append(f"s3a://refined/forecasting_features/year={current_date_hist.year}/month={current_date_hist.month:02d}/day={current_date_hist.day:02d}/")
            current_date_hist += timedelta(days=1)
        
        historical_refined_df_spark = None
        try:
            # Read historical refined data only for the relevant look-back window
            print(f"Loading historical refined data for lag calculations (from {historical_look_back_start_date} to {historical_look_back_end_date})...")
            historical_refined_df_spark = spark.read.parquet(*historical_paths)
            historical_refined_df_spark = historical_refined_df_spark.filter(col("date") >= historical_look_back_start_date).filter(col("date") <= historical_look_back_end_date)
        except Exception as e:
            print(f"Error reading historical refined data for lags. This might be expected if paths don't exist yet: {e}")
            # Define schema for empty DataFrame based on expected refined data columns (excluding total_sales_amount)
            # This is a heuristic, better to get schema dynamically
            from pyspark.sql.types import StructType, StructField, DateType, StringType, DoubleType, IntegerType
            historical_schema = StructType([
                StructField("date", DateType(), True),
                StructField("item_id", IntegerType(), True),
                StructField("store_id", IntegerType(), True),
                StructField("total_quantity", DoubleType(), True), # Target variable needs to be null for future, but in history it's actual
                StructField("avg_base_price", DoubleType(), True),
                StructField("markdown_quantity", DoubleType(), True),
                StructField("markdown_price", DoubleType(), True),
                StructField("promo_price", DoubleType(), True),
                StructField("on_promotion", IntegerType(), True),
                StructField("day_of_week", IntegerType(), True),
                StructField("month", IntegerType(), True),
                StructField("year", IntegerType(), True),
                StructField("week_of_year", IntegerType(), True),
                StructField("is_weekend", IntegerType(), True),
                StructField("lag_7_quantity", DoubleType(), True),
                StructField("lag_30_quantity", DoubleType(), True),
                StructField("lag_90_quantity", DoubleType(), True),
                StructField("lag_365_quantity", DoubleType(), True),
                StructField("rolling_mean_7_quantity", DoubleType(), True),
                StructField("rolling_mean_30_quantity", DoubleType(), True),
                StructField("dept_name", StringType(), True),
                StructField("class_name", StringType(), True),
                StructField("subclass_name", StringType(), True),
                StructField("item_type", StringType(), True),
                StructField("division", StringType(), True),
                StructField("format", StringType(), True),
                StructField("city", StringType(), True),
                StructField("area", StringType(), True)
            ])
            historical_refined_df_spark = spark.createDataFrame([], schema=historical_schema) # Empty DataFrame

        # Create a DataFrame for all item-store combinations for the entire prediction horizon (next 7 days)
        all_forecast_dates = [forecast_start_date + timedelta(days=i) for i in range(prediction_horizon_days)]
        
        all_items_stores_for_prediction = catalog_df.select("item_id").distinct() \
                                     .crossJoin(stores_df.select("store_id").distinct())
                                     
        # Create a combined DataFrame for all future dates and all item/store combinations
        future_prediction_df_spark = None
        for future_date in all_forecast_dates:
            dated_items_stores = all_items_stores_for_prediction.withColumn("date", lit(future_date.strftime('%Y-%m-%d')).cast(DateType()))
            if future_prediction_df_spark is None:
                future_prediction_df_spark = dated_items_stores
            else:
                future_prediction_df_spark = future_prediction_df_spark.union(dated_items_stores)
        
        # Join future prediction rows with planned pricing/promo data
        future_prediction_df_spark = future_prediction_df_spark \
            .join(markdowns_static_df, on=["date", "item_id", "store_id"], how="left") \
            .join(discounts_static_df, on=["date", "item_id", "store_id"], how="left") \
            .join(price_history_static_df, on=["date", "item_id", "store_id"], how="left") # Use price_history for avg_base_price if available
        
        # Prepare future_prediction_df_spark to union with historical_refined_df_spark
        # Add placeholder columns with null values to match the schema for union
        # Ensure 'total_quantity' is None for future prediction rows
        for col_name in historical_refined_df_spark.columns:
            if col_name not in future_prediction_df_spark.columns:
                future_prediction_df_spark = future_prediction_df_spark.withColumn(col_name, expr("null").cast(historical_refined_df_spark.schema[col_name].dataType))
        
        # Explicitly set total_quantity to None for future prediction rows
        future_prediction_df_spark = future_prediction_df_spark.withColumn("total_quantity", lit(None).cast(DoubleType()))

        # Union historical data with the new prediction rows for accurate feature engineering
        # Use allowMissingColumns=True for robustness, though schema should now match
        combined_df_spark = historical_refined_df_spark.unionByName(future_prediction_df_spark, allowMissingColumns=True)

    except Exception as e:
        if spark: spark.stop()
        print(f"Error loading inference data or creating base DataFrame: {e}")
        raise

    if combined_df_spark.count() == 0:
        print(f"No data to generate predictions from for date {forecast_start_date_str}. Exiting.")
        spark.stop()
        return

    # 3. Re-generate features for the inference data (including necessary lags/rolling means)
    # This is done on the combined historical + future dates dataset.
    window_spec_item_store = Window.partitionBy("item_id", "store_id").orderBy("date")
    
    combined_df_spark = combined_df_spark.withColumn("day_of_week", dayofweek(col("date"))) \
                                         .withColumn("month", month(col("date"))) \
                                         .withColumn("year", year(col("date"))) \
                                         .withColumn("week_of_year", weekofyear(col("date")))
    combined_df_spark = combined_df_spark.withColumn("is_weekend", ((col("day_of_week") == 1) | (col("day_of_week") == 7)).cast("integer"))

    # Lagged sales features & Rolling averages
    combined_df_spark = combined_df_spark.withColumn("lag_7_quantity", lag(col("total_quantity"), 7).over(window_spec_item_store))
    combined_df_spark = combined_df_spark.withColumn("lag_30_quantity", lag(col("total_quantity"), 30).over(window_spec_item_store))
    combined_df_spark = combined_df_spark.withColumn("lag_90_quantity", lag(col("total_quantity"), 90).over(window_spec_item_store))
    combined_df_spark = combined_df_spark.withColumn("lag_365_quantity", lag(col("total_quantity"), 365).over(window_spec_item_store))

    combined_df_spark = combined_df_spark.withColumn("rolling_mean_7_quantity", avg(col("total_quantity")).over(window_spec_item_store.rowsBetween(-6, 0)))
    combined_df_spark = combined_df_spark.withColumn("rolling_mean_30_quantity", avg(col("total_quantity")).over(window_spec_item_store.rowsBetween(-29, 0)))
    
    # Promotional indicator
    combined_df_spark = combined_df_spark.withColumn("on_promotion", (col("promo_type_code").isNotNull() | (col("markdown_quantity") > 0)).cast("integer"))

    # Fill NA for numeric features for prediction rows
    numeric_cols_to_fill = [
        "lag_7_quantity", "lag_30_quantity", "lag_90_quantity", "lag_365_quantity",
        "rolling_mean_7_quantity", "rolling_mean_30_quantity",
        "markdown_quantity", "markdown_price", "promo_price", "avg_base_price" # These should be filled by planned data if available, otherwise 0
    ]
    combined_df_spark = combined_df_spark.na.fill(0, subset=numeric_cols_to_fill) # Fill any remaining NaNs
    
    categorical_cols_to_fill = ['dept_name', 'class_name', 'subclass_name', 'item_type',
                                'division', 'format', 'city', 'area', 'promo_type_code']
    for col_name in categorical_cols_to_fill:
        if col_name in combined_df_spark.columns:
            combined_df_spark = combined_df_spark.na.fill("unknown", subset=[col_name])

    # Filter for the specific forecast dates and convert to Pandas for inference
    # Select only the rows for the future prediction horizon
    inference_df_spark = combined_df_spark.filter(col("date") >= forecast_start_date_str) \
                                          .filter(col("date") <= (forecast_start_date + timedelta(days=prediction_horizon_days-1)).strftime('%Y-%m-%d'))
    
    inference_pandas_df = inference_df_spark.toPandas()
    spark.stop()

    if inference_pandas_df.empty:
        print(f"No data for inference for dates from {forecast_start_date_str} to {forecast_start_date + timedelta(days=prediction_horizon_days-1)}. Exiting.")
        return

    # 4. Align features for prediction (One-hot encoding and column alignment)
    categorical_features_train = ["dept_name", "class_name", "subclass_name", "item_type",
                                  "division", "format", "city", "area"]
    
    for col_name in categorical_features_train:
        if col_name in inference_pandas_df.columns:
            inference_pandas_df[col_name] = inference_pandas_df[col_name].astype(str)

    inference_pandas_df = pd.get_dummies(inference_pandas_df, columns=categorical_features_train, dummy_na=False)
    
    # Align columns to what the model was trained on
    missing_cols = set(training_feature_columns) - set(inference_pandas_df.columns)
    for c in missing_cols:
        inference_pandas_df[c] = 0
    
    extra_cols = set(inference_pandas_df.columns) - set(training_feature_columns)
    inference_pandas_df = inference_pandas_df.drop(columns=list(extra_cols))
    
    # Ensure the order of columns is the same as training
    X_inference = inference_pandas_df[training_feature_columns]

    print("Generating forecasts...")
    forecasts = model.predict(X_inference)
    inference_pandas_df['predicted_quantity'] = forecasts.round().astype(int)
    inference_pandas_df['predicted_quantity'] = inference_pandas_df['predicted_quantity'].apply(lambda x: max(0, x))

    # Select relevant columns for forecasts (date, item_id, store_id, predicted_quantity)
    forecast_results_df = inference_pandas_df[['date', 'item_id', 'store_id', 'predicted_quantity']]

    # 5. Write forecasts to MinIO curated zone and PostgreSQL
    forecast_output_dir = f"forecasts/year={forecast_start_date.year}/month={forecast_start_date.month:02d}/day={forecast_start_date.day:02d}"
    local_forecast_path = f"./forecasts_temp_{forecast_start_date_str}.parquet" # Temporary local file
    os.makedirs("./forecasts", exist_ok=True) # Ensure local dir exists

    forecast_results_df.to_parquet(local_forecast_path, index=False)
    
    upload_to_minio(local_forecast_path, "curated", f"{forecast_output_dir}/data.parquet")
    print(f"Forecasts saved to {local_forecast_path} and uploaded to MinIO curated/{forecast_output_dir}/data.parquet")
    os.remove(local_forecast_path) # Clean up temp file

    # Write to PostgreSQL for Superset
    table_name = "daily_product_store_forecasts"
    forecast_results_df.to_sql(table_name, PG_ENGINE, if_exists='append', index=False, method='multi')
    print(f"Forecasts also saved to PostgreSQL table: {table_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate daily forecasts.")
    parser.add_argument("--forecast_date", type=str, help="Date for which to start generating forecasts (YYYY-MM-DD).")
    args = parser.parse_args()

    if not args.forecast_date:
        print("Please provide a forecast start date using --forecast_date (YYYY-MM-DD).")
        exit(1)
        
    os.makedirs("./models", exist_ok=True) # Ensure local model dir for download
    os.makedirs("./forecasts", exist_ok=True) # Ensure local forecast dir for output

    generate_daily_forecasts(args.forecast_date)
