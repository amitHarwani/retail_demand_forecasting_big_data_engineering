# predict_daily.py
import pandas as pd
import os
import boto3
from botocore.client import Config
from datetime import date, timedelta
import argparse
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, dayofweek, month, year, weekofyear, lag, avg, expr, lit, dayofmonth
from pyspark.sql.window import Window
from sqlalchemy import create_engine
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, DateType, StringType, DoubleType, IntegerType

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

# PostgreSQL connection (for Superset)
PG_CONN_STR = os.environ.get('PG_CONN_STR', "postgresql://airflow:airflow@postgres:5432/airflow")
PG_ENGINE = create_engine(PG_CONN_STR)

def download_from_minio(bucket_name, object_name, local_path):
    """Downloads a file or directory (recursively) from MinIO."""
    try:
        # Check if object_name is a directory prefix
        if object_name.endswith('/'):
            paginator = S3_CLIENT.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=object_name)
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        file_key = obj["Key"]
                        # Construct local file path
                        local_file_path = os.path.join(local_path, os.path.relpath(file_key, object_name))
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        S3_CLIENT.download_file(bucket_name, file_key, local_file_path)
                        print(f"Downloaded {file_key} to {local_file_path}")
        else:
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
        .config("spark.hadoop.ipc.client.connection.maxidletime",    "60000") \
        .config("spark.hadoop.fs.s3a.connection.timeout",            "60000") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout",  "60000") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.4.1") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def generate_daily_forecasts(forecast_start_date_str, prediction_horizon_days=7):
    """
    Generates forecasts for a range of dates across all active products and stores
    using the trained model and latest refined data.
    """
    spark = create_spark_session_for_prediction()

    forecast_start_date = date.fromisoformat(forecast_start_date_str)
    
    print(f"Generating forecasts for {prediction_horizon_days} days starting from {forecast_start_date_str}...")
    
    # 1. Load the Spark ML PipelineModel
    local_model_path = "./models/spark_ml_pipeline_model"
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True) # Ensure local dir exists
    
    # Need to download the entire directory structure for the Spark ML model
    download_from_minio("forecasts", "models/spark_ml_pipeline_model/", local_model_path)
    
    model = PipelineModel.load(local_model_path) # Load the Spark ML PipelineModel
    print("Spark ML PipelineModel loaded.")

    try:
        # Load static catalog and stores to get all combinations for prediction
        catalog_df = spark.read.csv(f"s3a://raw/static/catalog.csv", header=True, inferSchema=True)
        stores_df = spark.read.csv(f"s3a://raw/static/stores.csv", header=True, inferSchema=True)
        
        # Load future markdown/discount/price data from static files
        markdowns_static_df = spark.read.csv(f"s3a://raw/static/markdowns.csv", header=True, inferSchema=True)
        discounts_static_df = spark.read.csv(f"s3a://raw/static/discounts_history.csv", header=True, inferSchema=True)
        price_history_static_df = spark.read.csv(f"s3a://raw/static/price_history.csv", header=True, inferSchema=True)

        markdowns_static_df = markdowns_static_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
        discounts_static_df = discounts_static_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
        price_history_static_df = price_history_static_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))


        # Determine the date range for loading historical refined data for lag/rolling calculations
        historical_look_back_end_date = forecast_start_date - timedelta(days=1)
        historical_look_back_start_date = historical_look_back_end_date - timedelta(days=365)

        historical_paths = []
        current_date_hist = historical_look_back_start_date
        while current_date_hist <= historical_look_back_end_date:
            year_path = current_date_hist.year
            month_path = f"{current_date_hist.month:02d}"
            day_path = f"{current_date_hist.day:02d}"
            historical_paths.append(f"s3a://transformed/forecasting_features/year={year_path}/month={month_path}/day={day_path}/")
            current_date_hist += timedelta(days=1)
        
        historical_refined_df_spark = None
        try:
            print(f"Loading historical refined data for lag calculations (from {historical_look_back_start_date} to {historical_look_back_end_date})...")
            historical_refined_df_spark = spark.read.parquet(*historical_paths)
            historical_refined_df_spark = historical_refined_df_spark.filter(col("date") >= historical_look_back_start_date).filter(col("date") <= historical_look_back_end_date)
        except Exception as e:
            print(f"Error reading historical refined data for lags: {e}. Creating empty DataFrame.")
            # Define schema for empty DataFrame (excluding item_id and store_id as categorical features for the model)
            raise

        # Create a DataFrame for all item-store combinations for the entire prediction horizon (next 7 days)
        all_forecast_dates = [forecast_start_date + timedelta(days=i) for i in range(prediction_horizon_days)]
        
        # Future data - Forecast dates
        all_items_stores_for_prediction = catalog_df.select("item_id").distinct() \
                                     .crossJoin(stores_df.select("store_id").distinct())
                                     
        future_prediction_df_spark = None
        for future_date_obj in all_forecast_dates:
            future_date_str_lit = future_date_obj.strftime('%Y-%m-%d') # Converting to string
            dated_items_stores = all_items_stores_for_prediction.withColumn("date", lit(future_date_str_lit).cast(to_date(lit("2000-01-01"), "yyyy-MM-dd").dataType))
            if future_prediction_df_spark is None:
                future_prediction_df_spark = dated_items_stores
            else:
                future_prediction_df_spark = future_prediction_df_spark.union(dated_items_stores)
        
        # Join future prediction rows with planned pricing/promo data (from static files)
        future_prediction_df_spark = future_prediction_df_spark \
            .join(markdowns_static_df.select("date", "item_id", "store_id", "normal_price", "price"), on=["date", "item_id", "store_id"], how="left") \
            .withColumnRenamed("price", "markdown_price") \
            .withColumnRenamed("normal_price", "markdown_normal_price") \
            .join(discounts_static_df.select("date", "item_id", "store_id", "promo_type_code", "sale_price_time_promo", "sale_price_before_promo"), on=["date", "item_id", "store_id"], how="left") \
            .withColumnRenamed("sale_price_time_promo", "promo_price") \
            .withColumnRenamed("sale_price_before_promo", "discount_price_before_promo") \
            .join(price_history_static_df.select("date", "item_id", "store_id", "price_base"), on=["date", "item_id", "store_id"], how="left") \
            .withColumnRenamed("price_base", "avg_base_price")

        # Prepare future_prediction_df_spark to union with historical_refined_df_spark
        historical_schema = historical_refined_df_spark.schema
        
        all_items_stores_prep_df = future_prediction_df_spark 
        for field in historical_schema:
            if field.name not in all_items_stores_prep_df.columns:
                all_items_stores_prep_df = all_items_stores_prep_df.withColumn(field.name, lit(None).cast(field.dataType))
        
        all_items_stores_prep_df = all_items_stores_prep_df.withColumn("total_quantity", lit(None).cast(DoubleType()))

        # Union historical data with the new prediction rows for accurate feature engineering
        combined_df_spark = historical_refined_df_spark.unionByName(all_items_stores_prep_df, allowMissingColumns=True)

    except Exception as e:
        if spark: spark.stop()
        print(f"Error loading inference data or creating base DataFrame: {e}")
        raise

    if combined_df_spark.count() == 0:
        print(f"No data to generate predictions from for date {forecast_start_date_str}. Exiting.")
        spark.stop()
        return

    # 3. Re-generate features for the inference data (including necessary lags/rolling means)
    # This logic is mostly copied from transform_data.py to ensure consistency
    window_spec_item_store = Window.partitionBy("item_id", "store_id").orderBy("date")
    
    combined_df_spark = combined_df_spark.withColumn("day_of_week", dayofweek(col("date"))) \
                                         .withColumn("month", month(col("date"))) \
                                         .withColumn("year", year(col("date"))) \
                                         .withColumn("week_of_year", weekofyear(col("date"))) \
                                         .withColumn("day", dayofmonth(col("date")))
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
        "markdown_quantity", "markdown_price", "markdown_normal_price",
        "promo_price", "discount_price_before_promo",
        "avg_base_price"
    ]
    combined_df_spark = combined_df_spark.na.fill(0, subset=numeric_cols_to_fill)
    
    categorical_cols_to_fill = ['dept_name', 'class_name', 'subclass_name', 'item_type',
                                'division', 'format', 'city', 'area', 'promo_type_code'] 
    for col_name in categorical_cols_to_fill:
        if col_name in combined_df_spark.columns:
            combined_df_spark = combined_df_spark.na.fill("unknown", subset=[col_name])

    # Filter for the specific forecast dates for prediction
    inference_df_spark = combined_df_spark.filter(col("date") >= forecast_start_date_str) \
                                          .filter(col("date") <= (forecast_start_date + timedelta(days=prediction_horizon_days-1)).strftime('%Y-%m-%d'))
    
    if inference_df_spark.count() == 0:
        print(f"No data for inference for dates from {forecast_start_date_str} to {forecast_start_date + timedelta(days=prediction_horizon_days-1)}. Exiting.")
        spark.stop()
        return

    print("Generating forecasts using Spark ML PipelineModel...")
    # Apply the Spark ML PipelineModel for transformations and prediction
    predictions = model.transform(inference_df_spark)
    
    # Select relevant columns for forecasts (date, item_id, store_id, predicted_quantity)
    # The prediction column is named 'prediction' by Spark ML regressors by default
    forecast_results_df_spark = predictions.select(
        col("date"),
        col("item_id"),
        col("store_id"),
        col("prediction").cast(IntegerType()).alias("predicted_quantity") # Cast to int
    ).withColumn("predicted_quantity", expr("CASE WHEN predicted_quantity < 0 THEN 0 ELSE predicted_quantity END")) # Ensure non-negative

    # Convert to Pandas for writing to PostgreSQL (optional, but easier for direct to_sql)
    forecast_results_df = forecast_results_df_spark.toPandas()
    spark.stop()

    # 5. Write forecasts to MinIO curated zone and PostgreSQL
    forecast_output_dir = f"forecasts/year={forecast_start_date.year}/month={forecast_start_date.month:02d}/day={forecast_start_date.day:02d}"
    
    # Spark ML Model outputs are also Parquet. It's more natural to write directly using Spark.
    # We will write the forecast results directly as Parquet using Spark for consistency.
    spark_forecast_output_path = f"s3a://curated/{forecast_output_dir}/"
    print(f"Writing forecasts to MinIO curated/{forecast_output_dir}/")
    forecast_results_df_spark.write.mode("overwrite").parquet(spark_forecast_output_path) # Use Spark to write parquet
    print(f"Forecasts written to MinIO curated/{forecast_output_dir}/data.parquet")

    # Write to PostgreSQL for Superset (still using Pandas for this part)
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
        
    os.makedirs("./models", exist_ok=True) # Ensure local model dir for download (for PipelineModel)
    os.makedirs("./forecasts", exist_ok=True) # Ensure local forecast dir for any intermediate Pandas outputs

    generate_daily_forecasts(args.forecast_date)
