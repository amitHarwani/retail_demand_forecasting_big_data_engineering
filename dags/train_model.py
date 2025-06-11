# train_model.py
import os
import json
from datetime import date, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
import boto3
from botocore.client import Config

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

def upload_to_minio(local_path, bucket_name, object_name):
    """Uploads a file or directory (recursively) to MinIO."""
    try:
        # Check if it's a directory
        if os.path.isdir(local_path):
            for root, _, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, local_path)
                    minio_obj_name = os.path.join(object_name, rel_path)
                    S3_CLIENT.upload_file(file_path, bucket_name, minio_obj_name)
                    print(f"Uploaded {file_path} to {bucket_name}/{minio_obj_name}")
        else:
            S3_CLIENT.upload_file(local_path, bucket_name, object_name)
            print(f"Successfully uploaded {local_path} to {bucket_name}/{object_name}")
    except Exception as e:
        print(f"Error uploading {local_path}: {e}")
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
    spark = create_spark_session_for_training()

    training_end_date = date.fromisoformat(training_end_date_str)
    print(f"Loading data from {transformed_data_path} for training up to {training_end_date_str}...")
    
    try:
        # Read partitioned Parquet data for the training period up to training_end_date
        df_spark = spark.read.parquet(transformed_data_path)
        df_spark = df_spark.filter(col("date") <= to_date(lit(training_end_date_str), "yyyy-MM-dd")) 
        
        # Drop rows with NaNs in 'total_quantity' as it's the target
        df_spark = df_spark.na.drop(subset=['total_quantity'])

        if df_spark.count() == 0:
            print(f"No data available for training up to {training_end_date_str} after dropping NaNs. Exiting.")
            spark.stop()
            return

        print(f"Loaded rows for training after dropping NaNs in target.")

        # Define feature columns (excluding item_id and store_id from OHE per user request)
        numerical_features = [
            "store_id", "avg_base_price", "markdown_quantity", "markdown_price", "markdown_normal_price",
            "promo_price", "discount_price_before_promo",
            "on_promotion", "day_of_week", "month", "year", "week_of_year", "is_weekend", "day",
            "lag_7_quantity", "lag_30_quantity", "lag_90_quantity", "lag_365_quantity",
            "rolling_mean_7_quantity", "rolling_mean_30_quantity"
        ]
        
        # Categorical features that will be one-hot encoded
        categorical_features = ["item_id","dept_name", "class_name", "subclass_name", "item_type",
                                "division", "format", "city", "area"]

        # Spark ML Pipeline Stages
        pipeline_stages = []

        # StringIndexer and OneHotEncoder for categorical features
        encoded_features = []
        for feature in categorical_features:
            # Map each distinct category to a numeric index
            indexer = StringIndexer(inputCol=feature, outputCol=feature + "_indexed", handleInvalid="keep")
            # Turns the index into a sparse one-hot encoded vector
            encoder = OneHotEncoder(inputCol=feature + "_indexed", outputCol=feature + "_encoded")
            # Append the indexer and encoder to the pipeline
            pipeline_stages.extend([indexer, encoder])
            # Append the <feature>_encoded to encoded_features list
            encoded_features.append(feature + "_encoded")
        
        # All feature columns
        assembler_inputs = numerical_features + encoded_features
        # Assemble all features into a single vector
        vector_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")
        pipeline_stages.append(vector_assembler) # Appending to the pipeline

        # RandomForestRegressor model
        # Note: Spark ML RandomForestRegressor works on Vector type features
        rf = RandomForestRegressor(featuresCol="features", labelCol="total_quantity", numTrees=50, seed=42)
        pipeline_stages.append(rf)

        # Create the Pipeline
        pipeline = Pipeline(stages=pipeline_stages)

        print("Training Spark ML Pipeline...")
        # Split data for training and validation in Spark
        # Note: Time-based split in Spark is more involved, for PoC we'll do random split,
        # or rely on batch processing to generate data up to a certain date.
        # For a truly time-series validation, you'd need to filter data before fitting.
        # Here, the training_end_date parameter dictates the max data used.
        # If needed, a train-validation split can be added similar to Pandas, but on Spark DF.
        # For simplicity of this PoC, we train on all filtered data up to training_end_date.
        
        # For evaluation, we will predict on the same training data or a held-out validation set.
        # In a real scenario, you'd have a separate validation dataset
        
        # Let's perform a simple time-based split directly on the Spark DataFrame for evaluation
        # This will simulate holding out the last 14 days for validation within the loaded data
        val_split_date_spark = training_end_date - timedelta(days=14)
        
        training_data_spark = df_spark.filter(col("date") <= val_split_date_spark.strftime('%Y-%m-%d'))
        validation_data_spark = df_spark.filter(col("date") > val_split_date_spark.strftime('%Y-%m-%d'))

        if training_data_spark.count() == 0:
            print("Not enough training data after time-based split. Exiting model training.")
            spark.stop()
            return

        # Running the pipeline
        model = pipeline.fit(training_data_spark)
        print("Spark ML Pipeline training completed.")

        # Evaluate model if validation data exists
        if validation_data_spark.count() > 0:
            # applies the same index/encode/assemble pipeline to new data and then calls the RF’s .transform(), producing a "prediction" column.
            predictions = model.transform(validation_data_spark)
            
            # Spark ML provides evaluators
            from pyspark.ml.evaluation import RegressionEvaluator
            rmse_evaluator = RegressionEvaluator(labelCol="total_quantity", predictionCol="prediction", metricName="rmse")

            rmse = rmse_evaluator.evaluate(predictions)
            # MAPE often needs custom implementation or a compatible library. For simplicity, we'll skip MAPE metric if not directly supported by evaluator.
            # If using a newer Spark version, MAPE evaluator might be available. Otherwise, custom calculation is needed.
            print(f"Validation RMSE: {rmse:.2f}")
        else:
            print("No validation data available for the split.")

        # Save the Spark ML PipelineModel
        model_path_local = os.path.join(model_output_dir, "spark_ml_pipeline_model")
        if os.path.exists(model_path_local):
            import shutil
            shutil.rmtree(model_path_local) # Clear previous model if exists
        model.save(model_path_local)
        print(f"Spark ML PipelineModel saved to {model_path_local}")


    except Exception as e:
        print(f"Error during Spark ML model training: {e}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Forecasting Model using Spark ML.")
    parser.add_argument("--training_end_date", type=str, help="End date for training data (YYYY-MM-DD).")
    args = parser.parse_args()

    if not args.training_end_date:
        print("Please provide a training end date using --training_end_date (YYYY-MM-DD).")
        exit(1)

    os.makedirs("./models", exist_ok=True)
    train_forecasting_model(args.training_end_date)
