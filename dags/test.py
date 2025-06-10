# transform_data.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, dayofweek, month, year, weekofyear, lag, avg, expr, lit, dayofmonth, to_date
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StringType, StructField
from datetime import date, timedelta
import argparse

# MinIO configuration
MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

def create_spark_session():
    """Creates and configures a SparkSession for MinIO integration."""
    spark = SparkSession.builder \
        .appName("DataTransformation") \
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

def transform_and_feature_engineer(spark):

    
    
    df = spark.read.parquet("s3a://transformed/forecasting_features/year=2023/month=8/day=21")
    print(df.count())

if __name__ == "__main__":

    spark = create_spark_session()
    transform_and_feature_engineer(spark)
    spark.stop()
