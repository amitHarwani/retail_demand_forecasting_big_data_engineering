# One-off script to load stores.csv to PostgreSQL
import pandas as pd
from sqlalchemy import create_engine
import os
import boto3
from botocore.client import Config

# MinIO configuration (same as others)
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

S3_CLIENT = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

PG_CONN_STR = "postgresql://airflow:airflow@localhost:5432/airflow" # Hostname 'localhost' if running script outside docker-compose network
pg_engine = create_engine(PG_CONN_STR)

local_stores_path = "./data/stores.csv" # Temporary local path
S3_CLIENT.download_file("raw", "static/stores.csv", local_stores_path)
stores_df = pd.read_csv(local_stores_path)
stores_df.to_sql('stores', pg_engine, if_exists='replace', index=False)
print("Stores data loaded to PostgreSQL 'stores' table for Superset.")
os.remove(local_stores_path)