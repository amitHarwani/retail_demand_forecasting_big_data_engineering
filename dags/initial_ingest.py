# initial_ingest.py
import pandas as pd
import os
import boto3
from botocore.client import Config
import json # New import

# MinIO configuration
MINIO_ENDPOINT = os.environ.get('MINIO_ENDPOINT', "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', "minioadmin")
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', "minioadmin")

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

def upload_df_to_minio_parquet(df, bucket_name, object_prefix):
    """Uploads a Pandas DataFrame to MinIO as Parquet."""

    if 'date' in df.columns:
        df['date'] = df['date'].dt.date

    local_parquet_path = f"./temp_data_{pd.Timestamp.now().timestamp()}.parquet" # Unique name
    df.to_parquet(local_parquet_path, index=False)
    
    object_name = f"{object_prefix}/data.parquet" # Store within the partitioned path
    try:
        S3_CLIENT.upload_file(local_parquet_path, bucket_name, object_name)
        print(f"Successfully uploaded daily data slice to {bucket_name}/{object_name}")
    except Exception as e:
        print(f"Error uploading daily data slice: {e}")
        raise
    finally:
        if os.path.exists(local_parquet_path):
            os.remove(local_parquet_path)

def initial_ingestion(source_path='./data/split/', bucket_name='raw'):
    """Uploads initial training data (sales_train) partitioned, and static files to MinIO raw zone."""
    print(f"Starting initial ingestion from {source_path} to MinIO bucket '{bucket_name}'...")
    
    # Upload sales_train data partitioned by date
    sales_train_info_path = os.path.join(source_path, 'sales_train_partitioned_info.json')
    if os.path.exists(sales_train_info_path):
        with open(sales_train_info_path, 'r') as f:
            train_sales_partitions = json.load(f)
        
        for partition_info in train_sales_partitions:
            date_str = partition_info['date']
            file_path = partition_info['file_path']
            
            if os.path.exists(file_path):
                daily_slice_df = pd.read_csv(file_path, parse_dates=['date'])
                date_obj = pd.to_datetime(date_str).date()
                object_prefix = f"sales/year={date_obj.year}/month={date_obj.month:02d}/day={date_obj.day:02d}"
                upload_df_to_minio_parquet(daily_slice_df, bucket_name, object_prefix)
                os.remove(file_path) # Clean up temp CSV
            else:
                print(f"Warning: Temporary sales train file {file_path} not found. Skipping.")
        print("Initial partitioned sales_train data uploaded.")
    else:
        print(f"Warning: {sales_train_info_path} not found. sales_train data will not be partitioned.")

    # Upload static files under a 'static/' prefix
    files_to_upload_static = ['markdowns.csv', 'price_history.csv', 'discounts_history.csv',
                              'actual_matrix.csv', 'catalog.csv', 'stores.csv']
    for filename in files_to_upload_static:
        file_path = os.path.join(source_path, filename)
        if os.path.exists(file_path):
            upload_to_minio(file_path, bucket_name, f"static/{filename}")
        else:
            print(f"Warning: {file_path} not found. Skipping.")

if __name__ == "__main__":
    initial_ingestion()