# ingest_daily_slice.py
import pandas as pd
import os
import boto3
from botocore.client import Config
import argparse
from datetime import date # New import

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

def upload_df_to_minio_parquet(df, bucket_name, object_prefix):
    """
    Uploads a Pandas DataFrame to MinIO as Parquet.
    Ensures 'date' column is written as Parquet DATE type.
    """
    # Convert 'date' column to Python date objects before writing to Parquet
    # This ensures pyarrow writes it as a Parquet DATE type, which is more compatible.
    if 'date' in df.columns:
        df['date'] = df['date'].dt.date # Convert datetime to date object

    local_parquet_path = f"./temp_daily_data_{pd.Timestamp.now().timestamp()}.parquet" # Unique name to avoid conflicts
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


def ingest_daily_slice(ingestion_date_str, source_file='./data/split/sales_ingest_sim.csv', bucket_name='raw'):
    """
    Simulates daily ingestion by reading a slice of data for the given date
    from a local simulation file and uploading it to MinIO.
    """
    print(f"Ingesting daily sales slice for {ingestion_date_str} from {source_file}...")
    try:
        full_ingest_sim_df = pd.read_csv(source_file, parse_dates=['date']) # Read as datetime
        # Extract data for ingestion date
        daily_slice_df = full_ingest_sim_df[full_ingest_sim_df['date'].dt.strftime('%Y-%m-%d') == ingestion_date_str]

        if daily_slice_df.empty:
            print(f"No sales data for {ingestion_date_str} in simulation file. Skipping upload.")
            return

        # Define object path with date partitioning
        date_obj = pd.to_datetime(ingestion_date_str).date() # Ensure date_obj is Python date
        object_prefix = f"sales/year={date_obj.year}/month={date_obj.month:02d}/day={date_obj.day:02d}"
        
        upload_df_to_minio_parquet(daily_slice_df, bucket_name, object_prefix)
        print(f"Daily ingestion for {ingestion_date_str} completed.")

    except Exception as e:
        print(f"Error during daily ingestion for {ingestion_date_str}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest daily sales slice.")
    parser.add_argument("--ingestion_date", type=str, help="Date to ingest (YYYY-MM-DD).")
    args = parser.parse_args()

    if not args.ingestion_date:
        print("Please provide an ingestion date using --ingestion_date (YYYY-MM-DD).")
        exit(1)
        
    ingest_daily_slice(args.ingestion_date)
