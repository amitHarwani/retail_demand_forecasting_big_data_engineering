import pandas as pd
import os
from datetime import timedelta
import json

def split_dataset_by_date(raw_data_path='./data/raw/', split_data_path='./data/split/'):
    """
    Splits the sales data into training and daily ingestion simulation sets.
    - Training data: Most of the historical data for initial model training.
    - Daily Ingestion Simulation data: A recent segment of sales data to simulate daily new arrivals.
    
    Also, prepares metadata for partitioned upload of sales_train.
    """
    print("Starting data splitting...")
    os.makedirs(split_data_path, exist_ok=True)

    sales_df = pd.read_csv(os.path.join(raw_data_path, 'sales.csv'), parse_dates=['date'])
    
    # Sort by date to ensure correct splits
    sales_df = sales_df.sort_values(by='date').reset_index(drop=True)

    min_date_sales = sales_df['date'].min()
    max_date_sales = sales_df['date'].max()

    # Determine split point: last 30 days of sales_df for daily ingestion simulation
    ingestion_sim_start_date = max_date_sales - timedelta(days=30) # Inclusive of start date

    # Split sales data
    train_sales_df = sales_df[sales_df['date'] < ingestion_sim_start_date]
    daily_ingestion_sales_df = sales_df[sales_df['date'] >= ingestion_sim_start_date]

    print(f"Sales Training Data Range: {train_sales_df['date'].min().strftime('%Y-%m-%d')} to {train_sales_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Sales Daily Ingestion Simulation Data Range: {daily_ingestion_sales_df['date'].min().strftime('%Y-%m-%d')} to {daily_ingestion_sales_df['date'].max().strftime('%Y-%m-%d')}")
    
    # Save split data locally
    # daily_ingestion_sales_df will be read daily by ingest_daily_slice.py
    daily_ingestion_sales_df.to_csv(os.path.join(split_data_path, 'sales_ingest_sim.csv'), index=False)
    
    # Save train_sales_df as a list of daily dataframes to be uploaded partitioned
    train_sales_dates = sorted(train_sales_df['date'].unique())
    train_sales_partition_info = []
    for date_ts in train_sales_dates:
        date_str = pd.to_datetime(date_ts).strftime('%Y-%m-%d') # Converting to string of format YYYY-MM-DD
        daily_train_slice = train_sales_df[train_sales_df['date'] == date_ts]
        local_temp_path = os.path.join(split_data_path, f"sales_train_temp_{date_str}.csv")
        daily_train_slice.to_csv(local_temp_path, index=False)
        train_sales_partition_info.append({
            'date': date_str,
            'file_path': local_temp_path
        })
    
    # Save the info about partitioned train sales data
    with open(os.path.join(split_data_path, 'sales_train_partitioned_info.json'), 'w') as f:
        json.dump(train_sales_partition_info, f, indent=4)
    print("Prepared sales_train data for partitioned upload.")

    # Copy other static metadata directly to split directory.
    # These will be uploaded to MinIO raw zone under a 'static' prefix.
    other_files = ['markdowns.csv', 'price_history.csv', 'discounts_history.csv',
                   'actual_matrix.csv', 'catalog.csv', 'stores.csv'] # test.csv is now a static input for daily prediction
    for f in other_files:
        raw_file_path = os.path.join(raw_data_path, f)
        split_file_path = os.path.join(split_data_path, f)
        if os.path.exists(raw_file_path):
            pd.read_csv(raw_file_path).to_csv(split_file_path, index=False)
            print(f"Copied {f} to split directory.")
        else:
            print(f"Warning: {raw_file_path} not found. Skipping.")
            
    print("Data splitting completed. Files saved to ./data/split/")

if __name__ == "__main__":
    split_dataset_by_date()