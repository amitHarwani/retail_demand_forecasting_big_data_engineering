# dags/demand_forecast_pipeline_dag.py
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os
import pandas as pd # Used to get date range for simulation
import json # Used to read sales_train_partitioned_info.json

# Define local script paths (absolute paths within the Docker container)
# These scripts are expected to be in /opt/airflow/dags/
SPLIT_DATA_SCRIPT = "/opt/airflow/dags/split_data.py"
INITIAL_INGEST_SCRIPT = "/opt/airflow/dags/initial_ingest.py"
DAILY_INGEST_SCRIPT = "/opt/airflow/dags/ingest_daily_slice.py"
TRANSFORM_SCRIPT = "/opt/airflow/dags/transform_data.py"
TRAIN_MODEL_SCRIPT = "/opt/airflow/dags/train_model.py"
PREDICT_DAILY_SCRIPT = "/opt/airflow/dags/predict_daily.py"

@dag(
    dag_id='demand_forecast_pipeline_poc_v2_optimized_final', # Changed DAG ID to reflect final optimization
    start_date=days_ago(1), # This will be the earliest date Airflow considers for scheduling
    schedule_interval=None, # Set to None as this DAG is for manual triggering of the PoC flow
    catchup=False,
    tags=['demand_forecast', 'poc', 'big_data', 'optimized', 'final'],
)
def demand_forecast_pipeline_poc_optimized_final():

    # --- Initial Setup Tasks (Run Once, typically manually triggered) ---
    @task
    def run_data_splitting():
        """Splits the raw Kaggle data into training and daily ingestion simulation sets locally."""
        return BashOperator(
            task_id='split_raw_data',
            bash_command=f"python {SPLIT_DATA_SCRIPT}",
            do_xcom_push=False
        )

    @task
    def run_initial_ingestion():
        """Performs initial ingestion of training data and static files to MinIO raw zone."""
        return BashOperator(
            task_id='initial_ingest_to_raw',
            bash_command=f"python {INITIAL_INGEST_SCRIPT}",
            do_xcom_push=False
        )

    @task
    def run_initial_transformation_for_training_data():
        """Transforms initial training data and static files, writes to refined zone (overwrite)."""
        # Determine the end date of the training data split to correctly process it
        try:
            with open('/opt/airflow/data/split/sales_train_partitioned_info.json', 'r') as f:
                sales_train_info = json.load(f)
            training_data_end_date = max([item['date'] for item in sales_train_info])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            print("sales_train_partitioned_info.json not found or invalid locally. Ensure 'split_raw_data' task runs first.")
            # Fallback for initial run if split_data.py not run manually prior
            training_data_end_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d') # Placeholder

        return BashOperator(
            task_id='initial_transform_for_training',
            # Overwrite the refined zone with initial training data features
            bash_command=f"spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.1,com.amazonaws:aws-java-sdk-bundle:1.11.901 {TRANSFORM_SCRIPT} --process_date {training_data_end_date} --mode overwrite",
            do_xcom_push=False
        )

    @task
    def run_model_training():
        """Trains the demand forecasting model using the initial transformed data."""
        # Determine the end date of the training data split for the model
        try:
            with open('/opt/airflow/data/split/sales_train_partitioned_info.json', 'r') as f:
                sales_train_info = json.load(f)
            training_end_date = max([item['date'] for item in sales_train_info])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            print("sales_train_partitioned_info.json not found or invalid locally. Ensure 'split_raw_data' task runs first.")
            training_end_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d') # Placeholder

        return BashOperator(
            task_id='train_forecasting_model',
            bash_command=f"python {TRAIN_MODEL_SCRIPT} --training_end_date {training_end_date}",
            do_xcom_push=False
        )
    
    # --- Daily Pipeline Tasks (Simulated Loop) ---
    # This task extracts all unique dates from the daily ingestion simulation file.
    # The subsequent tasks will be dynamically created for each of these dates.
    @task
    def get_daily_ingestion_dates():
        """Reads sales_ingest_sim.csv to get the range of dates for daily ingestion simulation."""
        try:
            ingest_sim_df = pd.read_csv('/opt/airflow/data/split/sales_ingest_sim.csv', parse_dates=['date'])
        except FileNotFoundError:
            print("sales_ingest_sim.csv not found locally. Ensure 'split_raw_data' task runs first.")
            return [] # If file not found, no dynamic tasks will be created

        if ingest_sim_df.empty:
            print("No daily ingestion simulation data found in sales_ingest_sim.csv.")
            return []

        unique_dates = sorted(ingest_sim_df['date'].unique())
        # Convert numpy datetime64 to string format (YYYY-MM-DD) for task parameters
        daily_ingestion_dates = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in unique_dates]
        
        print(f"Daily ingestion simulation dates identified: {daily_ingestion_dates}")
        return daily_ingestion_dates

    @task
    def daily_ingest_slice_task(ingestion_date_str):
        """Simulates daily ingestion of a sales data slice to MinIO's raw zone."""
        return BashOperator(
            task_id=f'ingest_daily_slice_{ingestion_date_str.replace("-", "_")}',
            bash_command=f"python {DAILY_INGEST_SCRIPT} --ingestion_date {ingestion_date_str}",
            do_xcom_push=False
        )

    @task
    def daily_transform_and_feature_engineer_task(process_date_str):
        """Performs incremental transformation and feature engineering for the given date, appending to refined zone."""
        return BashOperator(
            task_id=f'transform_daily_data_{process_date_str.replace("-", "_")}',
            bash_command=f"spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.1,com.amazonaws:aws-java-sdk-bundle:1.11.901 {TRANSFORM_SCRIPT} --process_date {process_date_str} --mode append", # Append for daily updates
            do_xcom_push=False
        )

    @task
    def daily_prediction_and_load_task(forecast_date_str):
        """Generates 7-day forecasts across all product ranges and stores for the given date, and loads to PostgreSQL."""
        return BashOperator(
            task_id=f'predict_daily_forecasts_{forecast_date_str.replace("-", "_")}',
            bash_command=f"python {PREDICT_DAILY_SCRIPT} --forecast_date {forecast_date_str}",
            do_xcom_push=False
        )

    # --- DAG Flow Definition ---

    # 1. Initial Setup Chain:
    # This chain handles preparing the split data, performing initial ingestion,
    # transforming the training data, and training the first model.
    # It should be run manually *once* when setting up the PoC environment.
    split_data_task = run_data_splitting()
    initial_ingest_task = run_initial_ingestion()
    initial_transform_task = run_initial_transformation_for_training_data()
    train_model_task = run_model_training()

    # Define the dependencies for the initial setup.
    split_data_task >> initial_ingest_task >> initial_transform_task >> train_model_task

    # 2. Daily Pipeline Loop:
    # This part simulates the daily operational pipeline.
    # It dynamically creates tasks for each day in the `sales_ingest_sim.csv` data.
    
    # First, get the list of dates for the daily simulation. This task runs once.
    ingestion_dates_producer_task = get_daily_ingestion_dates()
    
    # Then, expand the daily tasks based on the list of dates from the producer task.
    daily_ingest_tasks = daily_ingest_slice_task.partial().expand(ingestion_date_str=ingestion_dates_producer_task)
    daily_transform_tasks = daily_transform_and_feature_engineer_task.partial().expand(process_date_str=ingestion_dates_producer_task)
    daily_predict_tasks = daily_prediction_and_load_task.partial().expand(forecast_date_str=ingestion_dates_producer_task)

    # Define dependencies for the daily pipeline tasks.
    # The daily ingestion and transformation steps depend on the initial setup completing successfully.
    # `train_model_task` must complete before any `daily_predict_tasks` run, as they need the trained model.
    
    # Ensure the daily ingestion starts only after the initial model training is complete
    train_model_task >> daily_ingest_tasks
    
    # Each daily transformation depends on its corresponding daily ingestion
    daily_ingest_tasks >> daily_transform_tasks
    
    # Each daily prediction depends on its corresponding daily transformation
    daily_transform_tasks >> daily_predict_tasks

# Instantiate the DAG
demand_forecast_pipeline = demand_forecast_pipeline_poc_optimized_final()
