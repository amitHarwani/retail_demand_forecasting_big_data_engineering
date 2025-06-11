# dags/demand_forecast_pipeline_dag.py
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from datetime import datetime, timedelta
import os
import pandas as pd # Used to get date range for simulation
import json # Used to read sales_train_partitioned_info.json

# Define local script paths (absolute paths within the Docker container)
# These scripts are expected to be in /opt/airflow/dags/
SPLIT_DATA_SCRIPT = "./dags/split_data.py"
INITIAL_INGEST_SCRIPT = "./dags/initial_ingest.py"
DAILY_INGEST_SCRIPT = "./dags/ingest_daily_slice.py"
TRANSFORM_SCRIPT = "./dags/transform_data.py"
TRAIN_MODEL_SCRIPT = "./dags/train_model.py"
PREDICT_DAILY_SCRIPT = "./dags/predict_daily.py"

WORKING_DIR = "/opt/airflow"


@dag(
    dag_id='demand_forecast_initial_training', 
    schedule_interval=None, # Set to None as this DAG is for manual triggering of the PoC flow
    catchup=False,
)
def demand_forecast_initial_train_pipeline():
      # --- Initial Setup Tasks (Run Once, typically manually triggered) ---
    @task.bash(cwd=WORKING_DIR)
    def run_data_splitting():
        """Splits the raw Kaggle data into training and daily ingestion simulation sets locally."""
        return f"python {SPLIT_DATA_SCRIPT}"


    @task.bash(cwd=WORKING_DIR)
    def run_initial_ingestion():
        """Performs initial ingestion of training data and static files to MinIO raw zone."""
        return f"python {INITIAL_INGEST_SCRIPT}"


    @task.bash(cwd=WORKING_DIR)
    def run_initial_transformation_for_training_data():
        return f"spark-submit --master local[*] --driver-memory 4g --executor-memory 4g --packages org.apache.hadoop:hadoop-aws:3.4.1 {TRANSFORM_SCRIPT}"


    @task.bash(cwd=WORKING_DIR)
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

        return f"spark-submit --master local[*] --driver-memory 4g --executor-memory 4g --packages org.apache.hadoop:hadoop-aws:3.4.1 {TRAIN_MODEL_SCRIPT} --training_end_date {training_end_date}"

    
    split = run_data_splitting()
    ingest = run_initial_ingestion()
    transform = run_initial_transformation_for_training_data()
    train = run_model_training()

    split >> ingest >> transform >> train
    
@dag(
    dag_id='demand_forecast_pipeline', 
    schedule_interval=None, # Set to None as this DAG is for manual triggering of the PoC flow
    catchup=False,
)
def demand_forecast_pipeline():

    # --- Incremental Daily Tasks (single-date) ---
    @task
    def get_next_date():
        # Read last_processed_date or default to training end date
        last = Variable.get(
            'last_processed_date',
            default_var=Variable.get('initial_training_end_date', '')
        )
        last_dt = datetime.strptime(last, '%Y-%m-%d')
        next_dt = last_dt + timedelta(days=1)
        next_str = next_dt.strftime('%Y-%m-%d')
        # Store for next run
        Variable.set('last_processed_date', next_str)
        return next_str

    @task.bash(cwd=WORKING_DIR)
    def daily_ingest_slice_task(ingestion_date_str):
        """Simulates daily ingestion of sales data slice to MinIO's raw zone."""
        return f"python {DAILY_INGEST_SCRIPT} --ingestion_date {ingestion_date_str}"


    @task.bash(cwd=WORKING_DIR)
    def daily_transform_and_feature_engineer_task(process_date_str):
        """Performs incremental transformation and feature engineering for the given date, appending to refined zone."""
        return f"spark-submit --master local[*] --driver-memory 4g --executor-memory 4g --packages org.apache.hadoop:hadoop-aws:3.4.1 {TRANSFORM_SCRIPT} --process_date {process_date_str} --mode append"


    @task.bash(cwd=WORKING_DIR)
    def daily_prediction_and_load_task(next_date):

        # Add 1 as forecasts needs to be generated for the next date
        date_passed = datetime.strptime(next_date, '%Y-%m-%d')
        forecast_date = date_passed + timedelta(days=1)
        forecast_date = forecast_date.strftime('%Y-%m-%d')
    
        """Generates 7-day forecasts across all product ranges and stores for the given date, and loads to PostgreSQL."""
        return f"spark-submit --master local[*] --driver-memory 4g --executor-memory 4g --packages org.apache.hadoop:hadoop-aws:3.4.1 {PREDICT_DAILY_SCRIPT} --forecast_date {forecast_date}"


    # --- DAG Flow Definition ---
    next_date = get_next_date()
    
    ingest = daily_ingest_slice_task(next_date)
    transform = daily_transform_and_feature_engineer_task(next_date)
    predict = daily_prediction_and_load_task(next_date)

    ingest >> transform >> predict

# Instantiate the DAG
demand_forecast_initial_train = demand_forecast_initial_train_pipeline()
demand_forecast_main = demand_forecast_pipeline()
