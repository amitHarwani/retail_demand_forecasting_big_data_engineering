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

def transform_and_feature_engineer(spark, process_date_str=None, mode="overwrite"):
    """
    Reads raw data, performs transformations and feature engineering,
    and writes to the transformed zone.
    
    If process_date_str is None (initial training data load):
        - Loads all historical sales data from 's3a://raw/sales/year=*/month=*/day=*/'.
    If process_date_str is provided (daily incremental load):
        - Loads sales data for a specific look-back window (e.g., last 365 days)
          relative to process_date_str to compute features accurately.
          Then filters the final output to only include data for `process_date_str`.
    """
    print(f"Starting Spark transformation for process_date: {process_date_str}, mode: {mode}")

    
    # Load metadata files from the 'static' prefix in raw zone
    try:
        markdowns_df = spark.read.csv(f"s3a://raw/static/markdowns.csv", header=True, inferSchema=True)
        price_history_df = spark.read.csv(f"s3a://raw/static/price_history.csv", header=True, inferSchema=True)
        discounts_history_df = spark.read.csv(f"s3a://raw/static/discounts_history.csv", header=True, inferSchema=True)
        actual_matrix_df = spark.read.csv(f"s3a://raw/static/actual_matrix.csv", header=True, inferSchema=True)
        catalog_df = spark.read.csv(f"s3a://raw/static/catalog.csv", header=True, inferSchema=True)
        stores_df = spark.read.csv(f"s3a://raw/static/stores.csv", header=True, inferSchema=True)
    except Exception as e:
        print(f"Error loading static data from MinIO: {e}")
        raise
    
    # Convert 'date' columns to date type (where applicable)
    markdowns_df = markdowns_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    price_history_df = price_history_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    discounts_history_df = discounts_history_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    actual_matrix_df = actual_matrix_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

    # Determine the date range for sales data loading to optimize for lag/rolling calculations
    if process_date_str is None:
        # Initial full load for training data - read all available sales data
        # This assumes all sales_train data has been partitioned and uploaded to raw/sales/
        sales_df_path = f"s3a://raw/sales/year=*/month=*/day=*/"
        print(f"Loading all sales data from {sales_df_path} for initial full transformation...")
        sales_df = spark.read.parquet(sales_df_path)
    else:
        # Daily incremental load - load sales data for a specific look-back window
        # Max lag is 365 days, so need to fetch at least 365 days prior to process_date
        process_date = date.fromisoformat(process_date_str)
        look_back_start_date = process_date - timedelta(days=365) # Max look back for lags/rolling
        
        # Construct paths for the required date range
        date_paths = []
        current_date = look_back_start_date
        while current_date <= process_date:
            date_paths.append(f"s3a://raw/sales/year={current_date.year}/month={current_date.month:02d}/day={current_date.day:02d}/")
            current_date += timedelta(days=1)
        
        # Read sales data only for the relevant window
        print(f"Loading sales data for look-back window ({look_back_start_date} to {process_date}) from {len(date_paths)} partitions...")
        try:
            sales_df = spark.read.parquet(*date_paths) # Use * to unpack list of paths
            sales_df = sales_df.filter(col("date") >= look_back_start_date).filter(col("date") <= process_date)
        except Exception as e:
            print(f"Error reading sales data for look-back window. This might be expected if paths don't exist yet: {e}")
            # This is a heuristic, better to get schema dynamically from actual data
            from pyspark.sql.types import StructType, StructField, DateType, StringType, DoubleType, IntegerType
            sales_schema_for_empty_df = StructType([
                StructField("date", DateType(), True),
                StructField("item_id", IntegerType(), True),
                StructField("quantity", DoubleType(), True),
                StructField("price_base", DoubleType(), True),
                StructField("sum_total", DoubleType(), True),
                StructField("store_id", IntegerType(), True)
            ])
            sales_df = spark.createDataFrame([], schema=sales_schema_for_empty_df) # Empty DataFrame with correct schema

    # 1. Data Cleaning and Type Conversion 
    sales_df = sales_df.na.fill(0, subset=['quantity', 'price_base', 'sum_total'])
    markdowns_df = markdowns_df.na.fill(0, subset=['quantity', 'price', 'normal_price'])

    # Extracting total_quantity and avg base price of an item at a store on a date
    sales_agg_df = sales_df.groupBy("date", "item_id", "store_id") \
                           .agg(
                               expr("sum(quantity)").alias("total_quantity"),
                               expr("avg(price_base)").alias("avg_base_price") # Use avg_base_price as a feature
                           )

    # 2. Data Transformation - Joining Datasets
    # Joining catalog i.e. Item information
    enriched_df = sales_agg_df.join(catalog_df, on="item_id", how="left")

    # Joining store information
    enriched_df = enriched_df.join(stores_df, on="store_id", how="left")
    
    # Extracting quantity of products and price sold at markdown by item, date and store
    markdowns_agg = markdowns_df.groupBy("date", "item_id", "store_id") \
                               .agg(
                                   expr("sum(quantity)").alias("markdown_quantity"),
                                   expr("avg(price)").alias("markdown_price"),
                                   expr("avg(normal_price)").alias("markdown_normal_price")
                               ).alias("m")
    
    # Joining enriched data with markdowns aggregated, and filling 0 for na
    enriched_df = enriched_df.join(markdowns_agg, on=["date", "item_id", "store_id"], how="left") \
                             .na.fill(0, subset=['markdown_quantity', 'markdown_price', "markdown_normal_price"])

    # Extracting max promo type code and sale price during the promo for an item at particular date and store
    discounts_agg = discounts_history_df.groupBy("date", "item_id", "store_id") \
                                     .agg(
                                         expr("max(promo_type_code)").alias("promo_type_code"),
                                         expr("avg(sale_price_time_promo)").alias("promo_price"),
                                         expr("avg(sale_price_before_promo)").alias("discount_price_before_promo")
                                     ).alias("d")
    
    # Joining the enriched data with the discounts data.
    enriched_df = enriched_df.join(discounts_agg, on=["date", "item_id", "store_id"], how="left") \
                             .na.fill(0, subset=['promo_price', 'discount_price_before_promo']) # Fill new feature


    # 3. Feature Engineering - Adding day of week, month, year, the week of the year, and whether the day is a weekend or not.
    enriched_df = enriched_df.withColumn("day_of_week", dayofweek(col("date"))) \
                             .withColumn("month", month(col("date"))) \
                             .withColumn("year", year(col("date"))) \
                             .withColumn("week_of_year", weekofyear(col("date"))) \
                             .withColumn("day", dayofmonth(col("date")))
    # 1 is Sunday, 7 is Saturday.
    enriched_df = enriched_df.withColumn("is_weekend", ((col("day_of_week") == 1) | (col("day_of_week") == 7)).cast("integer"))

    # Lagged sales features & Rolling averages (Max lag 365, rolling 30)
    # Group by item and store id i.e. calculate lag and rolling features for each item and store id combination.
    window_spec_item_store = Window.partitionBy("item_id", "store_id").orderBy("date")

    # Get the demand from 7, 30, 90 and 365 prior for each item store combination (Specified by window_spec_item_store)
    enriched_df = enriched_df.withColumn("lag_7_quantity", lag(col("total_quantity"), 7).over(window_spec_item_store))
    enriched_df = enriched_df.withColumn("lag_30_quantity", lag(col("total_quantity"), 30).over(window_spec_item_store))
    enriched_df = enriched_df.withColumn("lag_90_quantity", lag(col("total_quantity"), 90).over(window_spec_item_store))
    enriched_df = enriched_df.withColumn("lag_365_quantity", lag(col("total_quantity"), 365).over(window_spec_item_store))

    # Average of quantity sold in the last 7 and last 30 days.
    enriched_df = enriched_df.withColumn("rolling_mean_7_quantity", avg(col("total_quantity")).over(window_spec_item_store.rowsBetween(-6, 0)))
    enriched_df = enriched_df.withColumn("rolling_mean_30_quantity", avg(col("total_quantity")).over(window_spec_item_store.rowsBetween(-29, 0)))

    # If promo_type_code exists or markdown_quantity > 0
    enriched_df = enriched_df.withColumn("on_promotion", (col("promo_type_code").isNotNull() | (col("markdown_quantity") > 0)).cast("integer"))

    # Fill NA for new features
    numeric_cols_to_fill = [
        "lag_7_quantity", "lag_30_quantity", "lag_90_quantity", "lag_365_quantity",
        "rolling_mean_7_quantity", "rolling_mean_30_quantity",
        "markdown_quantity", "markdown_price", "markdown_normal_price", "promo_price", "avg_base_price", "discount_price_before_promo"
    ]
    enriched_df = enriched_df.na.fill(0, subset=numeric_cols_to_fill)
    categorical_cols_to_fill = ['dept_name', 'class_name', 'subclass_name', 'item_type',
                                'division', 'format', 'city', 'area', 'promo_type_code']
    for col_name in categorical_cols_to_fill:
        if col_name in enriched_df.columns:
            enriched_df = enriched_df.na.fill("unknown", subset=[col_name])

    # Select relevant columns for the final dataset (target variable and features for ML)
    # `total_sales_amount` is explicitly EXCLUDED as an ML feature.
    final_features_df = enriched_df.select(
        "date", "item_id", "store_id", "total_quantity", # total_quantity is the TARGET
        "avg_base_price", "markdown_quantity", "markdown_price", "markdown_normal_price", "promo_price", "discount_price_before_promo",
        "on_promotion", "day_of_week", "month", "year", "week_of_year", "is_weekend", "day",
        "lag_7_quantity", "lag_30_quantity", "lag_90_quantity", "lag_365_quantity",
        "rolling_mean_7_quantity", "rolling_mean_30_quantity",
        "dept_name", "class_name", "subclass_name", "item_type",
        "division", "format", "city", "area"
    )
    
    # If `process_date_str` is specified (daily run), filter to write only the data for that specific day
    if process_date_str is not None:
        final_features_df = final_features_df.filter(col("date") == to_date(lit(process_date_str), "yyyy-MM-dd"))

    # Write to MinIO transformed zone in Parquet format, partitioned by date
    output_path = f"s3a://transformed/forecasting_features/"
    print(f"Writing transformed data to {output_path} with mode '{mode}'...")
    final_features_df.write.mode(mode).partitionBy("year", "month", "day").parquet(output_path)
    print("Spark transformation completed. Data available in MinIO 'transformed' bucket.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark Data Transformation and Feature Engineering.")
    parser.add_argument("--process_date", type=str, help="Date up to which to process data (YYYY-MM-DD).")
    parser.add_argument("--mode", type=str, default="overwrite", choices=["overwrite", "append"],
                        help="Write mode for Parquet (overwrite or append).")
    args = parser.parse_args()

    spark = create_spark_session()
    transform_and_feature_engineer(spark, process_date_str=args.process_date, mode=args.mode)
    spark.stop()
