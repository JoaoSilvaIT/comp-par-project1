"""
run_pipeline.py
---------------
Standalone, spark-submit-ready PySpark pipeline for CompPar Project 1.
Implements:
  - M1: Data Ingestion, Schema Enforcement, Cleaning & Per-Year Removal Statistics
  - M2: Temporal Demand Analysis (Partitioned Parquet Output)
  - M3: Zone-Level Statistics & Rolling Window Anomaly Detection
  - O1: Fare Anomaly Detection (Weekday vs. Weekend)
  - O3: Demand Forecasting Features (4-Week Lag via Self-Join)

Usage:
    # Inside Docker cluster:
    spark-submit --master spark://spark-master:7077 \\
        /apps/project1/run_pipeline.py \\
        --data /data/project1/ --zones /data/project1/taxi_zone_lookup.csv \\
        --output /data/project1/output/

    # Local smoke-test with 10% sample:
    python run_pipeline.py --data ../data/ --sample 0.1 --master "local[2]"
"""

import argparse
import os
import sys
import time

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as sf
from pyspark.sql.functions import col, to_timestamp

try:
    from spark_profiler import SparkProfiler
except ImportError:
    try:
        from notebooks.spark_profiler import SparkProfiler
    except ImportError:
        SparkProfiler = None


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="/data/project1/",
                        help="Folder containing yellow_tripdata_*.parquet")
    parser.add_argument("--zones", default=None,
                        help="Path to taxi_zone_lookup.csv (defaults to <data>/taxi_zone_lookup.csv)")
    parser.add_argument("--output", default=None,
                        help="Output folder for results (defaults to <data>/output/)")
    parser.add_argument("--sample", type=float, default=None,
                        help="Fraction of records to sample for smoke testing (e.g. 0.1 for 10%%)")
    parser.add_argument("--year", type=int, default=None,
                        help="Filter data for a specific year (e.g. 2019)")
    parser.add_argument("--master", default=None,
                        help="Spark master URL (defaults to env SPARK_MASTER or local[*])")
    parser.add_argument("--profile", action="store_true",
                        help="Enable SparkProfiler stage & shuffle metrics collection")
    return parser


def load_lookup(spark, zones_path):
    print(f"[Ingestion] Loading zone lookup: {zones_path}")
    zones = spark.read.option("delimiter", ",").option("header", True).csv(zones_path)
    pu = zones.select(
        sf.col("LocationID").cast("int").alias("PULocationID"),
        sf.col("Borough").alias("PU_Borough"),
        sf.col("Zone").alias("PU_Zone"),
        sf.col("service_zone").alias("PU_service_zone")
    )
    return zones, pu


def load_trips(spark, data_dir, sample_fraction=None, target_year=None):
    if os.path.isfile(data_dir):
        parquet_path = data_dir
    else:
        if target_year:
            parquet_path = os.path.join(data_dir, f"yellow_tripdata_{target_year}-*.parquet")
        else:
            parquet_path = os.path.join(data_dir, "yellow_tripdata_*.parquet")

    print(f"[Ingestion] Reading Parquet files: {parquet_path}")
    df = spark.read.parquet(parquet_path)

    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        print(f"[Ingestion] Sampling {sample_fraction * 100:.1f}% of records for testing...")
        df = df.sample(fraction=sample_fraction, seed=42)

    df_organized = df.select(
        to_timestamp(col("tpep_pickup_datetime")).alias("tpep_pickup_datetime"),
        to_timestamp(col("tpep_dropoff_datetime")).alias("tpep_dropoff_datetime"),
        col("PULocationID").cast("int").alias("PULocationID"),
        col("DOLocationID").cast("int").alias("DOLocationID"),
        col("passenger_count").cast("int").alias("passenger_count"),
        col("trip_distance").cast("double").alias("trip_distance"),
        col("fare_amount").cast("double").alias("fare_amount"),
        col("tip_amount").cast("double").alias("tip_amount"),
        col("payment_type").cast("int").alias("payment_type"),
        col("total_amount").cast("double").alias("total_amount")
    )
    return df_organized


def run_m1_cleaning(df):
    print("\n--- Running M1: Data Ingestion & Cleaning ---")
    df_flagged = (df
        .withColumn("year", sf.year("tpep_pickup_datetime"))
        .withColumn("drop_fare", (col("fare_amount") <= 0) | col("fare_amount").isNull())
        .withColumn("drop_distance", (col("trip_distance") <= 0) | col("trip_distance").isNull())
        .withColumn("drop_passenger", (col("passenger_count") <= 0) | (col("passenger_count") > 6) | col("passenger_count").isNull())
        .withColumn("drop_timestamps", (col("tpep_dropoff_datetime") <= col("tpep_pickup_datetime")) | col("tpep_dropoff_datetime").isNull() | col("tpep_pickup_datetime").isNull())
        .withColumn("drop_locations", (~col("PULocationID").between(1, 263)) | (~col("DOLocationID").between(1, 263)) | col("PULocationID").isNull() | col("DOLocationID").isNull()))

    cleaning_stats = (df_flagged.groupBy("year").agg(
        sf.count("*").alias("total_records"),
        sf.sum(col("drop_fare").cast("long")).alias("removed_fare_le_0"),
        sf.sum(col("drop_distance").cast("long")).alias("removed_distance_le_0"),
        sf.sum(col("drop_passenger").cast("long")).alias("removed_invalid_passengers"),
        sf.sum(col("drop_timestamps").cast("long")).alias("removed_invalid_timestamps"),
        sf.sum(col("drop_locations").cast("long")).alias("removed_invalid_locations")
    ).orderBy("year"))

    print("Per-year cleaning statistics:")
    cleaning_stats.show(truncate=False)

    cleaned_df = (df_flagged.filter(
        (~col("drop_fare")) &
        (~col("drop_distance")) &
        (~col("drop_passenger")) &
        (~col("drop_timestamps")) &
        (~col("drop_locations"))
    ).drop("drop_fare", "drop_distance", "drop_passenger", "drop_timestamps", "drop_locations"))

    cleaned_df = cleaned_df.cache()
    cleaned_count = cleaned_df.count()
    print(f"Cleaned records: {cleaned_count}")
    return cleaned_df, cleaning_stats


def run_m2_temporal(cleaned_df, pu, output_dir):
    print("\n--- Running M2: Temporal Demand Analysis ---")
    cleaned_with_pu = cleaned_df.join(sf.broadcast(pu), on="PULocationID", how="inner")

    result2 = (cleaned_with_pu
        .withColumn("day_of_week", sf.dayofweek("tpep_pickup_datetime"))
        .withColumn("hour_of_day", sf.hour("tpep_pickup_datetime"))
        .withColumn("duration_min",
                    (sf.unix_timestamp("tpep_dropoff_datetime") - sf.unix_timestamp("tpep_pickup_datetime")) / 60.0)
        .groupBy(sf.col("PU_Borough").alias("pickup_borough"), "day_of_week", "hour_of_day")
        .agg(
            sf.count("*").alias("trip_count"),
            sf.avg("fare_amount").alias("mean_fare_amount"),
            sf.avg("duration_min").alias("mean_duration_min"),
            sf.avg("trip_distance").alias("mean_trip_distance")
        ))

    output_path = os.path.join(output_dir, "borough_hour_stats")
    print(f"Writing M2 partitioned Parquet to {output_path}...")
    result2.write.mode("overwrite").partitionBy("pickup_borough").parquet(output_path)
    print("M2 Sample Output:")
    result2.orderBy("pickup_borough", "day_of_week", "hour_of_day").show(10)
    return result2


def run_m3_zone_stats(cleaned_df, pu):
    print("\n--- Running M3: Zone-Level Statistics ---")
    cleaned_with_pu = cleaned_df.join(sf.broadcast(pu), on="PULocationID", how="inner")

    zone_monthly = (cleaned_with_pu
        .withColumn("month", sf.date_trunc("month", "tpep_pickup_datetime"))
        .withColumn("month_idx", sf.year("month") * 12 + sf.month("month"))
        .groupBy(sf.col("PU_Borough").alias("pickup_borough"), "PULocationID", "PU_Zone", "month_idx", "month")
        .agg(
            sf.count("*").alias("trips"),
            sf.avg("fare_amount").alias("mean_fare_amount"),
            sf.stddev("fare_amount").alias("std_fare_amount"),
            sf.avg("trip_distance").alias("mean_trip_distance"),
            sf.stddev("trip_distance").alias("std_trip_distance")
        ))

    w_rank = Window.partitionBy("pickup_borough", "month_idx").orderBy(sf.desc("trips"))
    w_roll_zone = Window.partitionBy("pickup_borough", "PULocationID").orderBy("month_idx").rangeBetween(-2, 0)
    w_bor_month = Window.partitionBy("pickup_borough", "month_idx")

    result3 = (zone_monthly
        .withColumn("rank_in_borough", sf.rank().over(w_rank))
        .withColumn("roll3_trips", sf.sum("trips").over(w_roll_zone))
        .withColumn("roll3_share", sf.col("roll3_trips") / sf.sum("roll3_trips").over(w_bor_month))
        .withColumn("roll_mean", sf.avg("trips").over(w_roll_zone))
        .withColumn("roll_std", sf.stddev("trips").over(w_roll_zone))
        .withColumn("anomaly",
                    sf.coalesce(sf.abs(sf.col("trips") - sf.col("roll_mean")) > 1.5 * sf.col("roll_std"),
                                sf.lit(False))))

    print("M3 Sample Output:")
    result3.orderBy("pickup_borough", "month_idx", "rank_in_borough").show(10, truncate=False)
    return result3


def run_o1_fare_anomalies(cleaned_df, pu):
    print("\n--- Running O1: Fare Anomaly Detection ---")
    cleaned_with_pu = cleaned_df.join(sf.broadcast(pu), on="PULocationID", how="inner")
    w_zone_hour = Window.partitionBy("PULocationID", "hour_of_day")

    result_o1 = (cleaned_with_pu
        .withColumn("hour_of_day", sf.hour("tpep_pickup_datetime"))
        .withColumn("unit_fare", sf.col("fare_amount") / sf.col("trip_distance"))
        .withColumn("mean_unit_fare", sf.avg("unit_fare").over(w_zone_hour))
        .withColumn("sigma_unit_fare", sf.stddev("unit_fare").over(w_zone_hour))
        .withColumn("anomaly",
                    sf.coalesce(sf.col("unit_fare") > (sf.col("mean_unit_fare") + 3 * sf.col("sigma_unit_fare")),
                                sf.lit(False)))
        .withColumn("day_type",
                    sf.when(sf.dayofweek("tpep_pickup_datetime").isin(1, 7), "weekend").otherwise("weekday"))
        .groupBy(sf.col("PU_Borough").alias("pickup_borough"), "payment_type", "day_type")
        .agg(sf.count("*").alias("trips"),
             sf.sum(sf.col("anomaly").cast("int")).alias("anomalies"))
        .withColumn("anomaly_rate", sf.col("anomalies") / sf.col("trips"))
        .groupBy("pickup_borough", "payment_type")
        .pivot("day_type", ["weekday", "weekend"])
        .agg(sf.first("anomaly_rate"))
        .withColumn("weekend_minus_weekday", sf.col("weekend") - sf.col("weekday")))

    print("O1 Sample Output:")
    result_o1.orderBy("pickup_borough", "payment_type").show(20)
    return result_o1


def run_o3_demand_forecasting(cleaned_df, zones):
    print("\n--- Running O3: Demand Forecasting Features ---")
    daily_hourly = (cleaned_df
        .withColumn("pickup_date", sf.to_date("tpep_pickup_datetime"))
        .withColumn("day_of_week", sf.dayofweek("tpep_pickup_datetime"))
        .withColumn("hour_of_day", sf.hour("tpep_pickup_datetime"))
        .groupBy("PULocationID", "pickup_date", "day_of_week", "hour_of_day")
        .agg(sf.count("*").alias("trip_count")))

    hist_stats = (daily_hourly
        .groupBy("PULocationID", "day_of_week", "hour_of_day")
        .agg(
            sf.avg("trip_count").alias("historical_mean"),
            sf.variance("trip_count").alias("historical_variance")
        ))

    curr = daily_hourly.alias("curr")
    lag = daily_hourly.alias("lag")

    joined_lag = curr.join(
        lag,
        (sf.col("curr.PULocationID") == sf.col("lag.PULocationID")) &
        (sf.col("curr.hour_of_day") == sf.col("lag.hour_of_day")) &
        (sf.col("lag.pickup_date") == sf.date_sub(sf.col("curr.pickup_date"), 28)),
        how="left"
    ).select(
        sf.col("curr.PULocationID").alias("PULocationID"),
        sf.col("curr.pickup_date").alias("pickup_date"),
        sf.col("curr.day_of_week").alias("day_of_week"),
        sf.col("curr.hour_of_day").alias("hour_of_day"),
        sf.col("curr.trip_count").alias("current_trip_count"),
        sf.coalesce(sf.col("lag.trip_count"), sf.lit(0)).alias("lag_4w_trip_count")
    )

    result_o3 = (joined_lag
        .join(hist_stats, on=["PULocationID", "day_of_week", "hour_of_day"], how="inner")
        .withColumn("demand_index",
                    sf.when(sf.col("historical_mean") > 0,
                            sf.col("current_trip_count") / sf.col("historical_mean")).otherwise(None))
        .withColumn("high_demand_flag", sf.coalesce(sf.col("demand_index") > 2.0, sf.lit(False)))
        .join(sf.broadcast(zones.select(
                  sf.col("LocationID").cast("int").alias("PULocationID"),
                  "service_zone", "Borough", "Zone")),
              on="PULocationID", how="left"))

    print("O3 Sample Output:")
    result_o3.orderBy(sf.desc("demand_index")).show(20, truncate=False)
    return result_o3


def main():
    args = build_arg_parser().parse_args()

    master = args.master or os.environ.get("SPARK_MASTER", "spark://spark-master:7077")
    output_dir = args.output or os.path.join(args.data, "output")
    zones_path = args.zones or os.path.join(args.data, "taxi_zone_lookup.csv")

    spark = (SparkSession.builder
             .master(master)
             .appName("CompPar-Project1-Pipeline")
             .config("spark.driver.memory", "2g")
             .config("spark.executor.memory", "1536m")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())

    print(f"Connected to Spark master: {master}")
    print(f"Spark UI: {spark.sparkContext.uiWebUrl}")
    print(f"Parallelism (cores): {spark.sparkContext.defaultParallelism}")

    profiler = None
    if args.profile and SparkProfiler:
        profiler = SparkProfiler(spark)
        print("SparkProfiler initialized.")

    start_time = time.perf_counter()

    zones, pu = load_lookup(spark, zones_path)
    df_raw = load_trips(spark, args.data, sample_fraction=args.sample, target_year=args.year)

    if profiler:
        with profiler.profile("full-pipeline"):
            cleaned_df, cleaning_stats = run_m1_cleaning(df_raw)
            run_m2_temporal(cleaned_df, pu, output_dir)
            run_m3_zone_stats(cleaned_df, pu)
            run_o1_fare_anomalies(cleaned_df, pu)
            run_o3_demand_forecasting(cleaned_df, zones)
            cleaned_df.unpersist()
        print("\n--- Profiler Report ---")
        print(profiler.report())
    else:
        cleaned_df, cleaning_stats = run_m1_cleaning(df_raw)
        run_m2_temporal(cleaned_df, pu, output_dir)
        run_m3_zone_stats(cleaned_df, pu)
        run_o1_fare_anomalies(cleaned_df, pu)
        run_o3_demand_forecasting(cleaned_df, zones)
        cleaned_df.unpersist()

    elapsed = time.perf_counter() - start_time
    print(f"\nPipeline finished successfully in {elapsed:.2f} seconds.")
    spark.stop()


if __name__ == "__main__":
    main()
