"""
run_pipeline.py
---------------
Standalone, spark-submit-ready version of the pipeline from Taxi_Trips.ipynb
(M1, M2, M3, O1, and O3).

Usage (from inside the spark-master container):

    spark-submit --master spark://spark-master:7077 \\
        /apps/project1/run_pipeline.py \\
        --data /data/project1/ --partitions 12

Or locally with a sample / single year:

    spark-submit src/run_pipeline.py \\
        --data data/ --year 2019 --sample 0.1 --partitions 8
"""

import argparse
import glob
from functools import reduce
import time

import pyspark.sql.functions as sf
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, to_timestamp


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/data/project1/",
                        help="Folder containing yellow_tripdata_*.parquet and taxi_zone_lookup.csv")
    parser.add_argument("--year", default="*",
                        help="Year to process (e.g. 2019, or '*' for all years)")
    parser.add_argument("--sample", type=float, default=None,
                        help="Fraction of records to sample for smoke-testing (e.g. 0.1)")
    parser.add_argument("--partitions", type=int, default=48,
                        help="Initial shuffle partition count to aid AQE (default: 48)")
    return parser


def organize(df):
    return df.select(
        to_timestamp(col("tpep_pickup_datetime")).alias("tpep_pickup_datetime"),
        to_timestamp(col("tpep_dropoff_datetime")).alias("tpep_dropoff_datetime"),
        col("PULocationID").cast("int").alias("PULocationID"),
        col("DOLocationID").cast("int").alias("DOLocationID"),
        col("passenger_count").cast("int"),
        col("trip_distance").cast("double"),
        col("fare_amount").cast("double"),
        col("tip_amount").cast("double"),
        col("payment_type").cast("int"),
        col("total_amount").cast("double")
    ).dropna()


def run(spark, data_folder, target_partitions, year="*", sample=None):
    if not data_folder.endswith("/"):
        data_folder += "/"

    # Adaptive Query Execution (AQE) configuration
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "16MB")
    spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "16MB")
    spark.conf.set("spark.sql.shuffle.partitions", target_partitions)

    zones = (spark.read.option("delimiter", ",").option("header", True)
             .csv(data_folder + "taxi_zone_lookup.csv"))

    pu = zones.select(sf.col("LocationID").cast("int").alias("PULocationID"),
                      sf.col("Borough").alias("PU_Borough"),
                      sf.col("Zone").alias("PU_Zone"),
                      sf.col("service_zone").alias("PU_service_zone"))

    files = sorted(glob.glob(f"{data_folder}yellow_tripdata_{year}-*.parquet"))
    df_organized = organize(spark.read.parquet(*files))

    if sample is not None:
        df_organized = df_organized.sample(False, sample)

    # M1 - Data ingestion and cleaning
    cleaning_stats = (df_organized
        .withColumn("year", sf.year("tpep_pickup_datetime"))
        .groupBy("year")
        .agg(sf.count("*").alias("total_trips"),
             sf.sum((sf.col("fare_amount") <= 0).cast("int")).alias("removed_fare"),
             sf.sum((sf.col("trip_distance") <= 0).cast("int")).alias("removed_distance"),
             sf.sum(((sf.col("passenger_count") == 0) | (sf.col("passenger_count") > 6)).cast("int")).alias("removed_passengers"),
             sf.sum((sf.col("tpep_dropoff_datetime") <= sf.col("tpep_pickup_datetime")).cast("int")).alias("removed_timestamps"),
             sf.sum((~sf.col("PULocationID").between(1, 263) | ~sf.col("DOLocationID").between(1, 263)).cast("int")).alias("removed_locations"))
        .orderBy("year"))
    cleaning_stats.show()

    result1 = (df_organized.where("""
        fare_amount > 0
        AND tpep_pickup_datetime IS NOT NULL
        AND tpep_dropoff_datetime IS NOT NULL
        AND trip_distance > 0
        AND passenger_count BETWEEN 1 AND 6
        AND tpep_dropoff_datetime > tpep_pickup_datetime
        AND PULocationID BETWEEN 1 AND 263
        AND DOLocationID BETWEEN 1 AND 263
    """).cache())
    result1.count()  # materialize the cache before it's used below

    # M2 - Temporal demand analysis
    result2 = (result1
        .join(sf.broadcast(pu), on="PULocationID", how="left")
        .withColumn("day_of_week", sf.dayofweek("tpep_pickup_datetime"))
        .withColumn("hour_of_day", sf.hour("tpep_pickup_datetime"))
        .withColumn("duration_min",
                    (sf.col("tpep_dropoff_datetime").cast("long")
                     - sf.col("tpep_pickup_datetime").cast("long")) / 60)
        .groupBy(sf.col("PU_Borough").alias("pickup_borough"), "day_of_week", "hour_of_day")
        .agg(sf.count("*").alias("trip_count"),
             sf.avg("fare_amount").alias("mean_fare_amount"),
             sf.avg("duration_min").alias("mean_duration_min"),
             sf.avg("trip_distance").alias("mean_trip_distance")))

    (result2.write
          .mode("overwrite")
          .partitionBy("pickup_borough")
          .parquet(data_folder + "output/borough_hour_stats"))

    # M3 - Zone-level statistics
    w_rank = Window.partitionBy("pickup_borough", "month_idx").orderBy(sf.desc("trips"))
    w_roll = Window.partitionBy("pickup_borough", "PULocationID").orderBy("month_idx").rangeBetween(-2, 0)
    w_bor  = Window.partitionBy("pickup_borough", "month_idx")

    result3 = (result1
        .join(sf.broadcast(pu), on="PULocationID", how="left")
        .withColumn("month", sf.date_trunc("month", "tpep_pickup_datetime"))
        .withColumn("month_idx", sf.year("month") * 12 + sf.month("month"))
        .groupBy(sf.col("PU_Borough").alias("pickup_borough"),
                 "PULocationID", "PU_Zone", "month_idx", "month")
        .agg(sf.count("*").alias("trips"),
             sf.avg("fare_amount").alias("mean_fare_amount"),
             sf.stddev("fare_amount").alias("std_fare_amount"),
             sf.avg("trip_distance").alias("mean_trip_distance"),
             sf.stddev("trip_distance").alias("std_trip_distance"))
        .withColumn("rank_in_borough", sf.rank().over(w_rank))
        .withColumn("roll3_trips", sf.sum("trips").over(w_roll))
        .withColumn("roll3_share", sf.col("roll3_trips") / sf.sum("roll3_trips").over(w_bor))
        .withColumn("roll_mean", sf.avg("trips").over(w_roll))
        .withColumn("roll_std", sf.stddev("trips").over(w_roll))
        .withColumn("anomaly",
                    sf.coalesce(sf.abs(sf.col("trips") - sf.col("roll_mean")) > 1.5 * sf.col("roll_std"),
                                sf.lit(False))))
    result3.show()

    # O1 - Fare anomaly detection
    w_zone_hour = Window.partitionBy("PULocationID", "hour_of_day")

    result4 = (result1
        .join(sf.broadcast(pu), on="PULocationID", how="left")
        .withColumn("hour_of_day", sf.hour("tpep_pickup_datetime"))
        .withColumn("mean_fare_amount",
                    sf.avg(sf.col("fare_amount") / sf.col("trip_distance")).over(w_zone_hour))
        .withColumn("sigma_fare_amount",
                    sf.stddev(sf.col("fare_amount") / sf.col("trip_distance")).over(w_zone_hour))
        .withColumn("anomaly",
                    sf.coalesce((sf.col("fare_amount") / sf.col("trip_distance"))
                                > sf.col("mean_fare_amount") + 3 * sf.col("sigma_fare_amount"),
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
    result4.orderBy("pickup_borough", "payment_type").show(50)

    # O3 - Demand forecasting features
    w_hist = Window.partitionBy("PULocationID", "day_of_week", "hour_of_day")

    buckets = (result1
        .withColumn("pickup_date", sf.to_date("tpep_pickup_datetime"))
        .withColumn("day_of_week", sf.dayofweek("tpep_pickup_datetime"))
        .withColumn("hour_of_day", sf.hour("tpep_pickup_datetime"))
        .groupBy("PULocationID", "pickup_date", "day_of_week", "hour_of_day")
        .agg(sf.count("*").alias("trips"))
        .cache())

    lag_buckets = buckets.select(
        "PULocationID",
        sf.date_add("pickup_date", 28).alias("pickup_date"),
        "hour_of_day",
        sf.col("trips").alias("lag_4w_trips")
    )

    result5 = (buckets
        .join(lag_buckets, on=["PULocationID", "pickup_date", "hour_of_day"], how="left")
        .withColumn("mean_trips", sf.avg("trips").over(w_hist))
        .withColumn("var_trips", sf.variance("trips").over(w_hist))
        .withColumn("demand_index", sf.col("trips") / sf.col("mean_trips"))
        .withColumn("anomaly",
                    sf.coalesce(sf.col("demand_index") > 2.0, sf.lit(False)))
        .join(sf.broadcast(pu), on="PULocationID", how="left"))
    result5.show()

    buckets.unpersist()
    result1.unpersist()


def main():
    args = build_arg_parser().parse_args()

    spark = (SparkSession.builder
             .appName("spark-profiling-taxi-pipeline")
             .config("spark.driver.memory", "2g")
             .config("spark.executor.memory", "1536m")   # stay under the worker's --memory 2G cap
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())

    target_partitions = args.partitions or spark.sparkContext.defaultParallelism * 2

    print(f"Spark UI: {spark.sparkContext.uiWebUrl}")
    print(f"Default parallelism: {spark.sparkContext.defaultParallelism}")
    print(f"Target partitions: {target_partitions}")

    start = time.perf_counter()
    run(spark, args.data, target_partitions, args.year, args.sample)
    elapsed = time.perf_counter() - start
    print(f"Wall time: {elapsed:.2f}s")

    spark.stop()


if __name__ == "__main__":
    main()
