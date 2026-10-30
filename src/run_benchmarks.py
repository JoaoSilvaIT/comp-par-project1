"""
run_benchmarks.py
-----------------
Automated performance engineering benchmark suite for CompPar Project 1.
Collects:
  1. M1 per-year cleaning statistics & sample outputs for M2, M3, O1, O3
  2. Extended & formatted physical execution plans (.explain("extended"))
  3. Step-by-step Optimisation Study (0-baseline -> 1-caching -> 2-pruned-scan -> 3-broadcast-join -> 4-aqe-tuned)
  4. Per-operation breakdown (M1, M2, M3, O1, O3) & stage-level task duration skew quantiles
  5. Dataset Scale study (10%, 20%, 50%, 100% of 2020 dataset)
  6. Cluster Size scaling study (called per worker count: 1, 2, 3, 4 workers + local)
"""

import argparse
import glob
import io
import json
import os
import sys
import time
import urllib.request
from contextlib import redirect_stdout
from functools import reduce

import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, to_timestamp

sys.path.append("/apps/project1")
sys.path.append("/workspace/project1")
from spark_profiler import SparkProfiler


def get_stage_skew_details(profiler, run_name):
    """Fetch per-stage task duration quantiles (min, 25%, median, 75%, max) from Spark REST API."""
    status = profiler.sc.statusTracker()
    all_stages = profiler._get_json(f"/api/v1/applications/{profiler.app_id}/stages")
    stage_details = []
    for s in all_stages:
        if s.get("status") != "COMPLETE":
            continue
        sid = s["stageId"]
        attempt = s.get("attemptId", 0)
        try:
            summary = profiler._get_json(
                f"/api/v1/applications/{profiler.app_id}/stages/{sid}/{attempt}/taskSummary?quantiles=0.0,0.25,0.5,0.75,1.0"
            )
            rt = summary.get("executorRunTime", [0, 0, 0, 0, 0])
        except Exception:
            rt = [0, 0, 0, 0, 0]
        median_ms = rt[2] if len(rt) > 2 else 0
        max_ms = rt[4] if len(rt) > 4 else 0
        skew_ratio = round(max_ms / median_ms, 2) if median_ms > 0 else 1.0
        stage_details.append({
            "stageId": sid,
            "name": s.get("name", ""),
            "numTasks": s.get("numTasks", 0),
            "executorRunTime_ms": s.get("executorRunTime", 0),
            "inputBytes": s.get("inputBytes", 0),
            "shuffleReadBytes": s.get("shuffleReadBytes", 0),
            "shuffleWriteBytes": s.get("shuffleWriteBytes", 0),
            "task_min_ms": rt[0] if len(rt) > 0 else 0,
            "task_p25_ms": rt[1] if len(rt) > 1 else 0,
            "task_median_ms": median_ms,
            "task_p75_ms": rt[3] if len(rt) > 3 else 0,
            "task_max_ms": max_ms,
            "skew_max_over_median": skew_ratio,
        })
    return stage_details


def organize_all_cols(df):
    return df.select(
        to_timestamp(col("tpep_pickup_datetime")).alias("tpep_pickup_datetime"),
        to_timestamp(col("tpep_dropoff_datetime")).alias("tpep_dropoff_datetime"),
        col("PULocationID").cast("int").alias("PULocationID"),
        col("DOLocationID").cast("int").alias("DOLocationID"),
        col("passenger_count").cast("int").alias("passenger_count"),
        col("trip_distance").cast("double").alias("trip_distance"),
        col("fare_amount").cast("double").alias("fare_amount"),
        col("tip_amount").cast("double").alias("tip_amount"),
        col("payment_type").cast("int").alias("payment_type"),
        col("total_amount").cast("double").alias("total_amount"),
    ).dropna()


def organize_pruned_cols(df):
    return df.select(
        to_timestamp(col("tpep_pickup_datetime")).alias("tpep_pickup_datetime"),
        to_timestamp(col("tpep_dropoff_datetime")).alias("tpep_dropoff_datetime"),
        col("PULocationID").cast("int").alias("PULocationID"),
        col("DOLocationID").cast("int").alias("DOLocationID"),
        col("passenger_count").cast("int").alias("passenger_count"),
        col("trip_distance").cast("double").alias("trip_distance"),
        col("fare_amount").cast("double").alias("fare_amount"),
        col("payment_type").cast("int").alias("payment_type"),
    ).dropna()


CLEAN_FILTER_SQL = """
    fare_amount > 0
    AND tpep_pickup_datetime IS NOT NULL
    AND tpep_dropoff_datetime IS NOT NULL
    AND trip_distance > 0
    AND passenger_count BETWEEN 1 AND 6
    AND tpep_dropoff_datetime > tpep_pickup_datetime
    AND PULocationID BETWEEN 1 AND 263
    AND DOLocationID BETWEEN 1 AND 263
"""


def build_queries(result1, pu, use_broadcast=False, cache_buckets=False):
    join_pu = sf.broadcast(pu) if use_broadcast else pu

    # M2
    result2 = (result1
        .join(join_pu, on="PULocationID", how="left")
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

    # M3
    w_rank = Window.partitionBy("pickup_borough", "month_idx").orderBy(sf.desc("trips"))
    w_roll = Window.partitionBy("pickup_borough", "PULocationID").orderBy("month_idx").rangeBetween(-2, 0)
    w_bor  = Window.partitionBy("pickup_borough", "month_idx")

    result3 = (result1
        .join(join_pu, on="PULocationID", how="left")
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

    # O1
    w_zone_hour = Window.partitionBy("PULocationID", "hour_of_day")

    result4 = (result1
        .join(join_pu, on="PULocationID", how="left")
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

    # O3
    w_hist = Window.partitionBy("PULocationID", "day_of_week", "hour_of_day")

    buckets = (result1
        .withColumn("pickup_date", sf.to_date("tpep_pickup_datetime"))
        .withColumn("day_of_week", sf.dayofweek("tpep_pickup_datetime"))
        .withColumn("hour_of_day", sf.hour("tpep_pickup_datetime"))
        .groupBy("PULocationID", "pickup_date", "day_of_week", "hour_of_day")
        .agg(sf.count("*").alias("trips")))

    if cache_buckets:
        buckets = buckets.cache()

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
        .join(join_pu, on="PULocationID", how="left"))

    return result2, result3, result4, result5, buckets


def execute_variant(spark, profiler, name, data_folder, use_single_scan=False,
                    use_cache=False, use_pruning=False, use_broadcast=False,
                    use_aqe=False, sample=None, record_per_op=False):
    spark.catalog.clearCache()

    target_partitions = spark.sparkContext.defaultParallelism * 2

    if use_aqe:
        spark.conf.set("spark.sql.adaptive.enabled", "true")
        spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
        spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
        spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "16MB")
        spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "16MB")
        spark.conf.set("spark.sql.shuffle.partitions", target_partitions)
    else:
        spark.conf.set("spark.sql.adaptive.enabled", "false")
        spark.conf.set("spark.sql.shuffle.partitions", 200)

    if use_broadcast:
        spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")
    else:
        spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

    per_op_times = {}

    with profiler.profile(name):
        zones = (spark.read.option("delimiter", ",").option("header", True)
                 .csv(data_folder + "taxi_zone_lookup.csv"))
        pu = zones.select(sf.col("LocationID").cast("int").alias("PULocationID"),
                          sf.col("Borough").alias("PU_Borough"),
                          sf.col("Zone").alias("PU_Zone"),
                          sf.col("service_zone").alias("PU_service_zone"))

        org_fn = organize_pruned_cols if use_pruning else organize_all_cols

        if use_single_scan:
            df_organized = org_fn(spark.read.parquet(data_folder + "yellow_tripdata_2020-*.parquet"))
        else:
            files = sorted(glob.glob(data_folder + "yellow_tripdata_2020-*.parquet"))
            df_organized = reduce(lambda a, b: a.unionByName(b),
                                  [org_fn(spark.read.parquet(f)) for f in files])

        if sample is not None and 0.0 < sample < 1.0:
            df_organized = df_organized.sample(False, sample, seed=42)

        # M1
        t_m1 = time.perf_counter()
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
        cleaning_stats.collect()

        result1 = df_organized.where(CLEAN_FILTER_SQL)
        if use_pruning:
            result1 = result1.drop("DOLocationID")
        if use_cache:
            result1 = result1.cache()
            result1.count()
        per_op_times["M1_s"] = round(time.perf_counter() - t_m1, 2)

        result2, result3, result4, result5, buckets = build_queries(
            result1, pu, use_broadcast=use_broadcast, cache_buckets=use_cache
        )

        # M2
        t_m2 = time.perf_counter()
        (result2.write
              .mode("overwrite")
              .partitionBy("pickup_borough")
              .parquet(data_folder + "output/borough_hour_stats"))
        per_op_times["M2_s"] = round(time.perf_counter() - t_m2, 2)

        # M3
        t_m3 = time.perf_counter()
        result3.limit(20).collect()
        per_op_times["M3_s"] = round(time.perf_counter() - t_m3, 2)

        # O1
        t_o1 = time.perf_counter()
        result4.orderBy("pickup_borough", "payment_type").collect()
        per_op_times["O1_s"] = round(time.perf_counter() - t_o1, 2)

        # O3
        t_o3 = time.perf_counter()
        result5.limit(20).collect()
        per_op_times["O3_s"] = round(time.perf_counter() - t_o3, 2)

        if use_cache:
            buckets.unpersist()
            result1.unpersist()

    metrics = dict(profiler.runs[name])
    metrics.update(per_op_times)
    return metrics


def capture_explain(df, mode="formatted"):
    buf = io.StringIO()
    with redirect_stdout(buf):
        df.explain(mode=mode)
    return buf.getvalue()


def run_optimisation_and_dataset_suite(spark, data_folder, results_dir):
    profiler = SparkProfiler(spark)

    # 1. Save M1 Cleaning Stats & Sample outputs + Explain plans
    print("=== Collecting M1 Cleaning Stats & Execution Plans ===")
    zones = spark.read.option("delimiter", ",").option("header", True).csv(data_folder + "taxi_zone_lookup.csv")
    pu = zones.select(sf.col("LocationID").cast("int").alias("PULocationID"),
                      sf.col("Borough").alias("PU_Borough"),
                      sf.col("Zone").alias("PU_Zone"),
                      sf.col("service_zone").alias("PU_service_zone"))

    df_full = organize_all_cols(spark.read.parquet(data_folder + "yellow_tripdata_2020-*.parquet"))
    cleaning_stats = (df_full
        .withColumn("year", sf.year("tpep_pickup_datetime"))
        .groupBy("year")
        .agg(sf.count("*").alias("total_trips"),
             sf.sum((sf.col("fare_amount") <= 0).cast("int")).alias("removed_fare"),
             sf.sum((sf.col("trip_distance") <= 0).cast("int")).alias("removed_distance"),
             sf.sum(((sf.col("passenger_count") == 0) | (sf.col("passenger_count") > 6)).cast("int")).alias("removed_passengers"),
             sf.sum((sf.col("tpep_dropoff_datetime") <= sf.col("tpep_pickup_datetime")).cast("int")).alias("removed_timestamps"),
             sf.sum((~sf.col("PULocationID").between(1, 263) | ~sf.col("DOLocationID").between(1, 263)).cast("int")).alias("removed_locations"))
        .orderBy("year"))
    cleaning_pdf = cleaning_stats.toPandas()
    cleaning_pdf.to_csv(os.path.join(results_dir, "m1_cleaning_stats.csv"), index=False)
    print(cleaning_pdf)

    # Capture Baseline vs Optimised Explain Plans
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
    spark.conf.set("spark.sql.shuffle.partitions", 200)
    res1_base = df_full.where(CLEAN_FILTER_SQL)
    r2_b, r3_b, r4_b, r5_b, _ = build_queries(res1_base, pu, use_broadcast=False, cache_buckets=False)

    plans_text = []
    for label, q in [("M2", r2_b), ("M3", r3_b), ("O1", r4_b), ("O3", r5_b)]:
        plans_text.append(f"================ BASELINE {label} (FORMATTED) ================\n" + capture_explain(q, "formatted"))
        plans_text.append(f"================ BASELINE {label} (EXTENDED) ================\n" + capture_explain(q, "extended"))

    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")
    spark.conf.set("spark.sql.shuffle.partitions", spark.sparkContext.defaultParallelism * 2)
    res1_opt = organize_pruned_cols(spark.read.parquet(data_folder + "yellow_tripdata_2020-*.parquet")).where(CLEAN_FILTER_SQL).drop("DOLocationID")
    r2_o, r3_o, r4_o, r5_o, _ = build_queries(res1_opt, pu, use_broadcast=True, cache_buckets=True)

    for label, q in [("M2", r2_o), ("M3", r3_o), ("O1", r4_o), ("O3", r5_o)]:
        plans_text.append(f"================ OPTIMISED {label} (FORMATTED) ================\n" + capture_explain(q, "formatted"))
        plans_text.append(f"================ OPTIMISED {label} (EXTENDED) ================\n" + capture_explain(q, "extended"))

    with open(os.path.join(results_dir, "explain_plans.txt"), "w") as f:
        f.write("\n".join(plans_text))

    # 2. Step-by-step Optimisation Study (on full 2020 dataset)
    print("\n=== Running Step-by-Step Optimisation Study (Full 2020 Dataset) ===")
    opt_configs = [
        ("0-baseline",      dict(use_single_scan=False, use_cache=False, use_pruning=False, use_broadcast=False, use_aqe=False)),
        ("1-caching",       dict(use_single_scan=False, use_cache=True,  use_pruning=False, use_broadcast=False, use_aqe=False)),
        ("2-pruned-scan",   dict(use_single_scan=True,  use_cache=True,  use_pruning=True,  use_broadcast=False, use_aqe=False)),
        ("3-broadcast-join",dict(use_single_scan=True,  use_cache=True,  use_pruning=True,  use_broadcast=True,  use_aqe=False)),
        ("4-aqe-tuned",     dict(use_single_scan=True,  use_cache=True,  use_pruning=True,  use_broadcast=True,  use_aqe=True)),
    ]

    opt_rows = []
    for name, kwargs in opt_configs:
        print(f"\n>>> Running {name} ...")
        m = execute_variant(spark, profiler, name, data_folder, **kwargs)
        m["run"] = name
        opt_rows.append(m)

    opt_df = pd.DataFrame(opt_rows)
    opt_df.to_csv(os.path.join(results_dir, "optimisation_study_raw.csv"), index=False)
    profiler.report().to_csv(os.path.join(results_dir, "optimisation_study_formatted.csv"))
    print("\n--- Optimisation Study Summary ---")
    print(profiler.report())
    print("\n--- Baseline vs Final Comparison ---")
    comp_df = profiler.compare("0-baseline", "4-aqe-tuned")
    comp_df.to_csv(os.path.join(results_dir, "baseline_vs_final.csv"))
    print(comp_df)

    # Save stage-level skew details
    stage_skew = get_stage_skew_details(profiler, "all")
    with open(os.path.join(results_dir, "stage_skew_details.json"), "w") as f:
        json.dump(stage_skew, f, indent=2)

    # 3. Dataset Scale Study (10%, 20%, 50%, 100%)
    print("\n=== Running Dataset Scale Study (10%, 20%, 50%, 100%) ===")
    scale_rows = []
    for frac in [0.1, 0.2, 0.5, 1.0]:
        run_name = f"scale-{int(frac*100)}pct"
        print(f"\n>>> Running {run_name} (sample={frac}) ...")
        m = execute_variant(
            spark, profiler, run_name, data_folder,
            use_single_scan=True, use_cache=True, use_pruning=True,
            use_broadcast=True, use_aqe=True, sample=frac
        )
        m["fraction"] = frac
        m["run"] = run_name
        scale_rows.append(m)

    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(os.path.join(results_dir, "dataset_scaling.csv"), index=False)
    print("\n--- Dataset Scaling Results ---")
    print(scale_df[["run", "fraction", "wall_time_s", "num_tasks", "input_bytes", "shuffle_read_bytes", "shuffle_write_bytes", "M1_s", "M2_s", "M3_s", "O1_s", "O3_s"]])


def run_single_cluster_point(spark, data_folder, results_dir, workers_label, num_workers):
    cores = spark.sparkContext.defaultParallelism
    warmup_profiler = SparkProfiler(spark)
    # Warm up JVM JIT, Catalyst codegen, and executor connections with a tiny 1% sample
    execute_variant(
        spark, warmup_profiler, "warmup", data_folder,
        use_single_scan=True, use_cache=True, use_pruning=True,
        use_broadcast=True, use_aqe=True, sample=0.01
    )

    profiler = SparkProfiler(spark)

    # Run tuned pipeline
    run_name = f"cluster-{workers_label}-tuned"
    print(f"\n>>> Running {run_name} (workers={num_workers}, cores={cores}) ...")
    m_tuned = execute_variant(
        spark, profiler, run_name, data_folder,
        use_single_scan=True, use_cache=True, use_pruning=True,
        use_broadcast=True, use_aqe=True, sample=None
    )
    m_tuned["workers_label"] = workers_label
    m_tuned["num_workers"] = num_workers
    m_tuned["cores"] = cores
    m_tuned["pipeline"] = "tuned"

    out_path = os.path.join(results_dir, "cluster_scaling.csv")
    df_row = pd.DataFrame([m_tuned])
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        existing = existing[~((existing["workers_label"] == workers_label) & (existing["pipeline"] == "tuned"))]
        df_row = pd.concat([existing, df_row], ignore_index=True)
    df_row.to_csv(out_path, index=False)
    print(df_row[["workers_label", "num_workers", "cores", "wall_time_s", "num_tasks", "executor_run_time_ms", "M1_s", "M2_s", "M3_s", "O1_s", "O3_s"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["opt_and_scale", "cluster_point"], required=True)
    parser.add_argument("--master", default="spark://spark-master:7077")
    parser.add_argument("--data", default="/data/project1/")
    parser.add_argument("--results", default="/data/project1/results")
    parser.add_argument("--workers-label", default="3w")
    parser.add_argument("--num-workers", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.results, exist_ok=True)

    spark_builder = (SparkSession.builder
                     .master(args.master)
                     .appName(f"CompPar-Bench-{args.mode}-{args.workers_label}")
                     .config("spark.driver.memory", "2g")
                     .config("spark.executor.memory", "1536m")
                     .config("spark.hadoop.fs.permissions.umask-mode", "000")
                     .config("spark.ui.showConsoleProgress", "false"))

    if args.master.startswith("spark://"):
        spark_builder = (spark_builder
                         .config("spark.driver.host", "jupyter")
                         .config("spark.driver.bindAddress", "0.0.0.0"))

    spark = spark_builder.getOrCreate()
    print(f"Connected to {args.master} | Parallelism: {spark.sparkContext.defaultParallelism}")

    if args.mode == "opt_and_scale":
        run_optimisation_and_dataset_suite(spark, args.data, args.results)
    elif args.mode == "cluster_point":
        run_single_cluster_point(spark, args.data, args.results, args.workers_label, args.num_workers)

    spark.stop()


if __name__ == "__main__":
    main()
