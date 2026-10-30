# CompPar 2026/27 — Project Assignment 1: Large-Scale Analytics with Spark

End-to-end PySpark analytical pipeline and performance engineering study over the **NYC Taxi & Limousine Commission (TLC) Yellow Taxi Trip Records**.

---

## 1. Operations Implemented

- **M1 — Data Ingestion & Cleaning:** Loads monthly Parquet trip records, enforces the schema (explicit type casting and null removal via `.dropna()`), computes per-year removal statistics across all 5 validity criteria (`fare_amount <= 0`, `trip_distance <= 0`, `passenger_count = 0 or > 6`, `dropoff <= pickup`, and `PULocationID / DOLocationID` outside `1–263`), and caches the cleaned dataset (`result1`).
- **M2 — Temporal Demand Analysis:** Joins pickup zones with `taxi_zone_lookup.csv` (`pu`) and aggregates total trip count, mean fare amount, mean duration in minutes, and mean trip distance per `(pickup_borough, day_of_week, hour_of_day)`, writing the output partitioned by `pickup_borough` in Parquet format (`data/output/borough_hour_stats`).
- **M3 — Zone-Level Statistics:** Computes monthly trip volume, mean/std fare amount, and mean/std trip distance per pickup zone, then applies window functions (`w_rank`, `w_roll`, `w_bor`) to rank zones within each borough, compute 3-month rolling trip shares, and flag volume anomalies ($> 1.5\sigma$ from the 3-month rolling mean).
- **O1 — Fare Anomaly Detection:** Computes the mean and standard deviation ($\sigma$) of unit fare (`fare_amount / trip_distance`) per `(PULocationID, hour_of_day)` window, flags trips exceeding $\text{mean} + 3\sigma$, and pivots weekday vs. weekend anomaly rates by `pickup_borough` and `payment_type`.
- **O3 — Demand Forecasting Features:** Aggregates trip counts per `(PULocationID, pickup_date, day_of_week, hour_of_day)`, performs a self-join on a 28-day (4-week) date offset (`sf.date_add("pickup_date", 28)`), computes historical mean and variance over `(PULocationID, day_of_week, hour_of_day)` windows, flags buckets with `demand_index > 2.0`, and annotates results with `service_zone` via broadcast join.

---

## 2. Dataset & Download Instructions

The dataset is not committed to the repository (per assignment rules). It is downloaded directly from the official NYC TLC CloudFront distribution:
- **Trip Records Base URL:** `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`
- **Zone Lookup URL:** `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv`
- **Primary Evaluation Dataset:** Full year **2020** (`yellow_tripdata_2020-01.parquet` through `yellow_tripdata_2020-12.parquet`, 12 files, `358 MB` in Snappy Parquet, `24,649,092` raw records / `23,838,845` non-null records).

To download the 2020 dataset and zone lookup table into `data/`:
```bash
curl -fL "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv" -o data/taxi_zone_lookup.csv
for m in $(seq -w 1 12); do
    curl -fL "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-${m}.parquet" \
         -o "data/yellow_tripdata_2020-${m}.parquet"
done
```

---

## 3. Environment & Cluster Topology

All experiments were conducted using the course `compar-spark-docker` cluster environment:
- **Apache Spark Version:** `4.2.0` (PySpark, OpenJDK 21)
- **Host Hardware:** Linux x86_64, 16 logical CPU cores, 23 GiB RAM
- **Docker Cluster Topology (`docker-spark-env/docker-compose.yml`):**
  - `1 × spark-master` (`spark://spark-master:7077`, Web UI on `http://localhost:8080`)
  - `N × spark-worker` (`--cores 2 --memory 2G` per worker container; scaled across `N = 1, 2, 3, 4, 6` workers, i.e., `2` to `12` cluster cores)
  - `1 × jupyter` driver container (`spark.driver.memory = 2g`, `spark.executor.memory = 1536m`, Spark Application UI on `http://localhost:4040`)

Start the cluster with 3 workers (default, 6 cores):
```bash
cd ../labs/compar-spark-docker/docker-spark-env
docker compose up -d --scale spark-worker=3
```

---

## 4. Non-Default Spark Configuration Properties

```python
spark.driver.memory = "2g"
spark.executor.memory = "1536m"
spark.sql.adaptive.enabled = "true"
spark.sql.adaptive.coalescePartitions.enabled = "true"
spark.sql.adaptive.skewJoin.enabled = "true"
spark.sql.adaptive.localShuffleReader.enabled = "true"
spark.sql.adaptive.advisoryPartitionSizeInBytes = "16MB"
spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes = "16MB"
spark.sql.shuffle.partitions = 48
```

---

## 5. Reproducing the Pipeline & Benchmarks

### A. Run Standalone Pipeline (`spark-submit`)
```bash
# Full 2020 dataset on the Docker Spark cluster
docker exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    /apps/project1/run_pipeline.py \
    --data /data/project1/ --year 2020 --partitions 48

# Smoke test with 10% sample
docker exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    /apps/project1/run_pipeline.py \
    --data /data/project1/ --year 2020 --sample 0.1 --partitions 48
```

### B. Reproduce Full Benchmark Suite & Figures
```bash
# 1. Optimisation study (0-baseline -> 4-aqe-tuned) + Dataset scaling (10%, 20%, 50%, 100%)
docker exec --user 185 jupyter python3 /apps/project1/run_benchmarks.py --mode opt_and_scale

# 2. Cluster size scaling (1w, 2w, 3w, 4w, 6w + local)
docker exec --user 185 jupyter python3 /apps/project1/run_benchmarks.py \
    --mode cluster_point --workers-label 3w --num-workers 3

# 3. Generate PDF/PNG charts for the report
docker exec jupyter python3 /apps/project1/generate_plots.py
```

All generated CSV tables, JSON stage profiles, physical execution plans, and PDF/PNG figures are stored in `report/results/` and `report/figures/`.
