"""
generate_plots.py
-----------------
Generates publication-ready PDF and PNG figures from the benchmark CSVs in
/data/project1/results/ for the LaTeX report.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/data/project1/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
})


def plot_optimisation_study():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "optimisation_study_raw.csv"))
    labels = ["0-Baseline", "1-Caching", "2-Pruned\nScan", "3-Broadcast\nJoin", "4-AQE\nTuned"]
    x = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # Left panel: Stacked per-operation wall time + total wall time
    ops = ["M1_s", "M2_s", "M3_s", "O1_s", "O3_s"]
    op_names = ["M1 (Clean/Cache)", "M2 (Temporal)", "M3 (Zone Window)", "O1 (Fare Anomaly)", "O3 (Lag Self-Join)"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    bottom = np.zeros(len(df))
    for op, name, color in zip(ops, op_names, colors):
        vals = df[op].values
        ax1.bar(x, vals, bottom=bottom, width=0.55, label=name, color=color, edgecolor="white", linewidth=0.5)
        bottom += vals

    for i, total in enumerate(df["wall_time_s"].values):
        ax1.text(i, bottom[i] + 1.5, f"{total:.1f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Wall-Clock Time (s)")
    ax1.set_title("(a) End-to-End Execution Time by Operation")
    ax1.set_ylim(0, max(bottom) * 1.16)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", frameon=True)

    # Right panel: Shuffle Read MB and Number of Tasks
    shuffle_mb = df["shuffle_read_bytes"].values / (1024 * 1024)
    tasks = df["num_tasks"].values

    color_shuf = "#C44E52"
    color_task = "#2B5C8F"
    w = 0.35
    bars1 = ax2.bar(x - w/2, shuffle_mb, width=w, color=color_shuf, label="Shuffle Read (MB)")
    ax2.set_ylabel("Shuffle Read Volume (MB)", color=color_shuf)
    ax2.tick_params(axis="y", labelcolor=color_shuf)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(shuffle_mb) * 1.2)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    ax2_t = ax2.twinx()
    line1 = ax2_t.plot(x + w/2, tasks, color=color_task, marker="o", linewidth=2, label="Total Tasks")
    ax2_t.set_ylabel("Number of Tasks", color=color_task)
    ax2_t.tick_params(axis="y", labelcolor=color_task)
    ax2_t.set_ylim(0, max(tasks) * 1.18)

    for i, t_val in enumerate(tasks):
        ax2_t.annotate(f"{int(t_val)}", (x[i] + w/2, t_val), textcoords="offset points",
                       xytext=(0, 6), ha="center", fontsize=8, color=color_task, fontweight="bold")

    ax2.set_title("(b) Shuffle Volume & Task Count Reduction")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"fig1_optimisation_study.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_scalability():
    scale_df = pd.read_csv(os.path.join(RESULTS_DIR, "dataset_scaling.csv")).sort_values("fraction")
    clust_df = pd.read_csv(os.path.join(RESULTS_DIR, "cluster_scaling.csv"))
    clust_w = clust_df[clust_df["workers_label"].isin(["1w", "2w", "3w", "4w", "6w"])].sort_values("num_workers")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # Left panel: Dataset Scale (10%, 20%, 50%, 100%)
    fracs = scale_df["fraction"].values * 100
    times = scale_df["wall_time_s"].values
    o1_times = scale_df["O1_s"].values

    # Linear extrapolation from 10% sample vs Gustafson-Barsis linear fit
    linear_from_10 = times[0] * (fracs / fracs[0])
    slope, intercept = np.polyfit(scale_df["fraction"].values, times, 1)
    fit_line = intercept + slope * (fracs / 100.0)

    ax1.plot(fracs, times, marker="o", linewidth=2.2, color="#2B5C8F", label="Measured Total Wall Time")
    ax1.plot(fracs, o1_times, marker="s", linewidth=1.8, linestyle="-.", color="#55A868", label="Measured O1 (Window Stage)")
    ax1.plot(fracs, fit_line, linestyle=":", color="#444444", linewidth=1.5,
             label=f"Affine Fit ($T = {intercept:.1f} + {slope:.1f}s$)")
    ax1.plot(fracs, linear_from_10, linestyle="--", color="#C44E52", alpha=0.75,
             label="Naïve $10\\times$ Linear Projection from 10%")

    for i, (f_val, t_val) in enumerate(zip(fracs, times)):
        offset = (6, -14) if i < 2 else (0, 8)
        ax1.annotate(f"{t_val:.1f}s", (f_val, t_val), textcoords="offset points",
                     xytext=offset, ha="center", fontsize=8.5, fontweight="bold", color="#2B5C8F")

    ax1.set_xlabel("Dataset Scale (% of 2020 NYC Taxi Dataset)")
    ax1.set_ylabel("Wall-Clock Time (s)")
    ax1.set_xticks([10, 20, 50, 100])
    ax1.set_xlim(2, 108)
    ax1.set_ylim(0, max(times) * 1.42)
    ax1.set_title("(a) Dataset Scale Analysis (Sub-Linear Scaling)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", frameon=True)

    # Right panel: Cluster Size Speedup & Efficiency (1w, 2w, 3w, 4w)
    workers = clust_w["num_workers"].values
    cores = clust_w["cores"].values
    t_total = clust_w["wall_time_s"].values
    t_o1 = clust_w["O1_s"].values

    speedup_total = t_total[0] / t_total
    speedup_o1 = t_o1[0] / t_o1
    ideal_speedup = workers / workers[0]

    ax2.plot(workers, ideal_speedup, linestyle="--", color="#888888", linewidth=1.5, label="Ideal Linear Speedup ($S(p) = p$)")
    ax2.plot(workers, speedup_total, marker="o", linewidth=2.2, color="#2B5C8F", label="Full Pipeline Speedup $S(p)$")
    ax2.plot(workers, speedup_o1, marker="s", linewidth=2.0, color="#C44E52", label="O1 Window Stage Speedup $S(p)$")

    for w_val, s_val, c_val in zip(workers, speedup_total, cores):
        eff = (s_val / w_val) * 100.0
        if w_val == 1:
            xy_off = (18, -18)
        elif w_val == 2:
            xy_off = (-8, 8)
        else:
            xy_off = (0, -25)
        ax2.annotate(f"{s_val:.2f}x\n({eff:.0f}% eff)", (w_val, s_val),
                     textcoords="offset points", xytext=xy_off,
                     ha="center", fontsize=8, color="#2B5C8F", fontweight="bold")

    ax2.set_xlabel("Cluster Size: Number of Spark Workers (2 Cores / Worker)")
    ax2.set_ylabel("Speedup $S(p) = T_1 / T_p$")
    ax2.set_xticks(workers)
    ax2.set_xticklabels([f"{w}w\n({c} cores)" for w, c in zip(workers, cores)])
    ax2.set_xlim(0.65, 6.55)
    ax2.set_ylim(0.35, 6.4)
    ax2.set_title("(b) Cluster Strong Scaling: Speedup & Efficiency")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"fig2_scalability_analysis.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_execution_plan_diagram():
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.axis("off")

    # Draw two clean vertical pipelines: Baseline (Left) vs Optimised (Right)
    box_Base = dict(boxstyle="round,pad=0.3", facecolor="#FDECEC", edgecolor="#C44E52", linewidth=1.2)
    box_Opt  = dict(boxstyle="round,pad=0.3", facecolor="#EAF4EC", edgecolor="#55A868", linewidth=1.2)
    box_Neut = dict(boxstyle="round,pad=0.3", facecolor="#EEF3F8", edgecolor="#4C72B0", linewidth=1.1)

    ax.text(0.24, 0.94, "Baseline Plan (M2)", ha="center", va="center", fontweight="bold", fontsize=9, color="#902020")
    ax.text(0.76, 0.94, "Optimised Plan (M2)", ha="center", va="center", fontweight="bold", fontsize=9, color="#206030")

    base_steps = [
        (0.76, "Scan Parquet (10 cols) + CSV", box_Neut),
        (0.54, "Shuffle Both Sides (200 parts)\n+ Full Sort (22.7M rows)", box_Base),
        (0.30, "SortMergeJoin (LeftOuter)", box_Base),
        (0.08, "HashAggregate (200 parts)", box_Neut),
    ]
    opt_steps = [
        (0.76, "InMemoryScan (7 cols, cached)", box_Opt),
        (0.54, "Broadcast Lookup CSV (265 rows)\n(Zero Shuffle on Trips)", box_Opt),
        (0.30, "BroadcastHashJoin (LeftOuter)", box_Opt),
        (0.08, "AQE HashAggregate (Coalesced)", box_Opt),
    ]

    for y, text, style in base_steps:
        ax.text(0.24, y, text, ha="center", va="center", fontsize=7.5, bbox=style)
    for y, text, style in opt_steps:
        ax.text(0.76, y, text, ha="center", va="center", fontsize=7.5, bbox=style)

    arrow_kw = dict(arrowstyle="->", color="#555555", lw=1.1)
    for y1, y2 in [(0.68, 0.63), (0.44, 0.37), (0.22, 0.15)]:
        ax.annotate("", xy=(0.24, y2), xytext=(0.24, y1), arrowprops=arrow_kw)
        ax.annotate("", xy=(0.76, y2), xytext=(0.76, y1), arrowprops=arrow_kw)

    fig.tight_layout(pad=0.1)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"fig0_execution_plan.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_execution_plan_diagram()
    plot_optimisation_study()
    plot_scalability()
    print("Plots generated successfully in", RESULTS_DIR)
