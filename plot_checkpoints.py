# -*- coding: utf-8 -*-
"""This script generates a combined plot of training metrics across checkpoints."""

import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = (
    "eval_outputs/repos_reaper_test/test/unsloth/DeepSeek-R1-Distill-Qwen-14B-outputs"
)
METRICS = [
    "valid thinking format",
    "valid Java markdown format",
    "correctly raising merge conflict",
    "semantically correctly resolved merges",
    "correctly resolved merges",
]
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
]  # Distinct colors for each metric
CHECKPOINT_PATTERN = r"checkpoint-(\d+)"


def extract_checkpoint_number(checkpoint_dir):
    """Extract numerical checkpoint value from directory name"""
    match = re.search(CHECKPOINT_PATTERN, checkpoint_dir)
    return int(match.group(1)) if match else -1


def parse_log_file(filepath):
    """Parse eval.log file and return metric percentages"""
    metrics = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if "SUCCESS" in line and any(metric in line for metric in METRICS):
                    for metric in METRICS:
                        if metric in line:
                            value_str = line.split(":")[-1].strip().rstrip("%")
                            metrics[metric] = float(value_str)
                            break
    except FileNotFoundError:
        print(f"Warning: File not found {filepath}")
    return metrics


def collect_data(base_dir):
    """Collect metrics from all checkpoints"""
    data = defaultdict(dict)

    checkpoints = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
    checkpoints.sort(key=extract_checkpoint_number)

    for checkpoint in checkpoints:
        cp_num = extract_checkpoint_number(checkpoint)
        log_path = os.path.join(base_dir, checkpoint, "eval.log")
        metrics = parse_log_file(log_path)

        if metrics:
            data[cp_num] = metrics
        else:
            print(f"Warning: No valid metrics found in {checkpoint}")

    return data


def plot_metrics(data):
    """Plot all metrics in a single plot"""
    if not data:
        print("No data to plot")
        return

    checkpoints = sorted(data.keys())
    metric_values = {metric: [] for metric in METRICS}

    for cp in checkpoints:
        for metric in METRICS:
            metric_values[metric].append(data[cp].get(metric, 0))

    plt.figure(figsize=(12, 6))

    # Plot all metrics
    for idx, metric in enumerate(METRICS):
        plt.plot(
            checkpoints,
            metric_values[metric],
            marker="o",
            linestyle="-",
            color=COLORS[idx % len(COLORS)],
            linewidth=2,
            markersize=6,
            label=metric,
        )

    plt.axhline(y=53.60, color="r", linestyle="-", label="R1 Semantic Correctness")
    plt.axhline(y=45.66, color="b", linestyle="-", label="R1 Exact Correct")
    plt.title("Training Metrics Across Checkpoints", fontsize=14)
    plt.xlabel("Checkpoint Number", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(checkpoints, rotation=45)
    plt.tight_layout()

    # Save as high-resolution PNG
    plt.savefig("all_metrics_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    metrics_data = collect_data(BASE_DIR)
    plot_metrics(metrics_data)
    print("Combined plot generated as 'all_metrics_plot.png'")
