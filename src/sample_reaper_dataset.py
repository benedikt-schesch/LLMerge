# -*- coding: utf-8 -*-
"""Sample script to filter top N repositories by stars."""

import argparse
import os
import pandas as pd


def filter_top_n_repos(input_path, n):
    """Filter top N repositories by stars."""
    # Read the CSV file
    df = pd.read_csv(input_path)

    # Ensure 'stars' column is numeric
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")

    # Drop rows with NaN values in 'stars'
    df = df.dropna(subset=["stars"])

    # Sort by 'stars' in descending order
    df_sorted = df.sort_values(by="stars", ascending=False)

    # Select top n rows
    df_top_n = df_sorted.head(n)

    # Output file name
    output_path = f"input_data/repos_reaper_{n}.csv"

    # Save to new CSV file
    df_top_n.to_csv(output_path, index=False)
    print(f"Top {n} repositories saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter top N repositories by stars.")
    parser.add_argument(
        "--n", type=int, default=100, help="Number of top repositories to select."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="input_data/repos_reaper.csv",
        help="Path to the input CSV file.",
    )
    args = parser.parse_args()

    if os.path.exists(args.input_path):
        filter_top_n_repos(args.input_path, args.n)
    else:
        print(f"Error: The file {args.input_path} does not exist.")
