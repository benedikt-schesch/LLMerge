#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CreateConflictMetrics.py

Usage:
    python CreateConflictMetrics.py \
        --input_dir /path/to/output_folder \
        --csv_out /path/to/output_metrics.csv

What does this script do?
  - Finds *.conflict and *.resolved_conflict files in `--input_dir` (pairs).
  - For each pair <basename><n>.conflict / <basename><n>.resolved_conflict:
      1) Reads the conflict snippet, which includes context + conflict markers.
      2) Identifies:
           - context_before (lines above <<<<<<<),
           - conflict_block (lines from <<<<<<< up through >>>>>>>),
           - context_after (lines after >>>>>>>).
      3) Reads the resolved snippet (the entire file).
      4) Computes the sizes (in lines) of each portion.
      5) Appends a row to a CSV with these metrics.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple


def split_single_conflict_snippet(
    lines: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Given a list of lines that contain exactly one conflict block
    (which should have <<<<<<< ... >>>>>>>),
    returns (before_context, conflict_block, after_context).

    If markers are not found, returns ([], [], []).
    """
    start_idx = -1
    end_idx = -1

    # Locate <<<<<<<
    for i, line in enumerate(lines):
        if line.startswith("<<<<<<<"):
            start_idx = i
            break

    if start_idx == -1:
        # No conflict marker
        return ([], [], [])

    # Locate >>>>>>>
    for j in range(start_idx, len(lines)):
        if lines[j].startswith(">>>>>>>"):
            end_idx = j
            break

    if end_idx == -1:
        # No matching end marker
        return ([], [], [])

    before_context = lines[:start_idx]
    conflict_block = lines[start_idx : end_idx + 1]  # inclusive of end_idx
    after_context = lines[end_idx + 1 :]

    return (before_context, conflict_block, after_context)


def main():  # pylint: disable=too-many-locals
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compute metrics for each conflict snippet (context before/after + resolution)."
    )
    parser.add_argument(
        "--input_dir",
        default="merges/repos_small/conflict_blocks",
        help="Directory containing <basename><n>.conflict and "
        "<basename><n>.resolved_conflict files",
    )
    parser.add_argument(
        "--csv_out",
        default="merges/repos_small/conflict_metrics.csv",
        help="Path to the output CSV file.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.stderr.write(f"ERROR: input_dir '{input_dir}' is not a directory.\n")
        sys.exit(1)

    # Collect all *.conflict files; we'll match each with a corresponding *.resolved_conflict
    conflict_files = sorted(input_dir.glob("*.conflict"))
    if not conflict_files:
        print("No '.conflict' files found.")
        sys.exit(0)

    # Prepare CSV
    csv_header = [
        "conflict_id",
        "context_before_size",
        "conflict_size",
        "context_after_size",
        "resolution_size",
    ]
    rows = []

    for conflict_path in conflict_files:
        # For each .conflict, find the .resolved_conflict
        resolved_path = conflict_path.with_suffix(".resolved_conflict")
        if not resolved_path.exists():
            # If it doesn't exist, skip (or warn)
            print(f"No matching .resolved_conflict for {conflict_path}. Skipped.")
            continue

        # We'll treat the conflict_id minus extension as the "identifier"
        # e.g. "myfile1.conflict" -> "myfile1"
        identifier = conflict_path.stem  # e.g. "myfile1"

        # Read lines (without newline chars)
        conflict_lines = conflict_path.read_text(encoding="utf-8").splitlines()
        resolved_lines = resolved_path.read_text(encoding="utf-8").splitlines()

        # Re-split the snippet
        before_ctx, conflict_block, after_ctx = split_single_conflict_snippet(
            conflict_lines
        )

        # Remove the before_context and after_context from the resolved_lines
        # to get the resolved conflict block
        resolved_lines = resolved_lines[len(before_ctx) : -len(after_ctx)]

        # If we fail to find conflict markers, skip or record zeros
        if not conflict_block:
            print(f"Could not find valid conflict markers in {conflict_path}. Skipped.")
            continue

        # Compute lengths
        context_before_size = len(before_ctx)
        conflict_size = len(conflict_block)
        context_after_size = len(after_ctx)
        resolution_size = len(resolved_lines)

        row_data = {
            "conflict_id": identifier,
            "context_before_size": context_before_size,
            "conflict_size": conflict_size,
            "context_after_size": context_after_size,
            "resolution_size": resolution_size,
        }
        rows.append(row_data)

    # Write the CSV
    with open(args.csv_out, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metrics have been written to {args.csv_out}")


if __name__ == "__main__":
    main()
