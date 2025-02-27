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
      1) Reads the conflict snippet (which includes context + conflict markers).
      2) Identifies:
           - context_before (lines above <<<<<<<),
           - conflict_block (lines from <<<<<<< up through >>>>>>>),
           - context_after (lines after >>>>>>>).
      3) Further splits the conflict block into:
           - left_parent (lines after <<<<<<< until a base marker or =======),
           - base (lines after a base marker “|||||||” until =======, if present),
           - right_parent (lines after ======= until >>>>>>>).
      4) Reads the resolved snippet (the entire file).
      5) Computes metrics (in lines) for each portion.
      6) Writes all metrics to a CSV file using pandas.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from transformers import AutoTokenizer
from variables import MODEL
from build_dataset import build_query


def get_token_count(tokenizer, conflict_query: List[str]) -> int:
    """Returns the token count of the conflict query"""
    query = build_query("\n".join(conflict_query))
    return len(tokenizer(query)["input_ids"])


def split_single_conflict_snippet(
    lines: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Given a list of lines that contain exactly one conflict block
    (which should have <<<<<<< ... >>>>>>>),
    returns (before_context, conflict_block, after_context).

    If markers are not found, raises a ValueError.
    """
    start_idx = -1
    end_idx = -1

    # Locate <<<<<<< marker
    for i, line in enumerate(lines):
        if line.startswith("<<<<<<<"):
            start_idx = i
            break

    if start_idx == -1:
        raise ValueError("Could not find start marker (<<<<<<<).")

    # Locate >>>>>>> marker
    for j in range(start_idx, len(lines)):
        if lines[j].startswith(">>>>>>>"):
            end_idx = j
            break

    if end_idx == -1:
        raise ValueError("Found start marker but no end marker (>>>>>>>).")

    before_context = lines[:start_idx]
    conflict_block = lines[start_idx : end_idx + 1]  # include end marker line
    after_context = lines[end_idx + 1 :]

    return before_context, conflict_block, after_context


def split_conflict_block(
    conflict_block: List[str],
) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    Splits the conflict block into left_parent, right_parent, and base.

    The expected format is:
      <<<<<<< [optional identifier]
      (left parent lines)
      [optional base marker: "|||||||", followed by base lines]
      =======
      (right parent lines)
      >>>>>>>

    If a base marker is not found, base is returned as None.
    """
    left_parent: List[str] = []
    base: List[str] = []
    right_parent: List[str] = []
    encountered_base_marker = False

    state = "left"
    for line in conflict_block:
        if line.startswith("<<<<<<<"):
            # skip the start marker line
            continue
        if line.startswith("|||||||"):
            state = "base"
            encountered_base_marker = True
            continue
        if line.startswith("======="):
            state = "right"
            continue
        if line.startswith(">>>>>>>"):
            break
        if state == "left":
            left_parent.append(line)
        elif state == "base":
            base.append(line)
        elif state == "right":
            right_parent.append(line)

    return left_parent, right_parent, None if not encountered_base_marker else base


def is_sublist(sub: List[str], parent: List[str]) -> bool:
    """
    Checks if the list 'sub' is a contiguous sublist of 'parent'.
    Returns True if it is, otherwise False.
    """
    if not sub:
        return True
    for i in range(len(parent) - len(sub) + 1):
        if parent[i : i + len(sub)] == sub:
            return True
    return False


def main():  # pylint: disable=too-many-locals
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compute metrics for each conflict snippet "
        "(context, conflict, resolution, and parent versions)."
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

    # Prepare a list for metric rows
    rows = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    for conflict_path in conflict_files:
        # For each .conflict, find the corresponding .resolved_conflict file
        resolved_path = conflict_path.with_suffix(".resolved_conflict")
        if not resolved_path.exists():
            print(f"No matching .resolved_conflict for {conflict_path}. Skipped.")
            continue

        # Use the filename (minus extension) as the conflict identifier
        identifier = conflict_path.stem  # e.g. "myfile1"

        # Read lines (without newline characters)
        conflict_lines = conflict_path.read_text(encoding="utf-8").splitlines()
        resolved_lines = resolved_path.read_text(encoding="utf-8").splitlines()

        try:
            # Split the conflict snippet into context and conflict block
            before_ctx, conflict_block, after_ctx = split_single_conflict_snippet(
                conflict_lines
            )
        except ValueError as e:
            print(f"{e} in {conflict_path}. Skipped.")
            continue

        # Further split the conflict block into left_parent, right_parent, and base (if available)
        left_parent, right_parent, base = split_conflict_block(conflict_block)

        # Derive the resolved conflict block by removing the context portions
        # from the resolved file.
        # (This assumes the resolution maintains the context length.)
        if len(after_ctx) < 1:
            resolution = resolved_lines[len(before_ctx) :]
        else:
            resolution = resolved_lines[len(before_ctx) : -len(after_ctx)]

        # Check if the resolution is fully contained in either left_parent or right_parent.
        res_in_left = is_sublist(resolution, left_parent)
        res_in_right = is_sublist(resolution, right_parent)

        # Compute all metrics using the standard interface, including the new metric.
        metrics = {
            "full_conflict_lines": len(conflict_lines),
            "context_before_lines": len(before_ctx),
            "conflict_lines": len(conflict_block),
            "context_after_lines": len(after_ctx),
            "resolution_lines": len(resolution),
            "parent1_lines": len(left_parent),
            "parent2_lines": len(right_parent),
            "base_lines": len(base) if base is not None else -1,
            "num_tokens_query": get_token_count(tokenizer, conflict_lines),
            "resolution_in_left_or_right": res_in_left or res_in_right,
        }

        # Prepare row data
        row_data = {"conflict_id": identifier}
        row_data.update(metrics)  # type: ignore
        rows.append(row_data)

    # Create a pandas DataFrame from the list of rows and write it to CSV
    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False, encoding="utf-8")

    print(f"Metrics have been written to {args.csv_out}")


if __name__ == "__main__":
    main()
