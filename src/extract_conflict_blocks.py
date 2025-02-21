#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CreateConflictDataset_ContextMatching.py

Usage:
    python CreateConflictDataset_ContextMatching.py \
        --conflict_dir path/to/file_conflicts \
        --output_dir /path/to/output_folder \
        --context 3

This script:
  1. Recursively scans `--conflict_dir` for "*.conflict" files.
  2. Identifies the corresponding "*.final_merged" file for each.
  3. Splits the conflict-markers file into conflict blocks.
  4. For each block:
       - gathers N lines of context before <<<<<<< and after >>>>>>>,
       - locates that context in the merged file,
       - extracts everything in between as the “resolved conflict.”
  5. Writes each conflict block to two files:
       - <basename><n>.conflict        (the conflict block + context from the .conflict file)
       - <basename><n>.resolved_conflict (the code found in the merged
        file between the matched contexts)
       where <basename> is the base of the input file (e.g. "1a") and n is the conflict number.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def split_conflict_blocks(lines: List[str]) -> List[Tuple[int, int]]:
    """
    Find conflict blocks by locating:
        <<<<<<<
          ...
        =======
          ...
        >>>>>>>
    Returns a list of (start_index, end_index) inclusive of the conflict markers.
    """
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<<"):
            start_idx = i
            sep_idx = -1
            end_idx = -1
            j = i + 1
            while j < len(lines) and sep_idx < 0:
                if lines[j].startswith("======="):
                    sep_idx = j
                j += 1
            while j < len(lines) and end_idx < 0:
                if lines[j].startswith(">>>>>>>"):
                    end_idx = j
                j += 1
            if sep_idx == -1 or end_idx == -1:
                break
            blocks.append((start_idx, end_idx))
            i = end_idx + 1
        else:
            i += 1
    return blocks


def match_subsequence(lines: List[str], subseq: List[str], start_idx: int = 0) -> int:
    """
    Searches for `subseq` in `lines` (exact match on each line, in order),
    starting from `start_idx`.
    Returns the index in `lines` where `subseq` begins, or -1 if not found.
    """
    if not subseq:
        return start_idx
    n_lines = len(lines)
    n_subseq = len(subseq)
    for i in range(start_idx, n_lines - n_subseq + 1):
        if all(lines[i + j] == subseq[j] for j in range(n_subseq)):
            return i
    return -1


def get_before_after_context(
    lines: List[str], start_idx: int, end_idx: int, context: int
) -> Tuple[List[str], List[str]]:
    """
    Returns two lists of lines:
      before_context: the N lines immediately above start_idx
      after_context: the N lines immediately below end_idx
    """
    before_context = lines[max(0, start_idx - context) : start_idx]
    after_context = lines[end_idx + 1 : min(len(lines), end_idx + 1 + context)]
    return before_context, after_context


def gather_conflict_plus_context(
    lines: List[str], start_idx: int, end_idx: int, context: int
) -> str:
    """
    Returns a snippet containing the conflict block (with markers) plus
    N lines before and N lines after.
    """
    snippet = lines[
        max(0, start_idx - context) : min(len(lines), end_idx + 1 + context)
    ]
    return "".join(snippet)


def extract_resolved_code(
    merged_lines: List[str], before_context: List[str], after_context: List[str]
) -> List[str]:
    """
    1. Finds `before_context` in merged_lines (exact line match).
    2. Starting immediately after that block, finds `after_context`.
    3. Returns the lines in between as the resolved code.
    Returns an empty list if not found.
    """
    i1 = match_subsequence(merged_lines, before_context, 0)
    if i1 < 0:
        return []
    while i1 >= 0:
        start_after = i1 + len(before_context)
        i2 = match_subsequence(merged_lines, after_context, start_after)
        if i2 >= 0:
            return merged_lines[start_after:i2]
        i1 = match_subsequence(merged_lines, before_context, i1 + 1)
    return []


def process_conflict_file(  # pylint: disable=too-many-locals
    conflict_file: Path, final_file: Path, context: int, output_dir: Path
) -> None:
    """
    Processes one pair of files:
      - Reads the conflict-markers file and its corresponding final_merged file.
      - Splits the conflict file into conflict blocks.
      - For each block:
           * Gathers context (N lines before and after).
           * In the merged file, locates that context and extracts the resolved code.
           * Writes the conflict snippet and the resolved snippet to:
             <basename><n>.conflict and <basename><n>.resolved_conflict
           where <basename> is the stem of conflict_file (e.g. "1a") and n is the block number.
    """
    basename = conflict_file.stem  # e.g. "1a" from "1a.conflict"
    conflict_lines = conflict_file.read_text(encoding="utf-8").splitlines(keepends=True)
    merged_lines = final_file.read_text(encoding="utf-8").splitlines(keepends=True)
    blocks = split_conflict_blocks(conflict_lines)
    if not blocks:
        return
    for n, (start_idx, end_idx) in enumerate(blocks, start=1):
        conflict_snippet = gather_conflict_plus_context(
            conflict_lines, start_idx, end_idx, context
        )
        before_ctx, after_ctx = get_before_after_context(
            conflict_lines, start_idx, end_idx, context
        )
        resolved_lines = extract_resolved_code(merged_lines, before_ctx, after_ctx)
        # Assemble the resolved snippet: context before + resolved block + context after
        resolved_snippet = (
            "".join(before_ctx) + "".join(resolved_lines) + "".join(after_ctx)
        )
        conflict_output = output_dir / f"{basename}{n}.conflict"
        resolved_output = output_dir / f"{basename}{n}.resolved_conflict"
        conflict_output.write_text(conflict_snippet, encoding="utf-8")
        resolved_output.write_text(resolved_snippet, encoding="utf-8")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract conflict blocks using context matching from conflict files."
    )
    parser.add_argument(
        "--conflict_dir",
        required=True,
        help="Base directory containing .conflict and .final_merged files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for <basename><n>.conflict and"
        "<basename><n>.resolved_conflict files",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=3,
        help="Number of context lines to include before/after the conflict block",
    )
    args = parser.parse_args()

    conflict_dir = Path(args.conflict_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conflict_files = sorted(conflict_dir.rglob("*.conflict"))
    for cfile in conflict_files:
        final_file = cfile.with_suffix(".final_merged")
        if not final_file.exists():
            sys.stderr.write(f"No matching .final_merged for {cfile}\n")
            continue
        process_conflict_file(cfile, final_file, args.context, output_dir)

    print(f"Done processing conflict files. Output is in {output_dir}")


if __name__ == "__main__":
    main()
