#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CreateConflictDataset_ContextMatching.py

Usage:
    python CreateConflictDataset_ContextMatching.py \
        --conflict_dir path/to/conflict_files \
        --output_dir /path/to/base_output_folder \
        --context 3

This script:
  1. Recursively scans `--conflict_dir` for "*.conflict" files.
  2. Identifies the corresponding "*.final_merged" file for each.
  3. Splits the conflict-markers file into conflict blocks.
  4. For each block:
       - gathers up to 20 lines of context before <<<<<<< and after >>>>>>>,
         stopping if it hits another conflict marker.
       - locates that context in the merged file,
       - extracts everything in between as the “resolved conflict.”
  5. Writes each conflict block to two files:
       - <basename><n>.conflict        (the conflict block + context from the .conflict file)
       - <basename><n>.resolved_conflict (the code found in the merged
         file between the matched contexts)
       where <basename> is the base of the input file (e.g. "1a") and n is the conflict number.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import shutil
import random
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from rich.progress import Progress

logger.add("run.log", backtrace=True, diagnose=True)

MAX_CONTEXT_RESOLUTION_EXTRACTION = 20


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
                logger.error(
                    "Incomplete conflict block found. Stopping further processing."
                )
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


def find_all_matches(
    lines: List[str], subseq: List[str], start_idx: int = 0
) -> List[int]:
    """
    Returns a list of all indices in `lines` where `subseq` (exact match) occurs.
    """
    matches = []
    idx = start_idx
    while idx < len(lines):
        pos = match_subsequence(lines, subseq, idx)
        if pos == -1:
            break
        matches.append(pos)
        idx = pos + 1
    return matches


def get_before_after_context(
    lines: List[str],
    start_idx: int,
    end_idx: int,
) -> Tuple[List[str], List[str]]:
    """
    Returns two lists of lines: (before_context, after_context), each containing
    *up to* 20 lines of context before/after the conflict block.
    We stop collecting context lines if we encounter another conflict marker.
    """
    # Collect "before" context, up to 20 lines or until another conflict marker.
    before_context: List[str] = []
    cur = start_idx - 1
    while cur >= 0 and len(before_context) < MAX_CONTEXT_RESOLUTION_EXTRACTION:
        if (
            lines[cur].startswith("<<<<<<<")
            or lines[cur].startswith("=======")
            or lines[cur].startswith(">>>>>>>")
        ):
            break
        before_context.append(lines[cur])
        cur -= 1
    before_context.reverse()

    # Collect "after" context, up to 20 lines or until another conflict marker.
    after_context: List[str] = []
    cur = end_idx + 1
    while cur < len(lines) and len(after_context) < MAX_CONTEXT_RESOLUTION_EXTRACTION:
        if (
            lines[cur].startswith("<<<<<<<")
            or lines[cur].startswith("=======")
            or lines[cur].startswith(">>>>>>>")
        ):
            break
        after_context.append(lines[cur])
        cur += 1

    return before_context, after_context


# Updated function to handle non-unique matches.
def extract_resolved_code(
    merged_lines: List[str], conflict_lines: List[str], start_idx: int, end_idx: int
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extracts the resolved code from the merged file by matching the context
    before and after the conflict block.

    If both before and after contexts are non-unique in the merged file,
    logs a warning and raises ValueError (skipping the case).
    If one context is unique, selects the closest match for the other context
    based on line distance.
    """
    before_ctx, after_ctx = get_before_after_context(conflict_lines, start_idx, end_idx)

    before_matches = find_all_matches(merged_lines, before_ctx)
    after_matches = find_all_matches(merged_lines, after_ctx)

    if not before_matches:
        logger.error("Before context not found in merged file.")
        raise ValueError("Before context not found in merged file.")
    if not after_matches:
        logger.error("After context not found in merged file.")
        raise ValueError("After context not found in merged file.")

    # If both contexts are non-unique, log a warning and skip this case.
    if len(before_matches) > 1 and len(after_matches) > 1:
        logger.warning("Both before and after contexts are not unique in merged file.")
        raise ValueError("Resolved code not found due to non-unique context matches.")

    # If before context is unique, pick the closest after context (by line distance)
    if len(before_matches) == 1:
        i1 = before_matches[0]
        valid_after = [a for a in after_matches if a >= i1 + len(before_ctx)]
        if not valid_after:
            logger.warning(
                "No valid after context match found after the unique before context."
            )
            raise ValueError("Resolved code not found: no valid after context match.")
        i2 = min(valid_after, key=lambda a: a - (i1 + len(before_ctx)))
    # Else if after context is unique, pick the closest before context.
    elif len(after_matches) == 1:
        i2 = after_matches[0]
        valid_before = [b for b in before_matches if b + len(before_ctx) <= i2]
        if not valid_before:
            logger.warning(
                "No valid before context match found before the unique after context."
            )
            raise ValueError("Resolved code not found: no valid before context match.")
        i1 = min(valid_before, key=lambda b: i2 - (b + len(before_ctx)))
    else:
        # Both contexts are unique.
        i1 = before_matches[0]
        i2 = after_matches[0]
        if i1 + len(before_ctx) > i2:
            logger.warning(
                "Resolved code not found in merged file due to context ordering."
            )
            raise ValueError("Resolved code not found in merged file.")

    start_after = i1 + len(before_ctx)
    merged_before_ctx = merged_lines[i1:start_after]
    merged_after_ctx = merged_lines[i2 : i2 + len(after_ctx)]
    return merged_lines[start_after:i2], merged_before_ctx, merged_after_ctx


def gather_conflict_plus_context(
    lines: List[str], start_idx: int, end_idx: int, context: int
) -> Tuple[str, List[str], List[str]]:
    """
    Returns a snippet containing the conflict block (with markers) plus
    N lines before and N lines after.
    """
    context_begin_idx = max(0, start_idx - context)
    context_end_idx = min(len(lines), end_idx + 1 + context)
    before_context = lines[context_begin_idx:start_idx]
    after_context = lines[end_idx + 1 : context_end_idx]
    snippet = lines[context_begin_idx : min(len(lines), context_end_idx)]
    return "".join(snippet), before_context, after_context


def check_coherence(
    ctx_conflict: List[str], ctx_merged: List[str], alignment: str
) -> None:
    """
    Checks that the shorter context is fully contained in the corresponding
    portion (prefix or suffix) of the longer context. Raises ValueError if not.

    :param ctx_conflict: context lines from the conflict file
    :param ctx_merged:   context lines from the merged file
    :param alignment:    'prefix' or 'suffix'
    """
    if not ctx_conflict or not ctx_merged:
        # If one is empty, no conflict so no check needed
        return

    len_conflict = len(ctx_conflict)
    len_merged = len(ctx_merged)

    # Identify the common length to compare
    common_length = min(len_conflict, len_merged)

    if alignment == "prefix":
        # Compare from the start
        conflict_slice = ctx_conflict[:common_length]
        merged_slice = ctx_merged[:common_length]
    elif alignment == "suffix":
        # Compare from the end
        conflict_slice = ctx_conflict[-common_length:]
        merged_slice = ctx_merged[-common_length:]
    else:
        raise ValueError(f"Unknown alignment '{alignment}'")

    if conflict_slice != merged_slice:
        raise ValueError(
            f"Incoherent context in {alignment} check. "
            f"Conflict context slice: {conflict_slice} "
            f"vs Merged context slice: {merged_slice}"
        )


def process_conflict_file(  # pylint: disable=too-many-locals
    conflict_file: Path, final_file: Path, context: int, output_dir: Path
) -> None:
    """
    Processes one pair of files:
      - Reads the conflict-markers file and its corresponding final_merged file.
      - Splits the conflict file into conflict blocks.
      - For each block:
           * Gathers a normal snippet (+/- context lines) for writing .conflict.
           * Uses up to 20 lines of marker-free context to match inside the final_merged.
           * Extracts the resolved code from final_merged.
           * Writes conflict snippet and resolved snippet.
           * Performs coherence checks on before/after contexts.
    """
    basename = conflict_file.stem  # e.g. "1a" from "1a.conflict"
    logger.info(f"Processing file: {conflict_file}")
    conflict_lines = conflict_file.read_text(encoding="utf-8").splitlines(keepends=True)
    merged_lines = final_file.read_text(encoding="utf-8").splitlines(keepends=True)
    blocks = split_conflict_blocks(conflict_lines)
    logger.info(f"Found {len(blocks)} conflict block(s) in {conflict_file}")
    for n, (start_idx, end_idx) in enumerate(blocks, start=1):
        if basename == "12-54-0-9" and n == 9:
            print("STOP")
        conflict_snippet, before_ctx, after_ctx = gather_conflict_plus_context(
            conflict_lines, start_idx, end_idx, context
        )
        try:
            (
                resolved_lines,
                merged_before_ctx,
                merged_after_ctx,
            ) = extract_resolved_code(merged_lines, conflict_lines, start_idx, end_idx)
        except ValueError:
            logger.warning(
                f"Skipping conflict block {basename}-{n} due to missing resolved code."
            )
            continue

        # ------------------
        # 1) Coherence Check on "before" context (suffix alignment)
        #    The lines from the conflict file’s before_ctx must match
        #    the tail of merged_before_ctx (or vice versa) if one is shorter.
        # ------------------
        try:
            check_coherence(before_ctx, merged_before_ctx, alignment="suffix")
        except ValueError as e:
            logger.error(f"Before-context mismatch in {basename}-{n}: {e}")
            raise

        # ------------------
        # 2) Coherence Check on "after" context (prefix alignment)
        #    The lines from the conflict file’s after_ctx must match
        #    the beginning of merged_after_ctx if one is shorter.
        # ------------------
        try:
            check_coherence(after_ctx, merged_after_ctx, alignment="prefix")
        except ValueError as e:
            logger.error(f"After-context mismatch in {basename}-{n}: {e}")
            raise

        # Build the final resolved snippet
        resolved_snippet = (
            "".join(before_ctx) + "".join(resolved_lines) + "".join(after_ctx)
        )

        # Minimal consistency check: ensure the actual resolved code is somewhere in merged.
        if "".join(resolved_lines) not in "".join(merged_lines):
            logger.error("Resolved snippet consistency check failed.")
            raise ValueError("Resolved snippet not found in merged file.")

        # Write the conflict snippet and resolved snippet
        conflict_output = output_dir / f"{basename}-{n}.conflict"
        resolved_output = output_dir / f"{basename}-{n}.resolved_conflict"

        conflict_output.write_text(conflict_snippet, encoding="utf-8")
        resolved_output.write_text(resolved_snippet, encoding="utf-8")
        logger.success(f"Successfully processed conflict block {basename}-{n}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract conflict blocks using context matching from conflict files."
    )
    parser.add_argument(
        "--input_dir",
        default="merges/repos_reaper_1000/conflict_files",
        help="Processing directory",
    )
    parser.add_argument(
        "--output_dir",
        default="merges/repos_reaper_1000/conflict_blocks",
        help="Output directory for conflict snippets",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=5,
        help="Number of context lines to include in the conflict snippet",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sort by full path string to ensure deterministic processing order
    conflict_files = sorted(input_dir.rglob("*.conflict"))
    conflict_files = [f for f in conflict_files if "conflict_blocks" not in f.parts]
    logger.info(f"Found {len(conflict_files)} conflict file(s) in {input_dir}")

    # Add seed parameter for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=8,
        help="Number of threads for parallel processing (default: 8)",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    with Progress() as progress:
        task = progress.add_task(
            "Processing conflict files...", total=len(conflict_files)
        )

        def process(cfile):
            final_file = cfile.with_suffix(".final_merged")
            if not final_file.exists():
                logger.warning(f"No matching .final_merged for {cfile}")
                return
            process_conflict_file(
                cfile, final_file, args.context, output_dir=output_dir
            )

        # Use fixed number of threads with ordered processing
        num_workers = args.n_threads
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create futures dictionary with index to preserve order
            futures_dict = {
                executor.submit(process, cfile): i
                for i, cfile in enumerate(conflict_files)
            }

            # Process results in deterministic order
            ordered_futures = [
                f for _, f in sorted([(i, f) for f, i in futures_dict.items()])
            ]

            for future in ordered_futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing conflict file: {e}")
                finally:
                    progress.advance(task)

    logger.info(f"Done processing conflict files. Output is in {output_dir}")
    print(f"Done processing conflict files. Output is in {output_dir}")


if __name__ == "__main__":
    main()
