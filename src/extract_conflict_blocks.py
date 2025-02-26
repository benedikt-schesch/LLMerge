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

from loguru import logger
from rich.progress import Progress

logger.add("run.log", rotation="10 MB", backtrace=True, diagnose=True)

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


# TODO: Improve this function
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


def extract_resolved_code(
    merged_lines: List[str], conflict_lines: List[str], start_idx: int, end_idx: int
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extracts the resolved code from the merged file by matching the context
    before and after the conflict block.
    """

    before_ctx, after_ctx = get_before_after_context(conflict_lines, start_idx, end_idx)

    i1 = match_subsequence(merged_lines, before_ctx, 0)
    if i1 < 0:
        logger.error("Before context not found in merged file.")
        raise ValueError("Before context not found in merged file.")
    start_search = i1 + len(before_ctx)
    i2 = match_subsequence(merged_lines, after_ctx, start_search)
    if i2 < 0 or i1 >= i2:
        logger.warning(
            "Resolved code not found in merged file due to context ordering."
        )
        raise ValueError("Resolved code not found in merged file.")
    start_after = i1 + len(before_ctx)
    merged_before_ctx = merged_lines[i1:start_after]
    merged_after_ctx = merged_lines[i2 : i2 + len(after_ctx)]
    return merged_lines[start_after:i2], merged_before_ctx, merged_after_ctx


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
        # NOTE: In extract_resolved_code, 'merged_after_ctx' currently
        # is everything from i2 to the end. You might want to slice it
        # if you need just the portion that should match after_ctx.
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
        logger.info(f"Successfully processed conflict block {basename}-{n}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract conflict blocks using context matching from conflict files."
    )
    parser.add_argument(
        "--input_dir",
        default="merges/repos_50/conflict_files",
        help="Processing directory",
    )
    parser.add_argument(
        "--output_dir",
        default="merges/repos_50/conflict_blocks",
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
    output_dir.mkdir(parents=True, exist_ok=True)
    conflict_files = sorted(input_dir.rglob("*.conflict"))
    conflict_files = [f for f in conflict_files if "conflict_blocks" not in f.parts]
    logger.info(f"Found {len(conflict_files)} conflict file(s) in {input_dir}")

    with Progress() as progress:
        task = progress.add_task(
            "Processing conflict files...", total=len(conflict_files)
        )
        for cfile in conflict_files:
            final_file = cfile.with_suffix(".final_merged")
            if not final_file.exists():
                logger.warning(f"No matching .final_merged for {cfile}")
                progress.advance(task)
                continue
            process_conflict_file(
                cfile, final_file, args.context, output_dir=output_dir
            )
            progress.advance(task)

    logger.info(f"Done processing conflict files. Output is in {output_dir}")
    print(f"Done processing conflict files. Output is in {output_dir}")


if __name__ == "__main__":
    main()
