#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
"""
Usage:
    python3 extract_conflict_files.py \
        --repos path/to/repos.csv \
        --output_dir path/to/conflict_files \
        --n_threads 8

Modified to use a two-step process:
1. Collect all merges in parallel (one task per repository)
2. Process conflict files in parallel (one task per merge)

Output files are named as:
    merge_id + letter .conflict
    merge_id + letter .final_merged
and a new CSV is dumped mapping each merge (using its merge_id) to the comma-separated
list of conflict file IDs.
"""

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import os
import sys
import shutil
from pathlib import Path
from typing import List
import pandas as pd
from git import GitCommandError, Repo
from loguru import logger
from rich.progress import Progress
import timeout_decorator

from find_merges import get_repo, get_merges


logger.add("run.log", backtrace=True, diagnose=True)

WORKING_DIR = Path(".workdir")


def concatenate_csvs(input_path: Path) -> pd.DataFrame:
    """
    Finds all CSV files in the given directory and its subdirectories, concatenates them,
    and saves the merged file to the specified output path.

    :param input_path: Directory containing CSV files.
    """
    csv_files = list(Path(input_path).rglob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {input_path}")

    dataframes = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


def checkout_commit(repo: Repo, commit_sha: str, branch_name: str) -> None:
    """Checkout a commit by SHA and create a new branch."""
    git = repo.git
    try:
        git.branch("-D", branch_name, force=True)
    except GitCommandError:
        pass
    git.checkout(commit_sha, b=branch_name)


def copy_conflicting_files_and_goal(
    conflict_files: List[Path],
    final_repo: Repo,
    final_commit_sha: str,
    output_dir: Path,
    merge_id: int,
) -> List[str]:
    """
    Copy conflict-marked files from the repo working tree, and also
    copy the final merged ("goal") file from the real merged commit.
    Files are written into output_dir with names: merge_id-index.conflict
    and merge_id-index.final_merged.
    Returns the list of conflict IDs (e.g. ["1-0", "1-1", ...]) for this merge.
    """
    conflicts: List[str] = []
    for conflict_num, conflict_file in enumerate(conflict_files):
        conflict_id = f"{merge_id}-{conflict_num}"

        # 1) conflict-marked file from local working tree
        conflict_file_path = Path(final_repo.working_dir) / conflict_file
        if not conflict_file_path.is_file():
            logger.warning(
                f"Conflict file not found in working tree: {conflict_file_path}"
            )
            continue
        conflict_marked_target = output_dir / f"{conflict_id}.conflict"
        conflict_marked_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(conflict_file_path, conflict_marked_target)
        conflicts.append(conflict_id)

        # 2) final merged version from the real merged commit
        final_file_target = output_dir / f"{conflict_id}.final_merged"
        try:
            content = final_repo.git.show(f"{final_commit_sha}:{conflict_file}")
            final_file_target.parent.mkdir(parents=True, exist_ok=True)
            final_file_target.write_text(content, encoding="utf-8")
        except GitCommandError:
            logger.warning(
                f"File {conflict_file_path} not found in final merged commit {final_commit_sha}"
            )
    return conflicts


def reproduce_merge_and_extract_conflicts(
    repo_slug: str,
    left_sha: str,
    right_sha: str,
    merge_sha: str,
    output_dir: Path,
    merge_id: int,
) -> List[str]:
    """
    Checkout left_sha, merge right_sha.
    If conflicts occur, copy conflict-marked files and final merged files to output_dir.
    Additionally, cache these files under "merge_cache/conflicts" using the original
    filenames. When copying them to output_dir, the files are renamed using merge_id-index.
    If the cache already exists, the cached files are simply copied over.
    Returns the list of conflict IDs for this merge.
    """
    cache_folder = Path("merge_cache/conflicts") / repo_slug / merge_sha
    conflict_cache_folder = cache_folder / "conflict"
    final_cache_folder = cache_folder / "final_merged"

    # Check if cache exists (both subdirectories must be present)
    if conflict_cache_folder.exists() and final_cache_folder.exists():
        logger.info(f"Using cached merge for {merge_sha} from {cache_folder}")
        cached_conflict_files = sorted(
            conflict_cache_folder.iterdir(), key=lambda p: p.name
        )
        conflicts = []
        for i, cached_file in enumerate(cached_conflict_files):
            conflict_id = f"{merge_id}-{i}"
            destination_conflict = output_dir / f"{conflict_id}.conflict"
            destination_final = output_dir / f"{conflict_id}.final_merged"
            destination_conflict.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_file, destination_conflict)
            # Look for corresponding final merged file using the same original filename
            final_file = final_cache_folder / cached_file.name
            if final_file.is_file():
                shutil.copy2(final_file, destination_final)
            conflicts.append(conflict_id)
        return conflicts

    # No cache exists, so reproduce the merge.
    logger.info(f"Reproducing merge {merge_sha} for {repo_slug}")
    conflict_cache_folder.mkdir(parents=True, exist_ok=True)
    final_cache_folder.mkdir(parents=True, exist_ok=True)
    repo = get_repo(repo_slug)
    temp_dir = WORKING_DIR / f"{repo_slug}_merge_{merge_id}"
    shutil.copytree(repo.working_dir, temp_dir, dirs_exist_ok=True)
    repo = Repo(temp_dir)
    repo.git.checkout(left_sha, force=True)
    conflict_files: List[Path] = []
    try:
        repo.git.merge(right_sha)
    except GitCommandError:
        status_output = repo.git.status("--porcelain")
        for line in status_output.splitlines():
            if line.startswith("UU "):
                path_part = line[3:].strip()
                if path_part.endswith(".java"):
                    conflict_files.append(Path(path_part))

    result: List[str] = []
    if conflict_files:
        logger.info(
            f"Conflict in {left_sha} + {right_sha} => {merge_sha}, files: {conflict_files}"
        )
        conflict_files.sort()
        result = copy_conflicting_files_and_goal(
            conflict_files=conflict_files,
            final_repo=repo,
            final_commit_sha=merge_sha,
            output_dir=output_dir,
            merge_id=merge_id,
        )
        for i, conflict_file in enumerate(conflict_files):
            conflict_id = f"{merge_id}-{i}"
            original_name = conflict_file.name
            # Cache conflict-marked file
            src_conflict = output_dir / f"{conflict_id}.conflict"
            dst_conflict = conflict_cache_folder / original_name
            if src_conflict.is_file():
                shutil.copy2(src_conflict, dst_conflict)
            # Cache final merged file
            src_final = output_dir / f"{conflict_id}.final_merged"
            dst_final = final_cache_folder / original_name
            if src_final.is_file():
                shutil.copy2(src_final, dst_final)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return result


def collect_merges(repo_slug: str, output_dir: Path) -> pd.DataFrame:
    """
    Step 1: Collect merges for a single repository.
    """
    try:
        repo = get_repo(repo_slug)
        merges = get_merges(repo, repo_slug, output_dir / "merges")
        logger.info(f"Collected merges for {repo_slug}")
    except Exception as e:
        logger.error(f"Error collecting merges for {repo_slug}: {e}")
        return pd.DataFrame()
    return merges


@timeout_decorator.timeout(5 * 60, use_signals=False)
def process_merge(merge_row, output_dir: Path) -> tuple:
    """
    Step 2: Process a single merge to extract conflict files.
    """
    repo_slug = merge_row["repository"]
    merge_id = merge_row.name
    try:
        conflicts = reproduce_merge_and_extract_conflicts(
            repo_slug=repo_slug,
            left_sha=merge_row["parent_1"],
            right_sha=merge_row["parent_2"],
            merge_sha=merge_row["merge_commit"],
            output_dir=output_dir / "conflict_files",
            merge_id=merge_id,
        )
        conflict_str = ";".join(conflicts)
        return merge_id, conflict_str
    except Exception as e:
        logger.error(
            f"Error processing merge {merge_row['merge_commit']} in {repo_slug}: {e}"
        )
    return merge_id, ""


def main():
    """Main function with two steps: collect merges, then extract conflict files."""
    parser = argparse.ArgumentParser(description="Extract conflict files from merges.")
    parser.add_argument(
        "--repos",
        default="input_data/repos_small.csv",
        help="CSV with merges (org/repo, merge_commit, parent_1, parent_2)",
    )
    parser.add_argument(
        "--output_dir",
        default="merges/repos_small",
        help="Directory to store conflict files",
    )
    parser.add_argument(
        "--n_threads",
        required=False,
        type=int,
        default=None,
        help="Number of parallel threads (if not specified: use all CPU cores)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conflict_files/cache").mkdir(parents=True, exist_ok=True)

    repos_df = pd.read_csv(args.repos)
    num_workers = os.cpu_count() - 1 if args.n_threads is None else args.n_threads  # type: ignore

    # STEP 1: Collect all merges in parallel
    logger.info(
        f"Step 1: Collecting merges for {len(repos_df)} repos using {num_workers} threads..."
    )

    result = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = [
            executor.submit(collect_merges, row["repository"], output_dir)
            for _, row in repos_df.iterrows()
        ]

        with Progress() as progress:
            progress_task = progress.add_task("Collecting merges...", total=len(tasks))
            for future in as_completed(tasks):
                try:
                    result.append(future.result())
                except Exception as exc:
                    logger.error(f"Worker thread raised an exception: {exc}")
                progress.advance(progress_task)

    # Combine all merge CSVs
    all_merges_df = pd.concat(result)
    all_merges_df["merge_idx"] = range(0, len(all_merges_df))
    all_merges_df.set_index("merge_idx", inplace=True)
    all_merges_df.to_csv(output_dir / "all_merges.csv")
    logger.info(f"Found {len(all_merges_df)} merges in total.")
    # Make sure merge_commit is unique
    if all_merges_df[["repository", "merge_commit"]].duplicated().any():
        logger.error("Duplicate merge_commit found in all_merges.csv")
        sys.exit(1)

    # STEP 2: Process each merge in parallel to extract conflict files
    logger.info(
        f"Step 2: Processing {len(all_merges_df)} merges for "
        f"conflicts using {num_workers} threads..."
    )
    all_merges_df["conflicts"] = ""

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_merge, merge_row, output_dir): merge_id
            for merge_id, merge_row in all_merges_df.iterrows()
        }
        with Progress() as progress:
            progress_task = progress.add_task(
                "Extracting conflicts...", total=len(futures)
            )
            # Process each future as it completes.
            for future in as_completed(futures):
                try:
                    merge_id, conflict_str = future.result()
                    all_merges_df.loc[merge_id, "conflicts"] = conflict_str
                except timeout_decorator.TimeoutError:
                    logger.error("Task timed out.")
                progress.advance(progress_task)

    # Combine all results
    all_merges_df = all_merges_df[all_merges_df["conflicts"] != ""]  # pylint: disable=C1804
    all_merges_df.to_csv(output_dir / "conflict_files.csv")

    logger.info("Done extracting conflict files.")


if __name__ == "__main__":
    main()
