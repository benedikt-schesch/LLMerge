#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
"""
Usage:
    python3 extract_conflict_files.py \
        --repos path/to/repos.csv \
        --output_dir path/to/conflict_files \
        --n_threads 8

Modified to process each merge (row) in parallel rather than each repository in parallel
to reduce heat imbalance. Each merge commit is one task in the thread pool.
Output files are now named as:
    merge_id + letter .conflict
    merge_id + letter .final_merged
and a new CSV is dumped mapping each merge (using its merge_id) to the comma-separated
list of conflict file IDs.
"""

import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
from pathlib import Path
from typing import List
import pandas as pd
from git import GitCommandError, Repo
from loguru import logger
from rich.progress import Progress

from find_merges import get_repo, get_merges


logger.add("run.log", rotation="10 MB")

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
    repo_idx: int,
) -> List[str]:
    """
    Copy conflict-marked files from the merge_repo working tree, and also
    copy the final merged ("goal") file from the real merged commit.
    Instead of writing under org/repo_name, files are written directly into output_dir
    using names: merge_id + letter (e.g. "1a.conflict", "1a.final_merged").
    Returns the list of conflict IDs (e.g. ["1a", "1b", ...]) for this merge.
    """
    conflicts: List[str] = []
    for conflict_num, conflict_file in enumerate(conflict_files):
        conflict_id = f"{repo_idx}-{merge_id}-{conflict_num}"

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
    repo: Repo,
    left_sha: str,
    right_sha: str,
    merge_sha: str,
    output_dir: Path,
    repo_idx: int,
    merge_id: int,
) -> List[str]:
    """
    Checkout parent1, merge parent2.
    If conflict, copy conflict-marked files and final merged files to output_dir.
    Returns the list of conflict IDs for this merge.

    When use_existing_clone=True the function assumes that the repo passed in is already
    a clone (a working copy) so it does not clone it again.
    """
    repo.git.checkout(left_sha, force=True)
    conflict_files: List[Path] = []
    try:
        repo.git.merge(right_sha)
    except GitCommandError as e:
        status_output = repo.git.status("--porcelain")
        for line in status_output.splitlines():
            if line.startswith("UU "):
                path_part = line[3:].strip()
                if path_part.endswith(".java"):
                    conflict_files.append(Path(path_part))
        if conflict_files:
            logger.info(
                f"Conflict in {left_sha} + {right_sha} "
                f"=> {merge_sha}, files: {conflict_files}"
            )
        else:
            logger.warning(f"Git error. {e}")
    result: List[str] = []
    if conflict_files:
        conflict_files.sort()
        result = copy_conflicting_files_and_goal(
            conflict_files=conflict_files,
            final_repo=repo,
            final_commit_sha=merge_sha,
            output_dir=output_dir,
            merge_id=merge_id,
            repo_idx=repo_idx,
        )
    # Clean up the clone for this merge task
    shutil.rmtree(repo.working_dir, ignore_errors=True)
    return result


def process_single_repo(repo_slug: str, repo_idx: int, output_dir: Path):
    """
    Worker function to handle all merges for one repository.
    Uses a cache (a CSV file) that records, for each merge,
    the conflict IDs (as a semicolon-separated string) produced by
    reproduce_merge_and_extract_conflicts.

    Before reprocessing a merge, it checks that for each conflict ID, both the conflict-marked file
    and final merged file exist. If they do, processing is skipped.
    At the end the cache is saved.

    This version first gets the merges (first-level task) and then processes
    each merge (second-level task)in parallel.
    For efficiency, it clones the repository only once (the base clone) and
    then for each merge, it copies the base clone to run the merge reproduction.
    """
    # Determine cache file name (one per repository)
    cache_file = output_dir / f"conflict_files/cache/{repo_slug}_line_conflicts.csv"
    if cache_file.exists():
        cache_df = pd.read_csv(cache_file)
        # Check cache: if a row exists and the conflict files are present, skip processing
        all_exist = True
        for idx, _ in cache_df.iterrows():
            if idx in cache_df.index:
                cached_row = cache_df.loc[idx]  # type: ignore
                cached_conflicts = cached_row["conflicts"]
                if pd.isna(cached_conflicts):
                    continue
                # Expecting conflict IDs to be stored as semicolon-separated string, e.g. "1a;1b"
                conflict_ids = [
                    cid.strip() for cid in cached_conflicts.split(";") if cid.strip()
                ]
                for cid in conflict_ids:
                    conflict_file = output_dir / f"conflict_files/{cid}.conflict"
                    if not conflict_file.exists():
                        all_exist = False
                        break
        if all_exist:
            logger.info(f"Skipping {repo_slug} as all conflicts are already processed.")
            return cache_df

    try:
        repo = get_repo(repo_slug)
    except Exception as e:
        logger.error(f"Error cloning {repo_slug}: {e}")
        return pd.DataFrame()
    merges = get_merges(repo, repo_slug, output_dir / "merges")

    cache_df = merges.copy()
    cache_df["conflicts"] = ""

    def process_merge(merge_id, merge_row):
        temp_dir = WORKING_DIR / f"{repo_slug}_merge_{merge_id}"
        shutil.copytree(repo.working_dir, temp_dir)
        temp_repo = Repo(temp_dir)
        try:
            conflicts = reproduce_merge_and_extract_conflicts(
                repo=temp_repo,
                left_sha=merge_row["parent_1"],
                right_sha=merge_row["parent_2"],
                merge_sha=merge_row["merge_commit"],
                output_dir=output_dir / "conflict_files",
                repo_idx=repo_idx,
                merge_id=merge_id,  # type: ignore
            )
        except Exception as e:
            logger.error(
                f"Error reproducing merge {merge_row['merge_commit']} in {repo_slug}: {e}"
            )
            conflicts = []
        shutil.rmtree(temp_dir, ignore_errors=True)
        return merge_id, ";".join(conflicts)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(process_merge, merge_id, merge_row): merge_id
            for merge_id, merge_row in merges.iterrows()
        }
        for future in as_completed(futures):
            merge_id, conflict_str = future.result()
            cache_df.at[merge_id, "conflicts"] = conflict_str

    # Save the updated cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(cache_file)
    # Also clean up the original repo clone, if any
    shutil.rmtree(repo.working_dir, ignore_errors=True)


def main():
    """Main function to extract conflict files from merges."""
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
        default=6,
        help="Number of parallel threads (if not specified: use all CPU cores)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repos_df = pd.read_csv(args.repos)

    num_workers = os.cpu_count() if args.n_threads is None else args.n_threads
    logger.info(f"Processing {len(repos_df)} repos using {num_workers} threads...")

    def worker(task):
        row, output_dir = task
        process_single_repo(
            repo_slug=row["repository"],
            repo_idx=row.name,
            output_dir=output_dir,
        )

    if num_workers is not None and num_workers < 2:
        for task in [(m[1], output_dir) for m in repos_df.iterrows()]:
            worker(task)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            tasks = [(m[1], output_dir) for m in repos_df.iterrows()]
            futures = [executor.submit(worker, task) for task in tasks]
            with Progress() as progress:
                progress_task = progress.add_task(
                    "Extracting conflicts...", total=len(futures)
                )
                for f in concurrent.futures.as_completed(futures):
                    try:
                        f.result()
                    except Exception as exc:
                        logger.error(f"Worker thread raised an exception: {exc}")
                    progress.advance(progress_task)
    df = concatenate_csvs(output_dir / "conflict_files/cache")
    df = df.dropna(subset=["conflicts"])
    df.drop(columns=["idx"], inplace=True)
    df.to_csv(output_dir / "conflict_files.csv")
    df = concatenate_csvs(output_dir / "merges")
    df.drop(columns=["idx"], inplace=True)
    df.to_csv(output_dir / "all_merges.csv")
    logger.info("Done extracting conflict files.")


if __name__ == "__main__":
    main()
