#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
"""
Usage:
    python3 extract_conflict_files.py \
        --merges path/to/merge_commits.csv \
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
import os
import shutil
import sys
from pathlib import Path
from typing import List

import pandas as pd
from git import GitCommandError, Repo
from loguru import logger

from find_merges import get_repo

logger.add("run.log", rotation="10 MB")


def create_temp_workdir(org: str, repo_name: str, merge_sha: str) -> Path:
    """Create a temporary working directory for the merge."""
    base_workdir = Path(os.getenv("WORKDIR_PATH", ".workdir"))
    temp_dir = base_workdir / org / repo_name / merge_sha
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    return temp_dir


def checkout_commit(repo: Repo, commit_sha: str, branch_name: str) -> None:
    """Checkout a commit by SHA and create a new branch."""
    git = repo.git
    try:
        git.branch("-D", branch_name)
    except GitCommandError:
        pass
    git.checkout(commit_sha, b=branch_name)


def copy_conflicting_files_and_goal(
    conflict_files: List[Path],
    final_repo: Repo,
    final_commit_sha: str,
    output_dir: Path,
    merge_id: str,
):
    """
    Copy conflict-marked files from the merge_repo working tree, and also
    copy the final merged ("goal") file from the real merged commit.
    Instead of writing under org/repo_name, files are written directly into output_dir
    using names: merge_id + letter (e.g. "1a.conflict", "1a.final_merged").
    Returns the list of conflict IDs (e.g. ["1a", "1b", ...]) for this merge.
    """
    for conflict_num, conflict_file in enumerate(conflict_files):
        letter = chr(ord("a") + conflict_num)
        conflict_id = f"{merge_id}{letter}"

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


def reproduce_merge_and_extract_conflicts(
    repo: Repo,
    left_sha: str,
    right_sha: str,
    merge_sha: str,
    output_dir: Path,
    org: str,
    repo_name: str,
    merge_id: str,
):
    """
    Checkout parent1, merge parent2.
    If conflict, copy conflict-marked files and final merged files to output_dir.
    Returns the list of conflict IDs for this merge.
    """
    git = repo.git
    checkout_commit(repo, left_sha, "left")

    conflict_files: List[Path] = []
    try:
        git.merge(right_sha)
    except GitCommandError as e:
        status_output = git.status("--porcelain")
        for line in status_output.splitlines():
            if line.startswith("UU "):
                path_part = line[3:].strip()
                if path_part.endswith(".java"):
                    conflict_files.append(path_part)
        if conflict_files:
            logger.info(
                f"Conflict in {org}/{repo_name} for {left_sha} + {right_sha} "
                f"=> {merge_sha}, files: {conflict_files}"
            )
        else:
            logger.warning(f"Merge error but no conflicts? {e}")

    if conflict_files:
        copy_conflicting_files_and_goal(
            conflict_files=conflict_files,
            final_repo=repo,
            final_commit_sha=merge_sha,
            output_dir=output_dir,
            merge_id=merge_id,
        )


def process_single_merge(
    org: str,
    repo_name: str,
    merge_sha: str,
    parent1: str,
    parent2: str,
    output_dir: Path,
    merge_id: str,
):
    """
    Worker function to handle exactly one merge from the CSV.
    1) Get or clone the repo once.
    2) Create a temporary directory for that merge.
    3) Reproduce the merge.
    Returns the list of conflict IDs (e.g. ["1a", "1b", ...]) for this merge.
    """
    try:
        repo = get_repo(org, repo_name, log=False)
    except Exception as e:
        logger.error(f"Skipping {org}/{repo_name} due to clone error: {e}")

    temp_dir = create_temp_workdir(org, repo_name, merge_sha)
    shutil.copytree(repo.working_dir, temp_dir, symlinks=True)
    repo_copy = Repo(temp_dir)

    try:
        reproduce_merge_and_extract_conflicts(
            repo=repo_copy,
            left_sha=parent1,
            right_sha=parent2,
            merge_sha=merge_sha,
            output_dir=output_dir,
            org=org,
            repo_name=repo_name,
            merge_id=merge_id,
        )
    except Exception as exc:
        logger.error(f"Error reproducing merge {merge_sha} in {org}/{repo_name}: {exc}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main function to extract conflict files from merges."""
    parser = argparse.ArgumentParser(description="Extract conflict files from merges.")
    parser.add_argument(
        "--merges",
        required=True,
        help="CSV with merges (org/repo, merge_commit, parent_1, parent_2)",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to store conflict files"
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

    df = pd.read_csv(args.merges)
    required_cols = {"repository", "merge_commit", "parent_1", "parent_2", "idx"}
    if not required_cols.issubset(df.columns):
        sys.exit(f"CSV must contain columns: {required_cols}")

    if df["merge_commit"].duplicated().any():
        raise ValueError("Duplicate merge commits found in the input CSV")

    num_workers = os.cpu_count() if args.n_threads is None else args.n_threads
    logger.info(f"Processing {len(df)} merges using {num_workers} threads...")

    def worker(task):
        row, output_dir = task
        org = row["repository"].split("/")[0]
        repo_name = row["repository"].split("/")[1]
        process_single_merge(
            org=org,
            repo_name=repo_name,
            merge_sha=row["merge_commit"],
            parent1=row["parent_1"],
            parent2=row["parent_2"],
            output_dir=output_dir,
            merge_id=str(row["idx"]),
        )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, (m[1], output_dir)) for m in df.iterrows()]
        for f in concurrent.futures.as_completed(futures):
            try:
                result = f.result()
                results.append(result)
            except Exception as exc:
                logger.error(f"Worker thread raised an exception: {exc}")

    logger.info("Done extracting conflicts.")


if __name__ == "__main__":
    main()
