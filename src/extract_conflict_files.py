#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
"""
ExtractConflictFiles.py

Modified to process each merge (row) in parallel rather than each repository in parallel
to reduce heat imbalance. Each merge commit is one task in the thread pool.
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
    """
    Creates a temporary working directory to reproduce the merge, e.g.
    .workdir/org/repo_name/<merge_sha>.

    Arguments:
        org: str
            GitHub organization name.
        repo_name: str
            GitHub repository name.
        merge_sha: str
            SHA of the merge commit.

    Returns:
        Path
            Path to the temporary working directory.
    """
    base_workdir = Path(os.getenv("WORKDIR_PATH", ".workdir"))
    temp_dir = base_workdir / org / repo_name / merge_sha
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    return temp_dir


def checkout_commit(repo: Repo, commit_sha: str, branch_name: str) -> None:
    """
    Creates/forces a local branch at commit_sha with given branch_name.
    If branch_name exists, it's reset to commit_sha.

    Arguments:
        repo: Repo
            The GitPython repository object.
        commit_sha: str
            The commit SHA to checkout.
        branch_name: str
            The name of the branch to create/force.
    """
    git = repo.git
    # If branch exists, delete it
    try:
        git.branch("-D", branch_name)
    except GitCommandError:
        pass
    # Create the new branch
    git.checkout(commit_sha, b=branch_name)


def copy_conflicting_files_and_goal(
    conflict_files: List[Path],
    final_repo: Repo,
    final_commit_sha: str,
    output_dir: Path,
    org: str,
    repo_name: str,
    left_sha: str,
    right_sha: str,
):
    """
    Copy conflict-marked files from the merge_repo working tree, and also
    copy the final merged ("goal") file from the real merged commit.

    Arguments:
        conflict_files: List[Path]
            List of conflict-marked files from the merge_repo working tree.
        final_repo: Repo
            The GitPython repository object for the final merged commit.
        final_commit_sha: str
            The SHA of the final merged commit.
        output_dir: Path
            The output directory to store the conflict files.
        org: str
            GitHub organization name.
        repo_name: str
            GitHub repository name.
        left_sha: str
            SHA of the left parent commit.
        right_sha: str
            SHA of the right parent commit.
    """
    conflict_root = (
        output_dir
        / "file_conflicts"
        / org
        / repo_name
        / f"{left_sha[:7]}-{right_sha[:7]}"
    )

    for conflict_num, conflict_file in enumerate(conflict_files):

        # 1) conflict-marked file from local working tree
        conflict_file_path = Path(final_repo.working_dir) / conflict_file
        if not conflict_file_path.is_file():
            logger.warning(f"Conflict file not found in working tree: {conflict_file_path}")
            continue
        conflict_marked_target = conflict_root / f"{conflict_num}.java"
        conflict_marked_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(conflict_file_path, conflict_marked_target)

        # 2) final merged version from the real merged commit
        final_file_target = conflict_root / f"{conflict_num}_goal.java"
        try:
            # "git show <commit>:<path>" to get content
            content = final_repo.git.show(f"{final_commit_sha}:{conflict_file}")
            final_file_target.parent.mkdir(parents=True, exist_ok=True)
            final_file_target.write_text(content, encoding="utf-8")
        except GitCommandError:
            # Possibly the file was deleted/renamed in final commit
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
):
    """
    `) Checkout parent1, merge parent2.
    3) If conflict => copy conflict-marked files + final merged files to output_dir.

    Arguments:
        repo: Repo
            The GitPython repository object.
        left_sha: str
            The SHA of the left parent commit.
        right_sha: str
            The SHA of the right parent commit.
        merge_sha: str
            The SHA of the merge commit.
        output_dir: Path
            The output directory to store the conflict files.
        org: str
            GitHub organization name.
        repo_name: str
            GitHub repository name.
    """
    # 1) checkout + merge
    git = repo.git

    # Checkout left parent
    checkout_commit(repo, left_sha, "left")

    conflict_files:List[Path] = []
    try:
        git.merge(right_sha)
    except GitCommandError as e:
        # Possibly a conflict
        status_output = git.status("--porcelain")
        for line in status_output.splitlines():
            # "UU <filename>"
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

    # 3) If we have conflict files, copy them
    if conflict_files:
        copy_conflicting_files_and_goal(
            conflict_files=conflict_files,
            final_repo=repo,
            final_commit_sha=merge_sha,
            output_dir=output_dir,
            org=org,
            repo_name=repo_name,
            left_sha=left_sha,
            right_sha=right_sha,
        )


def process_single_merge(
    org: str,
    repo_name: str,
    merge_sha: str,
    parent1: str,
    parent2: str,
    output_dir: Path,
):
    """
    Worker function to handle exactly one merge from the CSV.
    1) Get or clone the repo once (from the shared repo cache).
    2) Create a temporary directory for that merge.
    3) Reproduce the merge.

    Arguments:
        org: str
            GitHub organization name.
        repo_name: str
            GitHub repository name.
        merge_sha: str
            SHA of the merge commit.
        parent1: str
            SHA of the left parent commit.
        parent2: str
            SHA of the right parent commit.
        output_dir: Path
            The output directory to store the conflict files.
    """
    try:
        repo = get_repo(org, repo_name, log=False)
    except Exception as e:
        logger.error(f"Skipping {org}/{repo_name} due to clone error: {e}")
        return

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
        )
    except Exception as exc:
        logger.error(
            f"Error reproducing merge {merge_sha} in {org}/{repo_name}: {exc}"
        )
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """ Main function

    Raises:
        ValueError: Duplicate merge commits found in the input CSV
    """
    parser = argparse.ArgumentParser(
        description="Extract conflict files from merges listed in a CSV."
    )
    parser.add_argument("--merges",
                        required=True,
                        help="CSV with merges (org/repo, merge_commit, parent_1, parent_2)")
    parser.add_argument("--output_dir", required=True, help="Directory to store conflict files")
    parser.add_argument(
        "--n_threads",
        required=False,
        type=int,
        default=None,
        help="Number of parallel threads (if not specified: use all CPU cores)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merges)
    required_cols = {"repository", "merge_commit", "parent_1", "parent_2"}
    if not required_cols.issubset(df.columns):
        sys.exit(f"CSV must contain columns: {required_cols}")

    # Make sure all the merges are unique
    if df["merge_commit"].duplicated().any():
        raise ValueError("Duplicate merge commits found in the input CSV")

    # Prepare a list of merges (tasks)
    merges = []
    for _, row in df.iterrows():
        repo_str = row["repository"]
        merge_sha = row["merge_commit"]
        parent1 = row["parent_1"]
        parent2 = row["parent_2"]

        if "/" not in repo_str:
            logger.error(f"Invalid repository format '{repo_str}'")
            continue
        org, repo_name = repo_str.split("/", 1)

        merges.append((org, repo_name, merge_sha, parent1, parent2))

    # Determine number of workers
    num_workers = os.cpu_count() if args.n_threads is None else args.n_threads
    logger.info(f"Processing {len(merges)} merges using {num_workers} threads...")

    # Submit one task per merge
    def worker(task):
        org, repo_name, merge_sha, p1, p2 = task
        process_single_merge(
            org=org,
            repo_name=repo_name,
            merge_sha=merge_sha,
            parent1=p1,
            parent2=p2,
            output_dir=output_dir,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, m) for m in merges]
        for f in concurrent.futures.as_completed(futures):
            # If any exception escaped, raise it here
            exc = f.exception()
            if exc:
                logger.error(f"Worker thread raised an exception: {exc}")

    logger.info("Done extracting conflicts.")


if __name__ == "__main__":
    main()
