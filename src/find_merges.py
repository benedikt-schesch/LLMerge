#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FindMergeCommits.py

This script finds 2-parent merge commits in a set of GitHub repositories and outputs them to CSVs.
It is a Python translation of a Java program of the same name, enhanced with parallel processing,
environment-driven cache paths, and optional deletion of local clones.

**Workflow**:
1. Parse command line arguments (uses argparse).
2. Read a CSV (via pandas) of GitHub repositories
        (column named "repository" with values "org/repo").
3. For each repository:
   - Clone or reuse a local copy in a user-specified or environment-driven cache path.
   - Fetch pull-request branches.
   - Enumerate all branches; find merge commits with exactly 2 parents.
   - Compute a "merge base" commit:
       - If none is found, mark the merge as "two initial commits".
       - If the base is identical to one of the parents, mark "a parent is the base".
   - Write results to an output CSV file: <output-dir>/<org>/<repo>.csv
4. If the --delete flag is set, remove the local clone after processing.

**Parallelization**:
- By default, processes up to n_cpus repositories in parallel (can be changed via --threads).

**Authentication**:
- The script uses a GitHub token for cloning
        (Git operations require authentication for certain public or high-rate usage).
- Credentials can be supplied by:
  - A ~/.github-personal-access-token file (first line = username, second line = token),
  - Or an environment variable GITHUB_TOKEN, in which case we treat username="Bearer".
"""

import argparse
import concurrent.futures
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Set

import pandas as pd
from git import Commit, GitCommandError, Repo
from loguru import logger


def read_github_credentials() -> tuple[str, str]:
    """
    Returns a tuple (username, token) for GitHub authentication:
      1) Reads from ~/.github-personal-access-token if present
            (first line = user, second line = token).
      2) Otherwise uses environment variable GITHUB_TOKEN (with user="Bearer").
      3) Exits if neither is available.

    Raises:
        RuntimeError: If neither ~/.github-personal-access-token nor GITHUB_TOKEN
            environment variable is found.

    Returns:
        tuple[str, str]: A tuple containing:
            - username (str): GitHub username or "Bearer".
            - token (str): Personal access token for GitHub authentication.
    """
    token_file = Path.home() / ".github-personal-access-token"
    env_token = os.getenv("GITHUB_TOKEN")
    if token_file.is_file():
        lines = token_file.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            sys.exit("~/.github-personal-access-token must have at least two lines.")
        return lines[0].strip(), lines[1].strip()
    if env_token:
        return "Bearer", env_token
    raise RuntimeError("Need ~/.github-personal-access-token or GITHUB_TOKEN.")


def fetch_pr_branches(repo: Repo) -> None:
    """
    Fetch pull request branches (refs/pull/*/head) into local references.
    Equivalent to 'git fetch origin refs/pull/*/head:refs/remotes/origin/pull/*'.

    Arguments:
        repo (Repo): The Git repository object.
    """
    try:
        repo.remotes.origin.fetch(refspec="refs/pull/*/head:refs/remotes/origin/pull/*")
    except GitCommandError:
        # Some repos may not have PR refs; ignore errors
        pass


def get_merge_base(repo: Repo,
                   c1:Commit,
                   c2:Commit) -> Optional[Commit]:
    """
    Compute the "nearest common ancestor" (merge base) of two commits c1 and c2.
    If no common ancestor exists (meaning the commits share no history),
    return None ("two initial commits").
    If the merge base is one of the parents themselves, note that separately.

    Arguments:
        repo (Repo): The Git repository object.
        c1 (Commit): The first commit.
        c2 (Commit): The second commit.

    Raises:
        RuntimeError: If the same commit is passed twice.

    Returns:
        Optional[Commit]: The merge base commit, or None if no common ancestor exists
            (or if c1 and c2 are the same commit).
    """
    if c1.hexsha == c2.hexsha:
        raise RuntimeError(f"Same commit passed twice: {c1.hexsha}")
    # Gather ancestors of each commit in reverse topological order
    h1 = list(repo.iter_commits(c1))
    h1.reverse()
    h2 = list(repo.iter_commits(c2))
    h2.reverse()
    # Quick membership checks
    s1 = {x.hexsha for x in h1}
    s2 = {x.hexsha for x in h2}
    if c2.hexsha in s1:
        return c2
    if c1.hexsha in s2:
        return c1
    # Find last common prefix
    length = min(len(h1), len(h2))
    common_prefix = 0
    for i in range(length):
        if h1[i].hexsha == h2[i].hexsha:
            common_prefix += 1
        else:
            break
    return None if common_prefix == 0 else h1[common_prefix - 1]


def write_branch_merges(
    repo: Repo,
    branch_ref,
    writer,
    start_idx: int,
    written_shas: Set[str],
) -> int:
    """
    For the given branch reference, find all 2-parent merges.
    Write each merge to CSV with an index, incrementing from start_idx.
    Return the new index after writing merges.

    Arguments:
        repo (Repo): The Git repository object.
        branch_ref: The branch reference object.
        writer: The CSV writer object.
        start_idx (int): The starting index for writing merges.
        written_shas (Set[str]): A set of written merge commit SHAs.

    Returns:
        int: The new index after writing merges.
    """
    merges = []
    try:
        commits = list(repo.iter_commits(branch_ref.path))
    except GitCommandError:
        # If commits can't be iterated (edge cases?), skip
        return start_idx
    # Collect merges
    for commit in commits:
        if len(commit.parents) == 2:
            merges.append(commit)
    # Output merges
    idx = start_idx
    for merge_commit in merges:
        if merge_commit.hexsha in written_shas:
            continue
        written_shas.add(merge_commit.hexsha)
        p1, p2 = merge_commit.parents
        base = get_merge_base(repo, p1, p2)
        if base is None:
            notes = "two initial commits"
        elif base.hexsha in (p1.hexsha, p2.hexsha):
            notes = "a parent is the base"
        else:
            notes = ""
        writer.write(
            f"{idx},{branch_ref.path},{merge_commit.hexsha},{p1.hexsha},{p2.hexsha},{notes}\n"
        )
        idx += 1
    return idx


def write_all_branches(repo: Repo, output_file: Path) -> None:
    """
    Discover all local + remote branches, deduplicate them by their head commit,
    find merges in each, and write to output_file in CSV format.

    Arguments:
        repo (Repo): The Git repository object.
        output_file (Path): The output file to write the CSV
    """
    with output_file.open("w", encoding="utf-8") as w:
        w.write("idx,branch_name,merge_commit,parent_1,parent_2,notes\n")
        # Filter references we care about
        references = [
            r
            for r in repo.references
            if r.path.startswith(("refs/heads/", "refs/remotes/"))
        ]
        # Deduplicate references by their HEAD commit
        seen_heads, filtered_refs = set(), []
        for ref in references:
            head_sha = ref.commit.hexsha
            if head_sha not in seen_heads:
                seen_heads.add(head_sha)
                filtered_refs.append(ref)
        # Write merges
        written_shas: Set[str] = set()
        idx = 1
        for ref in filtered_refs:
            idx = write_branch_merges(repo, ref, w, idx, written_shas)


def process_repo(
    org: str,
    repo: str,
    output_dir: Path,
    delete_local: bool,
) -> None:
    """
    Clone or reuse a local copy of 'org/repo' in 'cache_path', fetch PR branches,
    collect merges, and write them to <output_dir>/<org>/<repo>.csv.
    If delete_local is True, remove local clone after processing.

    Arguments:
        org (str): The GitHub organization.
        repo (str): The GitHub repository.
        output_dir (Path): The output directory for CSVs.
        delete_local (bool): If True, remove local clone after processing.
    """
    repo_name = f"{org}/{repo}"
    org_dir = output_dir / org
    org_dir.mkdir(parents=True, exist_ok=True)
    out_csv = org_dir / f"{repo}.csv"
    if out_csv.exists():
        logger.info(f"{repo_name:<30} SKIPPED -> already processed")
        return

    logger.info(f"{repo_name:<30} STARTED")

    local_dir = Path(os.getenv("REPOS_PATH", "repos")) / org / repo

    github_user, github_token = read_github_credentials()

    # If directory doesn't exist, clone; if it exists, reuse.
    if not local_dir.is_dir():
        local_dir.mkdir(parents=True, exist_ok=True)
        if github_user == "Bearer":
            clone_url = f"https://{github_token}@github.com/{org}/{repo}.git"
        else:
            clone_url = (
                f"https://{github_user}:{github_token}@github.com/{org}/{repo}.git"
            )
        try:
            rrepo = Repo.clone_from(clone_url, local_dir, multi_options=["--no-tags"])
        except GitCommandError:
            logger.info(f"{repo_name:<30} CLONE FAILED -> empty CSV")
            out_csv.write_text("idx,branch_name,merge_commit,parent_1,parent_2,notes\n")
            return
    else:
        # Already exists, open it
        try:
            rrepo = Repo(local_dir)
        except Exception:  # pylint: disable=broad-except
            logger.info(f"{repo_name:<30} BROKEN LOCAL REPO -> empty CSV")
            out_csv.write_text("idx,branch_name,merge_commit,parent_1,parent_2,notes\n")
            return

    fetch_pr_branches(rrepo)
    write_all_branches(rrepo, out_csv)
    logger.info(f"{repo_name:<30} DONE")

    # Optionally remove
    if delete_local:
        shutil.rmtree(local_dir, ignore_errors=True)


def main() -> None:
    """
    Main entry point. Parses arguments, reads repos from CSV via pandas,
    and processes them in parallel. Uses environment variable REPOS_PATH if set,
    otherwise defaults to /tmp for caching.
    """
    parser = argparse.ArgumentParser(
        description="Find 2-parent merge commits in GitHub repos."
    )
    parser.add_argument(
        "csv_file",
        help="Path to input CSV with a 'repository' column (org/repo).",
    )
    parser.add_argument(
        "output_dir",
        help="Directory for output CSVs (one per repository).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="If set, remove local clones after processing (default: keep).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=-1,
        help="Number of parallel threads to use (default: 8).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the CSV via pandas
    df = pd.read_csv(args.csv_file)
    if "repository" not in df.columns:
        sys.exit("CSV missing 'repository' column.")

    repos = df["repository"].tolist()
    logger.info(f"Found {len(repos)} repositories to process.")

    # Parallel processing of repositories
    num_workers = os.cpu_count() if args.threads == -1 else args.threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for repo_str in repos:
            parts = repo_str.split("/", maxsplit=1)
            if len(parts) != 2:
                logger.error(f"Invalid repository format: {repo_str}")
                continue
            org, repo = parts
            futures.append(
                executor.submit(
                    process_repo,
                    org,
                    repo,
                    output_dir,
                    args.delete,
                )
            )
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            _ = future.result()

    logger.info("All repositories processed.")


if __name__ == "__main__":
    main()
