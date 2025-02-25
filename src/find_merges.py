#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    python3 find_merges.py --repos repos.csv --output_file merges.csv --delete

This script finds 2-parent merge commits in a set of GitHub repositories and outputs them
to one consolidated CSV file.

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
   - Return merge rows in CSV format.
4. Write all rows to a single output CSV file.
5. If the --delete flag is set, remove the local clone after processing.

**Parallelization**:
- Processes up to n_cpus repositories in parallel (can be changed via --threads).

**Authentication**:
- Uses a GitHub token for cloning; see below for how credentials are read.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Set, Dict

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
        RuntimeError: If neither ~/.github-personal-access-token nor GITHUB_TOKEN is available.

    Returns:
        tuple[str, str]
            A tuple (username, token) for GitHub authentication.
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

    Arguments:
        repo: Repo
            A GitPython Repo object.
    """
    try:
        repo.remotes.origin.fetch(refspec="refs/pull/*/head:refs/remotes/origin/pull/*")
    except GitCommandError:
        # Some repos may not have PR refs; ignore errors
        pass


def get_merge_base(repo: Repo, c1: Commit, c2: Commit) -> Optional[Commit]:
    """
    Compute the nearest common ancestor (merge base) of two commits.
    If no common ancestor exists, return None.
    If the merge base is one of the parents, that is noted separately.

    Arguments:
        repo: Repo
            A GitPython Repo object.
        c1: Commit
            The first commit.
        c2: Commit
            The second commit.

    Raises:
        RuntimeError: If the same commit is passed twice.

    Returns:
        Optional[Commit]
            The merge base commit or None if no common ancestor.
    """
    if c1.hexsha == c2.hexsha:
        raise RuntimeError(f"Same commit passed twice: {c1.hexsha}")
    h1 = list(repo.iter_commits(c1))
    h1.reverse()
    h2 = list(repo.iter_commits(c2))
    h2.reverse()
    s1 = {x.hexsha for x in h1}
    s2 = {x.hexsha for x in h2}
    if c2.hexsha in s1:
        return c2
    if c1.hexsha in s2:
        return c1
    length = min(len(h1), len(h2))
    common_prefix = 0
    for i in range(length):
        if h1[i].hexsha == h2[i].hexsha:
            common_prefix += 1
        else:
            break
    return None if not common_prefix else h1[common_prefix - 1]


def collect_branch_merges(
    repo: Repo, branch_ref, repo_identifier: str, written_shas: Set[str]
) -> List[Dict[str, str]]:
    """
    For the given branch reference, find all 2-parent merge commits.
    Returns a list of CSV rows (without an index) for the branch.
    Columns: repository, branch_name, merge_commit, parent_1, parent_2, notes

    Arguments:
        repo: Repo
            A GitPython Repo object.
        branch_ref: Reference
            A GitPython Reference object for the branch.
        repo_identifier: str
            The repository identifier (org/repo).
        written_shas: Set[str]
            A set of written commit SHAs to avoid duplicates.

    Returns:
        List[Dict[str, str]]
            A list of CSV rows for the
    """
    rows: List[Dict[str, str]] = []
    try:
        commits = list(repo.iter_commits(branch_ref.path))
    except GitCommandError:
        return rows
    for commit in commits:
        if len(commit.parents) == 2:
            if commit.hexsha in written_shas:
                continue
            written_shas.add(commit.hexsha)
            p1, p2 = commit.parents
            base = get_merge_base(repo, p1, p2)
            if base is None:
                notes = "two initial commits"
            elif base.hexsha in (p1.hexsha, p2.hexsha):
                notes = "a parent is the base"
            else:
                notes = ""
            info = {
                "repository": repo_identifier,
                "branch_name": branch_ref.path,
                "merge_commit": commit.hexsha,
                "parent_1": p1.hexsha,
                "parent_2": p2.hexsha,
                "notes": notes,
            }
            rows.append(info)
    return rows


def collect_all_branches(repo: Repo, repo_identifier: str) -> pd.DataFrame:
    """
    Discover all local and remote branches, deduplicate them by their head commit,
    find merge commits in each, and return a list of CSV rows.

    Arguments:
        repo: Repo
            A GitPython Repo object.
        repo_identifier: str
            The repository identifier (org/repo).

    Returns:
        pd.DataFrame
            A DataFrame with columns: repository, branch_name, merge_commit,
            parent_1, parent_2,
    """
    rows: List[Dict[str, str]] = []
    references = [
        r
        for r in repo.references
        if r.path.startswith(("refs/heads/", "refs/remotes/"))
    ]
    seen_heads = set()
    filtered_refs = []
    for ref in references:
        head_sha = ref.commit.hexsha
        if head_sha not in seen_heads:
            seen_heads.add(head_sha)
            filtered_refs.append(ref)
    written_shas: Set[str] = set()
    for ref in filtered_refs:
        rows.extend(collect_branch_merges(repo, ref, repo_identifier, written_shas))
    df = pd.DataFrame(rows)
    return df


def get_repo_path(repo_slug: str) -> Path:
    """
    Return the local path where the repository should be cloned.

    Arguments:
        org: str
            The organization name.
        repo_name: str
            The repository name.

    Returns:
        Path
            The local path where the repository should be cloned.
    """
    repos_cache = Path(os.getenv("REPOS_PATH", "repos"))
    return repos_cache / f"{repo_slug}"


def get_repo(repo_slug: str, log: bool = False) -> Repo:
    """
    Clone or reuse a local copy of 'org/repo_name' under repos_cache/org/repo_name.
    Returns a GitPython Repo object.

    Arguments:
        org: str
            The organization name.
        repo_name: str
            The repository name.
        log: bool
            If True, log cloning/reusing messages.

    Raises:
        GitCommandError: If the repository cannot be cloned.

    Returns:
        Repo
            A GitPython Repo object for the
    """
    repo_dir = get_repo_path(repo_slug)
    github_user, github_token = read_github_credentials()

    if not repo_dir.is_dir():
        if log:
            logger.info(f"Cloning {repo_slug} into {repo_dir}...")
        repo_dir.mkdir(parents=True, exist_ok=True)
        if github_user == "Bearer":
            clone_url = f"https://{github_token}@github.com/{repo_slug}.git"
        else:
            clone_url = (
                f"https://{github_user}:{github_token}@github.com/{repo_slug}.git"
            )
        try:
            os.environ["GIT_TERMINAL_PROMPT"] = "0"
            os.environ["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes"
            repo = Repo.clone_from(clone_url, repo_dir, multi_options=["--no-tags"])
            repo.remote().fetch()
            repo.remote().fetch("refs/pull/*/head:refs/remotes/origin/pull/*")
            return repo
        except GitCommandError as e:
            logger.error(f"Failed to clone {repo_slug}: {e}")
            raise
    else:
        if log:
            logger.info(f"Reusing existing repo {repo_slug} at {repo_dir}")
        return Repo(str(repo_dir))


def get_merges(repo: Repo, repo_slug: str, out_dir: Path) -> pd.DataFrame:
    """
    Clone or reuse a local copy of 'org/repo', fetch PR branches,
    collect merge commit rows, and return them.
    If delete_local is True, remove the local clone after processing.

    Arguments:
        org: str
            The organization name.
        repo: str
            The repository name.
        out_dir: Path
            The output directory where the final CSV file will be saved.

    Returns:
        pd.DataFrame
            A DataFrame with columns: repository, branch_name, merge_commit,
            parent_1, parent_2,
    """
    results_path = out_dir / f"{repo_slug}.csv"
    if results_path.exists():
        return pd.read_csv(results_path, index_col="idx")
    logger.info(f"{repo_slug:<30} STARTED")

    fetch_pr_branches(repo)
    df = collect_all_branches(repo, repo_slug)
    logger.info(f"{repo_slug:<30} DONE")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.index.name = "idx"
    df.to_csv(results_path, index_label="idx")
    return df
