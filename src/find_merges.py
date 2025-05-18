#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    python3 find_merges.py --repos repos.csv --output_file merges.csv --delete

This script finds 2-parent merge commits in a set of GitHub repositories and outputs them
to one consolidated CSV file.

**Key change**: If `MAX_NUM_MERGES` is increased and a partial CSV result is already on disk,
we simply collect additional merges until we reach the new limit, without discarding previously
collected merges.
"""

import os
import sys
from pathlib import Path
import hashlib
from typing import List, Optional, Set, Dict, Tuple

import pandas as pd
from git import Commit, GitCommandError, Repo
from loguru import logger

# Create a cache folder for merge results
CACHE_DIR = Path("merge_cache/merges")
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def read_github_credentials() -> tuple[str, str]:
    """
    Returns a tuple (username, token) for GitHub authentication:
      1) Reads from ~/.github-personal-access-token if present
            (first line = user, second line = token).
      2) Otherwise uses environment variable GITHUB_TOKEN (with user="Bearer").
      3) Exits if neither is available.
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


def get_merge_base(repo: Repo, c1: Commit, c2: Commit) -> Optional[Commit]:
    """
    Compute the nearest common ancestor (merge base) of two commits.
    If no common ancestor exists, return None.
    If the merge base is one of the parents, that is noted separately.
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


def get_commits_for_branch(repo: Repo, branch_ref, repo_slug: str) -> List[Commit]:
    """
    Retrieve and cache the list of commits for the given branch reference.
    Results are cached to a CSV file in 'merge_cache'.
    """
    branch_hash = hashlib.md5(branch_ref.path.encode("utf-8")).hexdigest()
    commit_cache_file = (
        CACHE_DIR / f"{repo_slug.replace('/', '_')}_{branch_hash}_commits.csv"
    )

    try:
        if commit_cache_file.exists():
            try:
                df_commits = pd.read_csv(commit_cache_file)
                commit_hexshas = df_commits["commit_hexsha"].tolist()
                commits = [repo.commit(sha) for sha in commit_hexshas]
            except Exception as e:
                logger.error(
                    f"Error reading commit cache file {commit_cache_file}: {e}"
                )
                commits = list(repo.iter_commits(branch_ref.path))
        else:
            commits = list(repo.iter_commits(branch_ref.path))
            try:
                df_commits = pd.DataFrame(
                    {"commit_hexsha": [c.hexsha for c in commits]}
                )
                df_commits.to_csv(commit_cache_file, index=False)
            except Exception as e:
                logger.error(
                    f"Error writing commit cache file {commit_cache_file}: {e}"
                )
        return commits
    except GitCommandError:
        return []


def collect_branch_merges(  # pylint: disable=too-many-locals
    repo: Repo, branch_ref, repo_slug: str, written_shas: Set[str], max_num_merges: int
) -> List[Dict[str, str]]:
    """
    For the given branch reference, find all 2-parent merge commits.
    Return a list of CSV rows for the branch. (Without duplicates in 'written_shas'.)
    """
    # Branch-level merge cache
    branch_hash = hashlib.md5(branch_ref.path.encode("utf-8")).hexdigest()
    merges_cache_file = (
        CACHE_DIR / f"{repo_slug.replace('/', '_')}_{branch_hash}_merges.csv"
    )
    if merges_cache_file.exists():
        df_cached = pd.read_csv(merges_cache_file)
        result_rows = []
        for _, cached in df_cached.iterrows():
            if len(written_shas) >= max_num_merges:
                break
            sha = cached["merge_commit"]
            if sha in written_shas:
                continue
            written_shas.add(sha)
            result_rows.append(
                {
                    "repository": cached["repository"],
                    "branch_name": cached["branch_name"],
                    "merge_commit": cached["merge_commit"],
                    "parent_1": cached["parent_1"],
                    "parent_2": cached["parent_2"],
                    "notes": cached["notes"],
                }
            )
        return result_rows

    # Will hold all merges for caching
    rows_full: List[Dict[str, str]] = []

    rows: List[Dict[str, str]] = []
    commits = get_commits_for_branch(repo, branch_ref, repo_slug)

    for commit in commits:
        if len(written_shas) >= max_num_merges:
            break
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
                "repository": repo_slug,
                "branch_name": branch_ref.path,
                "merge_commit": commit.hexsha,
                "parent_1": p1.hexsha,
                "parent_2": p2.hexsha,
                "notes": notes,
            }
            rows_full.append(info)
            rows.append(info)
    pd.DataFrame(
        rows_full,
        columns=[
            "repository",
            "branch_name",
            "merge_commit",
            "parent_1",
            "parent_2",
            "notes",
        ],
    ).to_csv(merges_cache_file, index=False)
    return rows


def get_filtered_refs(repo: Repo, repo_slug: str) -> List:
    """
    Retrieve filtered branch references (local and remote) for a repository.
    Uses a CSV cache to avoid recomputation. Deduplicates by commit head.
    """
    filtered_refs_cache_file = (
        CACHE_DIR / f"{repo_slug.replace('/', '_')}_filtered_refs.csv"
    )
    filtered_refs = []

    if filtered_refs_cache_file.exists():
        try:
            df_refs = pd.read_csv(filtered_refs_cache_file)
            ref_paths = df_refs["ref_path"].tolist()
            all_refs = {r.path: r for r in repo.references}
            for rp in ref_paths:
                if rp in all_refs:
                    filtered_refs.append(all_refs[rp])
        except Exception as e:
            logger.error(
                f"Error reading filtered references cache file {filtered_refs_cache_file}: {e}"
            )
    else:
        references = [
            r
            for r in repo.references
            if r.path.startswith(("refs/heads/", "refs/remotes/"))
        ]
        seen_heads = set()
        for ref in references:
            head_sha = ref.commit.hexsha
            if head_sha not in seen_heads:
                seen_heads.add(head_sha)
                filtered_refs.append(ref)
        # Save to cache
        try:
            df_refs = pd.DataFrame({"ref_path": [r.path for r in filtered_refs]})
            df_refs.to_csv(filtered_refs_cache_file, index=False)
        except Exception as e:
            logger.error(
                f"Error writing filtered references cache file {filtered_refs_cache_file}: {e}"
            )
    return filtered_refs


def get_repo_path(repo_slug: str) -> Path:
    """
    Return the local path where the repository should be cloned.
    """
    repos_cache = Path(os.getenv("REPOS_PATH", "repos"))
    return repos_cache / f"{repo_slug}"


def get_repo(repo_slug: str, log: bool = False) -> Repo:
    """
    Clone or reuse a local copy of 'org/repo' under repos_cache/org/repo.
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


def collect_all_merges(
    repo: Repo,
    repo_slug: str,
    existing_shas: Optional[Set[str]] = None,
    max_num_merges: int = 100,
) -> Tuple[pd.DataFrame, bool]:
    """
    Discover all filtered branch references, find merge commits in each,
    and return a consolidated DataFrame. Uses 'existing_shas' to skip
    merges already found in a previous run (if provided).
    """
    rows: List[Dict[str, str]] = []
    filtered_refs = get_filtered_refs(repo, repo_slug)

    # If we already have some merges, start 'written_shas' with them
    written_shas = existing_shas if existing_shas is not None else set()
    total_merges = len(written_shas)

    for ref in filtered_refs:
        if total_merges >= max_num_merges:
            return pd.DataFrame(rows), True
        branch_merges = collect_branch_merges(
            repo, ref, repo_slug, written_shas, max_num_merges
        )
        rows.extend(branch_merges)
        total_merges += len(branch_merges)

    return pd.DataFrame(rows), False


def get_merges(  # pylint: disable=too-many-branches
    repo: Repo, repo_slug: str, out_dir: Path, max_num_merges: int = 100
) -> pd.DataFrame:
    """
    Clone/reuse a local copy of 'org/repo', fetch PR branches,
    and collect merge commits. If an existing CSV is found, we:
      - read it in
      - if it already has >= MAX_NUM_MERGES merges, return as-is
      - otherwise collect additional merges until reaching MAX_NUM_MERGES
      - write out the combined result
    """
    full_results_path = out_dir / f"{repo_slug}.csv"
    # Load existing results (if any)
    if full_results_path.exists():
        try:
            existing_df = pd.read_csv(full_results_path, index_col="merge_idx")
        except pd.errors.EmptyDataError:
            existing_df = pd.DataFrame()
        return existing_df.head(max_num_merges)

    results_partial_path = out_dir / f"{repo_slug}_partial.csv"
    if results_partial_path.exists():
        try:
            existing_df = pd.read_csv(results_partial_path, index_col="merge_idx")
        except pd.errors.EmptyDataError:
            existing_df = pd.DataFrame()
        if len(existing_df) >= max_num_merges:
            # Already have enough merges for this repo
            return existing_df.head(max_num_merges)

    # Determine which merges we already have
    if existing_df is not None and not existing_df.empty:
        existing_shas = set(existing_df["merge_commit"])
    else:
        existing_shas = set()

    # Collect new merges (skipping existing_shas)
    new_df, breaked = collect_all_merges(
        repo, repo_slug, existing_shas=existing_shas, max_num_merges=max_num_merges
    )

    if not new_df.empty:
        # Combine the new merges with any existing ones
        if existing_df is not None and not existing_df.empty:
            combined_df = pd.concat(
                [existing_df, new_df], ignore_index=True
            ).drop_duplicates(subset=["merge_commit"])
        else:
            combined_df = new_df
    else:
        # No new merges found, so just use existing
        combined_df = existing_df if existing_df is not None else pd.DataFrame()

    if breaked:
        # Save the partial results
        combined_df.to_csv(results_partial_path, index_label="merge_idx")
        logger.info(f"Partial results saved to {results_partial_path}")
    else:
        # Remove the partial results if they exist
        if results_partial_path.exists():
            results_partial_path.unlink()
            logger.info(f"Removed partial results file {results_partial_path}")
        # Write out the final DataFrame
        final_results_path = out_dir / f"{repo_slug}.csv"
        combined_df.index.name = "merge_idx"
        combined_df.to_csv(final_results_path, index_label="merge_idx")

    # Truncate to max_num_merges if needed
    if len(combined_df) > max_num_merges:
        combined_df = combined_df.head(max_num_merges)

    return combined_df
