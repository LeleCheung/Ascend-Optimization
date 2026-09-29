#!/usr/bin/env python3
"""Batch PR submission: read BatchSimpleOpt results and submit PRs in parallel.

Usage:
    cd /share-evpfs/tj/workspace/code-integration/kernelgen
    python3 -u examples/batch_pr/run_example.py \
        --batch-dir runs/batch_simple_opt/v4 \
        --target-repo flagos-ai/FlagGems-Experimental \
        --vendor ascend
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.workflows.batch_pr_workflow import BatchPRWorkflow


def _setup_flaggems_dir(flaggems_dir: Path, target_repo: str) -> None:
    """Clone target_repo and set up fork remote if flaggems_dir doesn't exist."""
    repo_name = target_repo.split("/")[-1]

    # Get current gh user
    r = subprocess.run(["gh", "api", "user", "--jq", ".login"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("gh CLI not logged in. Run: gh auth login")
    gh_user = r.stdout.strip()
    fork_slug = f"{gh_user}/{repo_name}"

    # Ensure fork exists
    fork_check = subprocess.run(
        ["gh", "api", f"repos/{fork_slug}", "--jq", ".full_name"],
        capture_output=True, text=True,
    )
    if fork_check.returncode != 0:
        print(f"  [setup] Forking {target_repo} to {fork_slug}...")
        subprocess.run(["gh", "repo", "fork", target_repo, "--clone=false"], check=True)

    # Clone upstream
    print(f"  [setup] Cloning {target_repo} to {flaggems_dir}...")
    subprocess.run(
        ["git", "clone", f"https://github.com/{target_repo}.git", str(flaggems_dir)],
        check=True,
    )

    # Set origin to fork, add upstream
    fork_url = f"https://github.com/{fork_slug}.git"
    subprocess.run(["git", "remote", "set-url", "origin", fork_url], cwd=str(flaggems_dir), check=True)
    subprocess.run(["git", "remote", "add", "upstream", f"https://github.com/{target_repo}.git"], cwd=str(flaggems_dir), check=True)
    subprocess.run(["git", "fetch", "upstream", "--no-tags"], cwd=str(flaggems_dir), check=True)
    print(f"  [setup] origin -> {fork_url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--flaggems-dir", type=Path, default=None,
                        help="Path to FlagGems repo. Auto-cloned from --target-repo if absent.")
    parser.add_argument("--target-repo", default="flagos-ai/FlagGems-Experimental")
    parser.add_argument("--vendor", default="ascend")
    parser.add_argument("--base-branch", default="master")
    parser.add_argument("--no-draft", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--model", default=os.environ.get("MODEL", "deepseek-v4-pro[1m]"))
    parser.add_argument("--base-url", default=os.environ.get("ANTHROPIC_BASE_URL"))
    parser.add_argument("--auth-token", default=os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    args = parser.parse_args()

    batch_dir = args.batch_dir.resolve()
    assert batch_dir.exists(), f"batch-dir not found: {batch_dir}"

    # Derive default flaggems_dir from target_repo name next to batch_dir's workspace root
    if args.flaggems_dir is None:
        repo_name = args.target_repo.split("/")[-1]
        flaggems_dir = batch_dir.parents[2] / repo_name
    else:
        flaggems_dir = args.flaggems_dir.resolve()

    if not flaggems_dir.exists():
        _setup_flaggems_dir(flaggems_dir, args.target_repo)

    def make_rt(path):
        return ClaudeRuntime(
            workspace=path,
            model=args.model,
            base_url=args.base_url,
            auth_token=args.auth_token,
            timeout=args.timeout,
            idle_timeout=args.timeout // 2,
        )

    wf = BatchPRWorkflow(cwd=str(batch_dir), runtime_factory=make_rt)
    out = wf.run({
        "batch_dir": str(batch_dir),
        "flaggems_dir": str(flaggems_dir),
        "vendor": args.vendor,
        "target_repo": args.target_repo,
        "base_branch": args.base_branch,
        "draft": not args.no_draft,
        "max_workers": args.max_workers,
    })

    print(f"\n{'='*60}")
    print(f"  {out.summary}")
    for r in sorted(out.results, key=lambda x: x.definition_name):
        line = f"  {r.status:10} {r.definition_name}"
        if r.pr_url:
            line += f"  -> {r.pr_url}"
        elif r.skip_reason:
            line += f"  ({r.skip_reason})"
        print(line)
    print(f"{'='*60}")

    failed = [r for r in out.results if r.status not in ("SUBMITTED", "SKIPPED")]
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
