"""Resolve a public FlagGems PR once; never run code from its checkout."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .source_profile import PROFILES, repository_profile

from kernelgen.data._atomic import atomic_write_json
from kernelgen.framework.local_state import file_lock


REPOSITORY = PROFILES[0].repository
REMOTE = repository_profile(REPOSITORY).remote
_PR_URL = re.compile(r"https://github\.com/([^/]+/[^/]+)/pull/([1-9][0-9]*)/?", re.IGNORECASE)
_SHA = r"^[0-9a-f]{40}$"


class PullRequestSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    repository: str = REPOSITORY
    number: int = Field(gt=0)
    base_sha: str = Field(pattern=_SHA)
    head_sha: str = Field(pattern=_SHA)
    merge_base_sha: str = Field(pattern=_SHA)

    @field_validator("repository")
    @classmethod
    def supported_repository(cls, value):
        return repository_profile(value).repository

    @property
    def url(self):
        return f"https://github.com/{self.repository}/pull/{self.number}"

    @property
    def remote(self):
        return repository_profile(self.repository).remote


def _parse_pull_request(url: str) -> tuple[str, int]:
    match = _PR_URL.fullmatch(url)
    if not match:
        raise ValueError("expected https://github.com/flagos-ai/FlagGems/pull/<number> or FlagGems-vllm PR")
    return repository_profile(match[1]).repository, int(match[2])


def pull_request_number(url: str) -> int:
    return _parse_pull_request(url)[1]


def _github(path: str):
    request = Request("https://api.github.com/" + path, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "KernelGen-PR-extractor",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except Exception as exc:
        # Do not include response bodies, proxy URLs or auth configuration.
        raise RuntimeError(f"GitHub metadata request failed ({type(exc).__name__})") from None


def resolve_pull_request(url: str) -> PullRequestSource:
    repository, number = _parse_pull_request(url)
    metadata = _github(f"repos/{repository}/pulls/{number}")
    if metadata["number"] != number or metadata["base"]["repo"]["full_name"].lower() != repository.lower():
        raise ValueError("GitHub returned a different PR repository or number")
    base, head = metadata["base"]["sha"], metadata["head"]["sha"]
    if not re.fullmatch(_SHA, base) or not re.fullmatch(_SHA, head):
        raise ValueError("GitHub returned invalid commit identities")
    comparison = _github(f"repos/{repository}/compare/{base}...{head}")
    return PullRequestSource(repository=repository, number=number, base_sha=base, head_sha=head,
                             merge_base_sha=comparison["merge_base_commit"]["sha"])


def _git(root: Path, *args: str, missing_ok: bool = False) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args],
            text=True, stderr=subprocess.PIPE, timeout=180,
        )
    except subprocess.CalledProcessError as exc:
        if missing_ok and exc.returncode == 1:
            return ""
        raise RuntimeError(f"PR source git {args[0]} failed; checkout retained") from None
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(f"PR source git {args[0]} failed ({type(exc).__name__}); checkout retained") from None


def verify_checkout(root: Path, revision: str, remote: str | None = None):
    remote = remote or REMOTE
    if _git(root, "rev-parse", "HEAD").strip() != revision:
        raise ValueError("PR checkout commit changed; use a new source workspace")
    if _git(root, "status", "--porcelain", "--untracked-files=all", "--ignored"):
        raise ValueError("PR checkout must remain clean, including untracked/ignored files")
    if _git(root, "remote", "get-url", "origin").strip() != remote:
        raise ValueError("PR checkout origin changed")
    entries = {}
    for entry in _git(root, "ls-files", "--stage", "-z").split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            entries[path] = metadata.split()[0]
    for path, mode in entries.items():
        if mode == "160000":
            raise ValueError("PR extraction does not follow repository submodules")
        if mode != "120000":
            continue
        link = root / path
        try:
            if link.readlink().is_absolute():
                raise ValueError("absolute link")
            target = link.resolve(strict=True).relative_to(root.resolve()).as_posix()
            if entries.get(target) not in {"100644", "100755"}:
                raise ValueError("target is not a tracked regular file")
        except (OSError, RuntimeError, ValueError):
            raise ValueError("PR extraction rejects unsafe repository symlinks") from None


def _checkout(root: Path, revision: str, remote: str | None = None):
    remote = remote or REMOTE
    if root.is_symlink():
        raise ValueError("PR checkout directory cannot be a symlink")
    if not root.exists():
        root.mkdir()
        _git(root, "init", "--quiet")
        _git(root, "remote", "add", "origin", remote)
    if not (root / ".git").is_dir():
        raise ValueError("refusing to overwrite an existing non-checkout directory")
    # A failed first fetch can leave an unborn repository. Never reset a HEAD.
    heads = _git(root, "rev-parse", "--verify", "--quiet", "HEAD", missing_ok=True)
    if heads:
        verify_checkout(root, revision, remote)
        return
    if _git(root, "status", "--porcelain", "--untracked-files=all", "--ignored"):
        raise ValueError("refusing to overwrite files in an incomplete PR checkout")
    if _git(root, "remote", "get-url", "origin").strip() != remote:
        raise ValueError("PR checkout origin changed")
    _git(root, "fetch", "--depth=1", "--no-tags", "origin", revision)
    _git(root, "checkout", "--detach", revision)
    verify_checkout(root, revision, remote)


def prepare_pull_request(url: str, workspace: str | Path) -> PullRequestSource:
    """Reuse frozen source.json on retries, even if the live PR has advanced."""
    repository, number = _parse_pull_request(url)
    root = Path(workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".source.lock"):
        receipt = root / "source.json"
        if receipt.exists():
            source = PullRequestSource.model_validate_json(receipt.read_text())
            if source.number != number or source.repository != repository:
                raise ValueError("source workspace belongs to a different PR")
        else:
            source = resolve_pull_request(url)
            atomic_write_json(receipt, source.model_dump(mode="json"))
        _checkout(root / "head", source.head_sha, source.remote)
        _checkout(root / "base", source.base_sha, source.remote)
        if not _git(root / "head", "rev-parse", "--verify", "--quiet", source.merge_base_sha + "^{commit}", missing_ok=True):
            _git(root / "head", "fetch", "--depth=1", "--no-tags", "origin", source.merge_base_sha)
        return source


def changed_paths(workspace: str | Path, source: PullRequestSource) -> list[str]:
    """Use the frozen merge base, not live master or a truncated HTTP file list."""
    paths = _git(Path(workspace) / "head", "diff", "--name-only", "-z",
                 source.merge_base_sha, source.head_sha, "--").split("\0")
    return [path for path in paths if path]
