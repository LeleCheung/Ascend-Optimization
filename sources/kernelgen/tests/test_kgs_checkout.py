"""The local Server start keeps the exact, clean KGS checkout invariant."""

from pathlib import Path
import subprocess

import pytest

from kernelgen.cli import server


def _repository(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()

    def git(*args, cwd=source):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    git("init", "-b", "fixture")
    (source / "fixture.txt").write_text("fixture\n")
    git("add", "fixture.txt")
    git("-c", "user.name=Test", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture")
    return source, git, git("rev-parse", "HEAD")


def test_prepare_checkout_reuses_locked_clean_source(tmp_path):
    source, git, commit = _repository(tmp_path)
    root = tmp_path / "installed"
    for _ in range(2):
        server._prepare_checkout(root, repository=str(source), release=None, branch=None, commit=commit)
        assert git("rev-parse", "HEAD", cwd=root) == commit
        assert not git("status", "--porcelain", cwd=root)


@pytest.mark.parametrize("conflict", ["dirty", "commit", "origin"])
def test_prepare_checkout_preserves_conflicting_existing_source(tmp_path, conflict):
    source, git, commit = _repository(tmp_path)
    root = tmp_path / "installed"
    server._prepare_checkout(root, repository=str(source), release=None, branch=None, commit=commit)
    if conflict == "dirty":
        (root / "fixture.txt").write_text("user change\n")
    elif conflict == "commit":
        git("-c", "user.name=Test", "-c", "user.email=fixture@example.invalid",
            "commit", "--allow-empty", "-m", "user commit", cwd=root)
    else:
        git("remote", "set-url", "origin", str(tmp_path / "other"), cwd=root)
    before = (git("rev-parse", "HEAD", cwd=root), git("status", "--porcelain", cwd=root),
              git("remote", "get-url", "origin", cwd=root))
    with pytest.raises(ValueError):
        server._prepare_checkout(root, repository=str(source), release=None, branch=None, commit=commit)
    assert before == (git("rev-parse", "HEAD", cwd=root), git("status", "--porcelain", cwd=root),
                      git("remote", "get-url", "origin", cwd=root))
