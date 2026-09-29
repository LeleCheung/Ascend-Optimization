"""Workspace base class and implementations."""

import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class Workspace(ABC):
    """Base class for workspace management. Subclasses implement _create/_cleanup."""

    def __init__(self):
        self._allocated: dict[str, Path] = {}

    def allocate(self, name: str) -> Path:
        path = self._create(name)
        self._allocated[name] = path
        return path

    def deallocate(self, name: str) -> None:
        path = self._allocated.pop(name, None)
        if path:
            self._cleanup(path)

    def cleanup_all(self) -> None:
        for name in list(self._allocated):
            self.deallocate(name)

    @abstractmethod
    def _create(self, name: str) -> Path:
        ...

    @abstractmethod
    def _cleanup(self, path: Path) -> None:
        ...


class GitWorktree(Workspace):
    def __init__(
        self,
        repo_dir: Path | str,
        base_branch: str = "master",
        branch_prefix: str = "kgrunner",
        worktree_dir: Path | str = ".worktrees",
    ):
        super().__init__()
        self.repo_dir = Path(repo_dir).resolve()
        self.base_branch = base_branch
        self.branch_prefix = branch_prefix
        wt = Path(worktree_dir)
        self.worktree_dir = wt if wt.is_absolute() else self.repo_dir / wt

    def _create(self, name: str) -> Path:
        branch_name = f"{self.branch_prefix}/{name}"
        wt_path = self.worktree_dir / name
        self.worktree_dir.mkdir(parents=True, exist_ok=True)

        self._force_remove(wt_path, branch_name)

        subprocess.run(
            ["git", "fetch", "origin", self.base_branch],
            cwd=str(self.repo_dir),
            capture_output=True,
        )

        subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, str(wt_path), "FETCH_HEAD"],
            cwd=str(self.repo_dir),
            capture_output=True,
            check=True,
        )
        logger.info("Created worktree: %s -> %s", branch_name, wt_path)
        return wt_path

    def _cleanup(self, path: Path) -> None:
        path = Path(path)
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=str(self.repo_dir),
            capture_output=True,
        )
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(self.repo_dir),
            capture_output=True,
        )

    def _force_remove(self, wt_path: Path, branch_name: str) -> None:
        # Find and remove any existing worktree using this branch
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(self.repo_dir),
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            for block in result.stdout.split("\n\n"):
                if f"branch refs/heads/{branch_name}" in block:
                    lines = block.strip().split("\n")
                    if lines:
                        old_path = lines[0].replace("worktree ", "")
                        subprocess.run(
                            ["git", "worktree", "remove", "--force", old_path],
                            cwd=str(self.repo_dir),
                            capture_output=True,
                        )

        if wt_path.exists():
            shutil.rmtree(wt_path, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(self.repo_dir),
            capture_output=True,
        )
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=str(self.repo_dir),
            capture_output=True,
        )


class TempDir(Workspace):
    def __init__(
        self,
        base_dir: Path | str = "/tmp/kgrunner_workspaces",
        seed_files: list[tuple[str, str | Path]] | None = None,
    ):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.seed_files = seed_files or []

    def _create(self, name: str) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=self.base_dir))
        for dst_name, src in self.seed_files:
            src = Path(src)
            dst = path / dst_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        return path

    def _cleanup(self, path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)


class Directory(Workspace):
    def __init__(self, path: Path | str):
        super().__init__()
        self._path = Path(path).resolve()

    def _create(self, name: str) -> Path:
        return self._path

    def _cleanup(self, path: Path) -> None:
        pass
