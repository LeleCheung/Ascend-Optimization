"""Materialize a commit-pinned Git repository as a KB SourcePackage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml

from kernelgen.knowledge.models import GitSourceManifest, SourcePackage
from kernelgen.knowledge.validation import source_content_checksum


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _normalize_excludes(values: list[str] | None) -> list[str]:
    normalized = []
    for value in values or []:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError(f"invalid excluded source path: {value}")
        candidate = path.as_posix().rstrip("/")
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _is_excluded(path: str, excludes: list[str]) -> bool:
    return any(path == item or path.startswith(item + "/") for item in excludes)


def _entries(
    repo: Path,
    revision: str,
    snapshot: Path,
    excludes: list[str],
) -> list[dict]:
    raw = _git(repo, "ls-tree", "-rz", "--full-tree", "--long", revision)
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode, object_type, object_id, encoded_size = metadata.split()
        path = encoded_path.decode("utf-8")
        if _is_excluded(path, excludes):
            continue
        item = {
            "path": path,
            "mode": mode.decode("ascii"),
            "object_type": object_type.decode("ascii"),
            "object_id": object_id.decode("ascii"),
            "size": None,
            "sha256": "",
        }
        if item["object_type"] == "blob":
            source = snapshot / path
            if item["mode"] == "120000":
                data = os.readlink(source).encode("utf-8")
            else:
                data = source.read_bytes()
            git_size = int(encoded_size)
            if len(data) != git_size:
                raise ValueError(f"Git archive size mismatch: {path}")
            item["size"] = len(data)
            item["sha256"] = "sha256:" + hashlib.sha256(data).hexdigest()
        entries.append(item)
    return entries


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def snapshot(
    *,
    repo: Path,
    kb: Path,
    package_path: Path,
    name: str,
    revision: str,
    replace: bool,
    exclude_paths: list[str] | None = None,
) -> dict:
    repo = repo.resolve()
    kb = kb.resolve()
    package_path = package_path.resolve()
    resolved_revision = _git(repo, "rev-parse", f"{revision}^{{commit}}").decode().strip()
    object_format = _git(repo, "rev-parse", "--show-object-format").decode().strip()
    excludes = _normalize_excludes(exclude_paths)
    short_revision = resolved_revision[:12]
    destination = kb / "sources" / "content" / name / short_revision
    manifest_path = kb / "sources" / "manifests" / f"{name}-{short_revision}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=".source-", dir=destination.parent))
    staged_content = staging / "content"
    staged_content.mkdir()
    archive = staging / "source.tar"
    backup = destination.with_name(
        f".{destination.name}.backup-{uuid.uuid4().hex}"
    )
    installed = False
    try:
        archive.write_bytes(_git(repo, "archive", "--format=tar", resolved_revision))
        with tarfile.open(archive, "r:") as handle:
            handle.extractall(staged_content, filter="data")
        for excluded in excludes:
            target = (staged_content / excluded).resolve()
            target.relative_to(staged_content.resolve())
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
        entries = _entries(repo, resolved_revision, staged_content, excludes)
        manifest = GitSourceManifest(
            source_package=yaml.safe_load(
                package_path.read_text(encoding="utf-8")
            )["id"],
            revision=resolved_revision,
            object_format=object_format,
            entries=entries,
        )
        if destination.exists():
            if not replace:
                raise FileExistsError(
                    f"destination exists; pass --replace after review: {destination}"
                )
            os.replace(destination, backup)
        try:
            os.replace(staged_content, destination)
            installed = True
        except BaseException:
            if backup.exists():
                os.replace(backup, destination)
            raise

        raw_package = yaml.safe_load(package_path.read_text(encoding="utf-8"))
        previous_revision = raw_package.get("revision")
        raw_package.update(
            {
                "revision": resolved_revision,
                "content_root": (
                    f"kb://sources/content/{name}/{short_revision}"
                ),
                "checksum": source_content_checksum(destination),
                "search_mode": "searchable",
                "manifest_root": (
                    f"kb://sources/manifests/{name}-{short_revision}.json"
                ),
            }
        )
        if previous_revision != resolved_revision:
            raw_package["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        package = SourcePackage.model_validate(raw_package)
        _atomic_text(
            manifest_path,
            json.dumps(
                manifest.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )
        _atomic_text(
            package_path,
            yaml.safe_dump(
                package.model_dump(mode="json"),
                sort_keys=False,
                allow_unicode=True,
            ),
        )
        if backup.exists():
            shutil.rmtree(backup)
        return {
            "source_package": package.id,
            "revision": resolved_revision,
            "content_root": package.content_root,
            "manifest_root": package.manifest_root,
            "entries": len(entries),
            "blobs": sum(item["object_type"] == "blob" for item in entries),
            "gitlinks": sum(item["object_type"] == "commit" for item in entries),
            "excluded_paths": excludes,
            "checksum": package.checksum,
        }
    except BaseException:
        if installed and backup.exists():
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--kb", type=Path, default=Path("kb"))
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude one tracked path or subtree; may be repeated.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing snapshot directory.",
    )
    args = parser.parse_args()
    result = snapshot(
        repo=args.repo,
        kb=args.kb,
        package_path=args.package,
        name=args.name,
        revision=args.revision,
        replace=args.replace,
        exclude_paths=args.exclude,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
