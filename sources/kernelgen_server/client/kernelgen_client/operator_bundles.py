"""Deterministic Bundle identity and wire models; no server-side storage."""

from __future__ import annotations

import hashlib
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


OPERATOR_BUNDLE_FORMAT = "kernelgen.operator-bundle/v1"
OPERATOR_BUNDLE_MEDIA_TYPE = "application/vnd.kernelgen.operator-bundle.v1+tar"
DEFAULT_OPERATOR_BUNDLE_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_OPERATOR_BUNDLE_MAX_FILES = 10_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OperatorBundleError(ValueError):
    """An operator bundle is malformed or inconsistent."""


class OperatorBundleTooLarge(OperatorBundleError):
    """An operator bundle exceeds a configured storage limit."""


class GemsDefinitionSource(BaseModel):
    """Origin of an uploaded ABI, not a global FlagGems revision requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evaluator: Literal["flaggems"] = "flaggems"
    benchmark_level: Literal["core"] = "core"
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_files: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def safe_source_paths(self):
        for name, digest in self.source_files.items():
            path = PurePosixPath(name)
            if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != name or "\\" in name:
                raise ValueError("Definition source paths must be canonical repository-relative paths")
            if not _SHA256.fullmatch(digest):
                raise ValueError("Definition source digests must be SHA-256")
        return self


class OperatorBundleInfo(BaseModel):
    """Stable metadata returned for one installed operator bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    definition: str = Field(min_length=1)
    archive_format: str = OPERATOR_BUNDLE_FORMAT
    size_bytes: int = Field(ge=0)
    created_at: str = Field(min_length=1)


def normalize_operator_bundle_sha256(value: str) -> str:
    digest = value.removeprefix("sha256:")
    if not _SHA256.fullmatch(digest):
        raise OperatorBundleError("operator bundle SHA-256 must be 64 lowercase hex digits")
    return digest


def pack_operator_bundle(operator_dir: str | Path, output: BinaryIO) -> tuple[str, int]:
    """Write one deterministic uncompressed tar and return its digest and size."""

    root = Path(operator_dir).resolve()
    if not root.is_dir():
        raise OperatorBundleError(f"operator directory does not exist: {root}")
    files: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise OperatorBundleError(f"operator bundle cannot contain symlinks: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise OperatorBundleError(f"operator bundle contains a special file: {relative}")
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files.append((path, relative))
    if not files:
        raise OperatorBundleError("operator directory contains no files")

    output.seek(0)
    output.truncate(0)
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path, relative in files:
            stat = path.stat()
            member = tarfile.TarInfo(relative)
            member.size = stat.st_size
            member.mode = 0o755 if stat.st_mode & 0o111 else 0o644
            member.mtime = 0
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            with path.open("rb") as source:
                archive.addfile(member, source)

    size = output.tell()
    output.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: output.read(1024 * 1024), b""):
        digest.update(chunk)
    output.seek(0)
    return digest.hexdigest(), size


__all__ = [
    "DEFAULT_OPERATOR_BUNDLE_MAX_BYTES", "DEFAULT_OPERATOR_BUNDLE_MAX_FILES",
    "GemsDefinitionSource", "OPERATOR_BUNDLE_FORMAT", "OPERATOR_BUNDLE_MEDIA_TYPE",
    "OperatorBundleError", "OperatorBundleInfo", "OperatorBundleTooLarge",
    "normalize_operator_bundle_sha256", "pack_operator_bundle",
]
