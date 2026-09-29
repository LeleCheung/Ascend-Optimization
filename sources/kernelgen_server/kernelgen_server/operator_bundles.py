"""Server-side Bundle validation and atomic storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from shutil import rmtree

from kernelgen_client.operator_bundles import (
    DEFAULT_OPERATOR_BUNDLE_MAX_BYTES, DEFAULT_OPERATOR_BUNDLE_MAX_FILES,
    GemsDefinitionSource, OPERATOR_BUNDLE_FORMAT, OPERATOR_BUNDLE_MEDIA_TYPE,
    OperatorBundleError, OperatorBundleInfo, OperatorBundleTooLarge,
    normalize_operator_bundle_sha256, pack_operator_bundle,
)
from .catalog import Catalog
from .protocol.schema import Definition

def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OperatorBundleStore:
    """Validate and atomically install content-addressed operator bundles."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_bytes: int = DEFAULT_OPERATOR_BUNDLE_MAX_BYTES,
        max_files: int = DEFAULT_OPERATOR_BUNDLE_MAX_FILES,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("operator bundle max_bytes must be positive")
        if max_files <= 0:
            raise ValueError("operator bundle max_files must be positive")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_files = max_files
        self._lock = threading.Lock()

    def _target(self, digest: str) -> Path:
        return self.root / normalize_operator_bundle_sha256(digest)

    @staticmethod
    def _metadata_path(target: Path) -> Path:
        return target / ".bundle.json"

    def get(self, digest: str) -> OperatorBundleInfo | None:
        target = self._target(digest)
        if target.is_symlink():
            raise OperatorBundleError(f"operator bundle path cannot be a symlink: {target.name}")
        if not target.is_dir():
            return None
        metadata = self._metadata_path(target)
        if metadata.is_symlink() or not metadata.is_file():
            raise OperatorBundleError(f"operator bundle metadata is missing: {target.name}")
        try:
            info = OperatorBundleInfo.model_validate_json(
                metadata.read_text(encoding="utf-8")
            )
        except (UnicodeError, ValueError) as exc:
            raise OperatorBundleError(
                f"operator bundle metadata is invalid: {target.name}"
            ) from exc
        if info.sha256 != target.name or info.bundle_id != f"sha256:{target.name}":
            raise OperatorBundleError(f"operator bundle metadata is inconsistent: {target.name}")
        if not (target / "manifest.json").is_file():
            raise OperatorBundleError(f"operator bundle manifest is missing: {target.name}")
        return info

    def catalog_path(self, digest: str) -> Path:
        target = self._target(digest)
        if self.get(digest) is None:
            raise KeyError(f"operator bundle not found: {target.name}")
        return target

    def describe(self) -> dict[str, object]:
        return {
            "enabled": True,
            "archive_format": OPERATOR_BUNDLE_FORMAT,
            "media_type": OPERATOR_BUNDLE_MEDIA_TYPE,
            "max_bytes": self.max_bytes,
            "max_files": self.max_files,
            "evaluation_binding": True,
            "evaluators": ["native", "flaggems"],
        }

    def install(
        self,
        archive_path: str | Path,
        expected_sha256: str,
    ) -> tuple[OperatorBundleInfo, bool]:
        archive = Path(archive_path)
        expected = normalize_operator_bundle_sha256(expected_sha256)
        size = archive.stat().st_size
        if size > self.max_bytes:
            raise OperatorBundleTooLarge(
                f"operator bundle exceeds {self.max_bytes} bytes"
            )
        if _hash_file(archive) != expected:
            raise OperatorBundleError("operator bundle SHA-256 does not match request path")

        with self._lock:
            existing = self.get(expected)
            if existing is not None:
                return existing, False
            staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
            try:
                operator_root = staging / "operator"
                operator_root.mkdir()
                self._extract(archive, operator_root)
                definition = self._load_definition(operator_root)
                manifest = {
                    "api_version": "v6.2", "name": f"uploaded-{expected[:12]}",
                    "evaluator": "native", "layout": "per-operator",
                    "source_format": OPERATOR_BUNDLE_FORMAT, "bundle_id": f"sha256:{expected}",
                }
                if definition.api_version == "v6.0":
                    members = {path.relative_to(operator_root).as_posix() for path in operator_root.rglob("*")}
                    if members != {"definition.json", "adapter.json"}:
                        raise OperatorBundleError("Gems Definition bundles contain only definition.json and adapter.json")
                    source = GemsDefinitionSource.model_validate_json((operator_root / "adapter.json").read_text())
                    definitions = staging / "definitions"
                    definitions.mkdir()
                    (operator_root / "definition.json").replace(definitions / f"{definition.name}.json")
                    (operator_root / "adapter.json").unlink()
                    operator_root.rmdir()
                    manifest.update(api_version="v6.0", evaluator="flaggems", layout="flat",
                                    benchmark_level="core", definition_source=source.model_dump(mode="json"))
                else:
                    ops_root = staging / "ops"
                    ops_root.mkdir()
                    operator_root.replace(ops_root / definition.name)
                (staging / "manifest.json").write_text(
                    json.dumps(
                        manifest,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                Catalog(staging).load(definition.name)
                info = OperatorBundleInfo(
                    bundle_id=f"sha256:{expected}",
                    sha256=expected,
                    definition=definition.name,
                    size_bytes=size,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                self._metadata_path(staging).write_text(
                    info.model_dump_json(indent=2) + "\n",
                    encoding="utf-8",
                )
                target = self._target(expected)
                try:
                    os.replace(staging, target)
                except OSError:
                    concurrent = self.get(expected)
                    if concurrent is not None:
                        return concurrent, False
                    raise
                return info, True
            except OperatorBundleError:
                raise
            except (OSError, tarfile.TarError, UnicodeError, ValueError) as exc:
                raise OperatorBundleError(f"invalid operator bundle: {exc}") from exc
            finally:
                if staging.exists():
                    rmtree(staging)

    @staticmethod
    def _load_definition(operator_root: Path) -> Definition:
        path = operator_root / "definition.json"
        if not path.is_file():
            raise OperatorBundleError("operator bundle is missing definition.json")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise OperatorBundleError(f"invalid definition.json: {exc}") from exc
        adapter = isinstance(raw, dict) and raw.get("api_version") == "v6.0" and (operator_root / "adapter.json").is_file()
        if not isinstance(raw, dict) or (raw.get("api_version") != "v6.2" and not adapter):
            raise OperatorBundleError("uploaded operator definition must use api_version=v6.2")
        if not isinstance(raw.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", raw["name"]):
            raise OperatorBundleError("operator Definition name must be a safe single path component")
        if adapter and {"reference", "reference_device", "source_policy_id"} & raw.keys():
            raise OperatorBundleError("Gems Definition bundles must not supply a Native reference or source policy")
        try:
            return Definition.model_validate(raw)
        except ValueError as exc:
            raise OperatorBundleError(f"invalid definition.json: {exc}") from exc

    def _extract(self, archive_path: Path, destination: Path) -> None:
        total_size = 0
        entry_count = 0
        seen: set[str] = set()
        try:
            archive = tarfile.open(archive_path, mode="r:")
        except tarfile.TarError as exc:
            raise OperatorBundleError(
                f"operator bundle must be an uncompressed tar: {exc}"
            ) from exc
        with archive:
            found_member = False
            for member in archive:
                found_member = True
                entry_count += 1
                if entry_count > self.max_files:
                    raise OperatorBundleTooLarge(
                        f"operator bundle exceeds {self.max_files} entries"
                    )
                member_name = member.name
                while member_name.startswith("./"):
                    member_name = member_name[2:]
                if member_name in {"", "."}:
                    if member.isdir():
                        continue
                    raise OperatorBundleError(
                        f"unsafe operator bundle path: {member.name}"
                    )
                relative = PurePosixPath(member_name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or relative.as_posix() != member_name.rstrip("/")
                ):
                    raise OperatorBundleError(f"unsafe operator bundle path: {member.name}")
                normalized_name = relative.as_posix()
                if normalized_name in seen:
                    raise OperatorBundleError(f"duplicate operator bundle path: {member.name}")
                seen.add(normalized_name)
                target = (destination / Path(*relative.parts)).resolve()
                if not target.is_relative_to(destination):
                    raise OperatorBundleError(f"operator bundle path escapes root: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise OperatorBundleError(
                        f"operator bundle supports only files and directories: {member.name}"
                    )
                if member.size < 0:
                    raise OperatorBundleError(
                        f"operator bundle file has a negative size: {member.name}"
                    )
                total_size += member.size
                if total_size > self.max_bytes:
                    raise OperatorBundleTooLarge(
                        f"operator bundle expands beyond {self.max_bytes} bytes"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise OperatorBundleError(f"cannot read operator bundle file: {member.name}")
                written = 0
                with source, target.open("xb") as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        written += len(chunk)
                if written != member.size:
                    raise OperatorBundleError(
                        f"operator bundle file size mismatch: {member.name}"
                    )
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
            if not found_member:
                raise OperatorBundleError("operator bundle archive is empty")


__all__ = [
    "GemsDefinitionSource",
    "DEFAULT_OPERATOR_BUNDLE_MAX_BYTES",
    "DEFAULT_OPERATOR_BUNDLE_MAX_FILES",
    "OPERATOR_BUNDLE_FORMAT",
    "OPERATOR_BUNDLE_MEDIA_TYPE",
    "OperatorBundleError",
    "OperatorBundleInfo",
    "OperatorBundleStore",
    "OperatorBundleTooLarge",
    "normalize_operator_bundle_sha256",
    "pack_operator_bundle",
]
