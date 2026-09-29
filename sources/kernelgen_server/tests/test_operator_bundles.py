from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from kernelgen_server import Catalog
from kernelgen_server.api.operator_bundles import create_operator_bundle_router
from kernelgen_server.operator_bundles import (
    OPERATOR_BUNDLE_MEDIA_TYPE,
    OperatorBundleError,
    OperatorBundleStore,
    OperatorBundleTooLarge,
    pack_operator_bundle,
)
from kernelgen_server.protocol import client as protocol_client


def _operator_dir(root: Path, *, api_version: str = "v6.2") -> Path:
    root.mkdir()
    (root / "definition.json").write_text(
        json.dumps(
            {
                "api_version": api_version,
                "name": "user_square",
                "parameters": [{"name": "x", "required": True}],
                "outputs": ["out"],
            }
        ),
        encoding="utf-8",
    )
    (root / "oracle.py").write_text(
        'REFERENCE_DEVICE = "target"\n\ndef run(x):\n    return x * x\n',
        encoding="utf-8",
    )
    (root / "correctness.jsonl").write_text(
        json.dumps(
            {
                "name": "user-square-correctness",
                "inputs": {"x": {"type": "random", "shape": [8], "dtype": "float32"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "timing.jsonl").write_text(
        json.dumps(
            {
                "name": "user-square-timing",
                "inputs": {"x": {"type": "random", "shape": [1024], "dtype": "float32"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assets = root / "assets"
    assets.mkdir()
    (assets / "metadata.json").write_text('{"source":"user"}\n', encoding="utf-8")
    pycache = root / "__pycache__"
    pycache.mkdir()
    (pycache / "oracle.cpython-312.pyc").write_bytes(b"generated")
    return root


def _packed(operator_dir: Path, destination: Path) -> tuple[str, int]:
    with destination.open("w+b") as output:
        return pack_operator_bundle(operator_dir, output)


def test_pack_is_deterministic_and_store_installs_a_loadable_catalog(tmp_path: Path):
    operator_dir = _operator_dir(tmp_path / "operator")
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    first_digest, first_size = _packed(operator_dir, first)
    second_digest, second_size = _packed(operator_dir, second)

    assert first_digest == second_digest
    assert first_size == second_size
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:") as archive:
        assert not any("__pycache__" in member.name for member in archive)

    store = OperatorBundleStore(tmp_path / "bundles")
    info, created = store.install(first, first_digest)
    repeated, repeated_created = store.install(second, second_digest)

    assert created is True
    assert repeated_created is False
    assert repeated == info
    assert info.bundle_id == f"sha256:{first_digest}"
    assert info.definition == "user_square"
    assert store.describe() == {
        "enabled": True,
        "archive_format": "kernelgen.operator-bundle/v1",
        "media_type": OPERATOR_BUNDLE_MEDIA_TYPE,
        "max_bytes": 256 * 1024 * 1024,
        "max_files": 10_000,
        "evaluation_binding": True,
        "evaluators": ["native", "flaggems"],
    }
    catalog = Catalog(store.catalog_path(info.bundle_id))
    operator = catalog.load("user_square")
    assert operator.definition.api_version == "v6.2"
    assert [item.name for item in operator.correctness_workloads] == [
        "user-square-correctness"
    ]
    assert [item.name for item in operator.timing_workloads] == ["user-square-timing"]
    assert (catalog.root / "ops" / "user_square" / "assets" / "metadata.json").is_file()


def test_store_rejects_digest_mismatch_and_non_v62_definition(tmp_path: Path):
    operator_dir = _operator_dir(tmp_path / "operator", api_version="v6.0")
    archive = tmp_path / "operator.tar"
    digest, _ = _packed(operator_dir, archive)
    store = OperatorBundleStore(tmp_path / "bundles")

    with pytest.raises(OperatorBundleError, match="does not match"):
        store.install(archive, "0" * 64)
    with pytest.raises(OperatorBundleError, match="api_version=v6.2"):
        store.install(archive, digest)

    assert store.get(digest) is None


def test_store_rejects_parent_traversal(tmp_path: Path):
    archive_path = tmp_path / "traversal.tar"
    with tarfile.open(archive_path, "w") as archive:
        traversal = tarfile.TarInfo("../escaped")
        traversal.size = 1
        archive.addfile(traversal, io.BytesIO(b"x"))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    store = OperatorBundleStore(tmp_path / "bundles")

    with pytest.raises(OperatorBundleError, match="unsafe operator bundle path"):
        store.install(archive_path, digest)

    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_store_rejects_links(tmp_path: Path, member_type: bytes):
    archive_path = tmp_path / f"link-{member_type.decode()}.tar"
    with tarfile.open(archive_path, "w") as archive:
        link = tarfile.TarInfo("assets/link")
        link.type = member_type
        link.linkname = "/etc/passwd"
        archive.addfile(link)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    store = OperatorBundleStore(tmp_path / "bundles")

    with pytest.raises(OperatorBundleError, match="only files and directories"):
        store.install(archive_path, digest)


def test_store_enforces_archive_size_before_extraction(tmp_path: Path):
    operator_dir = _operator_dir(tmp_path / "operator")
    archive = tmp_path / "operator.tar"
    digest, size = _packed(operator_dir, archive)
    store = OperatorBundleStore(tmp_path / "bundles", max_bytes=size - 1)

    with pytest.raises(OperatorBundleTooLarge, match="exceeds"):
        store.install(archive, digest)


def test_http_upload_is_queryable_and_idempotent(tmp_path: Path):
    pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    operator_dir = _operator_dir(tmp_path / "operator")
    archive = tmp_path / "operator.tar"
    digest, _ = _packed(operator_dir, archive)
    store = OperatorBundleStore(tmp_path / "bundles")
    app = FastAPI()
    app.include_router(create_operator_bundle_router(store))

    with TestClient(app) as client:
        assert client.head(f"/operator-bundles/{digest}").status_code == 404
        created = client.put(
            f"/operator-bundles/{digest}",
            content=archive.read_bytes(),
            headers={"Content-Type": OPERATOR_BUNDLE_MEDIA_TYPE},
        )
        head = client.head(f"/operator-bundles/{digest}")
        fetched = client.get(f"/operator-bundles/{digest}")
        repeated = client.put(
            f"/operator-bundles/{digest}",
            content=archive.read_bytes(),
            headers={"Content-Type": OPERATOR_BUNDLE_MEDIA_TYPE},
        )

    assert created.status_code == 201
    assert created.json()["definition"] == "user_square"
    assert head.status_code == 200
    assert head.headers["X-KernelGen-Bundle-Id"] == f"sha256:{digest}"
    assert fetched.status_code == 200
    assert fetched.json() == created.json()
    assert repeated.status_code == 200
    assert repeated.json() == created.json()


def test_http_upload_rejects_wrong_digest_and_media_type(tmp_path: Path):
    pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    operator_dir = _operator_dir(tmp_path / "operator")
    archive = tmp_path / "operator.tar"
    digest, _ = _packed(operator_dir, archive)
    app = FastAPI()
    app.include_router(create_operator_bundle_router(OperatorBundleStore(tmp_path / "bundles")))

    with TestClient(app) as client:
        wrong_digest = client.put(
            f"/operator-bundles/{'0' * 64}",
            content=archive.read_bytes(),
            headers={"Content-Type": OPERATOR_BUNDLE_MEDIA_TYPE},
        )
        wrong_type = client.put(
            f"/operator-bundles/{digest}",
            content=archive.read_bytes(),
            headers={"Content-Type": "application/json"},
        )

    assert wrong_digest.status_code == 422
    assert wrong_type.status_code == 415


def test_protocol_client_skips_upload_after_digest_is_present(tmp_path: Path, monkeypatch):
    operator_dir = _operator_dir(tmp_path / "operator")
    uploaded: dict[str, object] = {}
    methods: list[str] = []

    class Response:
        def __init__(self, status_code: int, payload: dict | None = None):
            self.status_code = status_code
            self.payload = payload
            self.text = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def json(self):
            return self.payload

    class Session:
        @staticmethod
        def request(method, url, **kwargs):
            methods.append(method)
            digest = url.rsplit("/", 1)[-1]
            if method == "GET":
                return Response(200, uploaded) if uploaded else Response(404)
            assert method == "PUT"
            content = kwargs["data"].read()
            assert hashlib.sha256(content).hexdigest() == digest
            assert kwargs["headers"] == {
                "Content-Type": OPERATOR_BUNDLE_MEDIA_TYPE,
                "Content-Length": str(len(content)),
            }
            uploaded.update(
                {
                    "bundle_id": f"sha256:{digest}",
                    "sha256": digest,
                    "definition": "user_square",
                    "archive_format": "kernelgen.operator-bundle/v1",
                    "size_bytes": len(content),
                    "created_at": "2026-09-05T00:00:00+00:00",
                }
            )
            return Response(201, uploaded)

    monkeypatch.setattr(protocol_client, "_SESSION", Session())

    first = protocol_client.upload_operator_bundle(operator_dir, "http://server")
    second = protocol_client.upload_operator_bundle(operator_dir, "http://server")

    assert first == second
    assert methods == ["GET", "PUT", "GET"]
