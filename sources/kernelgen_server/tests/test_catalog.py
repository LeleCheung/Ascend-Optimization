import json
from pathlib import Path

import pytest

from kernelgen_server import Catalog, builtin_catalog_path
from kernelgen_server.evaluation.loader import load_operator_adapter
from kernelgen_server.protocol.version import KERNELGEN_SUPPORTED_API_VERSIONS
from kernelgen_server.schema import Implementation, SourceFile


def test_bundled_simplified_v6_catalog_is_loadable():
    root = builtin_catalog_path()

    assert root == Path(root).resolve()
    assert root.name == "simple-v6-test"
    catalog = Catalog(root)
    assert catalog.operator_names == ("identity",)
    operator = catalog.load("identity")
    assert operator.definition.api_version == "v6.0"
    assert len(operator.correctness_workloads) == 1
    assert len(operator.timing_workloads) == 1


def test_repository_data_separates_unsupported_catalogs():
    data_root = Path(__file__).parents[1] / "data"
    active_manifests = sorted(data_root.glob("*/manifest.json"))
    archived_manifests = sorted((data_root / ".old").glob("*/manifest.json"))

    assert active_manifests
    assert archived_manifests
    for path in active_manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["api_version"] in KERNELGEN_SUPPORTED_API_VERSIONS
    for path in archived_manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["api_version"] not in KERNELGEN_SUPPORTED_API_VERSIONS


@pytest.mark.parametrize("name", ["../simple-v6-test", "nested/catalog", ""])
def test_builtin_catalog_path_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        builtin_catalog_path(name)


def _v62_catalog(
    tmp_path: Path,
    *,
    oracle: str | None = None,
    group: str | None = None,
) -> Path:
    root = tmp_path / "v62"
    operator = root / "ops"
    if group is not None:
        operator /= group
    operator /= "identity"
    operator.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "api_version": "v6.2",
                "evaluator": "native",
                "layout": "per-operator",
            }
        ),
        encoding="utf-8",
    )
    (operator / "definition.json").write_text(
        json.dumps(
            {
                "api_version": "v6.2",
                "name": "identity",
                "parameters": [{"name": "x", "required": True}],
                "outputs": ["out"],
            }
        ),
        encoding="utf-8",
    )
    (operator / "oracle.py").write_text(
        oracle
        or (
            'REFERENCE_DEVICE = "target"\n\n'
            "def run(x): return x\n"
        ),
        encoding="utf-8",
    )
    (operator / "correctness.jsonl").write_text(
        json.dumps(
            {
                "name": "one",
                "inputs": {"x": {"type": "scalar", "value": 1}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_v62_per_operator_catalog_injects_one_oracle_source(tmp_path):
    catalog = Catalog(_v62_catalog(tmp_path))
    operator = catalog.load("identity")

    assert operator.definition.api_version == "v6.2"
    assert operator.definition.reference_device.value == "target"
    assert "def run" in operator.definition.reference
    request = catalog.request(
        "identity",
        Implementation(
            name="candidate",
            definition="identity",
            language="python",
            entrypoint="main.py::run",
            sources=[SourceFile(path="main.py", content="def run(x): return x")],
        ),
    )
    assert request.api_version == "v6.2"


def test_v62_per_operator_catalog_allows_one_group_directory(tmp_path):
    catalog = Catalog(_v62_catalog(tmp_path, group="pointwise"))

    operator = catalog.load("identity")

    assert operator.record_id == "identity"
    assert operator.relative == "pointwise/identity"


def test_v62_catalog_rejects_multiple_reference_device_declarations(tmp_path):
    root = _v62_catalog(
        tmp_path,
        oracle=(
            'REFERENCE_DEVICE = "target"\n'
            'REFERENCE_DEVICE = "cpu"\n'
            "def correctness_run(x): return x\n"
        ),
    )
    with pytest.raises(ValueError, match="exactly once"):
        Catalog(root).load("identity")


def test_v62_catalog_rejects_separate_reference_files(tmp_path):
    root = _v62_catalog(tmp_path)
    (root / "ops" / "identity" / "torch_reference.py").write_text(
        "def run(x): return x\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="oracle.py"):
        Catalog(root).load("identity")


def test_v62_definition_json_must_remain_pure_abi(tmp_path):
    root = _v62_catalog(tmp_path)
    path = root / "ops" / "identity" / "definition.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["reference_device"] = "target"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="public ABI"):
        Catalog(root)


def test_v62_flaggems_framework_binding_requires_repository_branch_and_revision(
    tmp_path,
):
    root = _v62_catalog(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "framework": "flaggems",
            "framework_repository": "https://example.com/FlagGems.git",
            "framework_branch": "feat/kernelgen",
            "framework_revision": "abc",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="full lowercase commit hash"):
        Catalog(root)

    manifest["framework_revision"] = "a" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    catalog = Catalog(root)

    assert catalog.framework == "flaggems"
    assert catalog.framework_repository == "https://example.com/FlagGems.git"
    assert catalog.framework_branch == "feat/kernelgen"
    assert catalog.framework_revision == "a" * 40


def test_v62_operator_private_assets_are_loaded_with_oracle(tmp_path):
    root = _v62_catalog(
        tmp_path,
        oracle=(
            'REFERENCE_DEVICE = "target"\n'
            "from pathlib import Path\n"
            "def run(x):\n"
            "    value = Path(__file__).parent / 'assets' / 'offset.txt'\n"
            "    return x + int(value.read_text())\n"
        ),
    )
    assets = root / "ops" / "identity" / "assets"
    assets.mkdir()
    (assets / "offset.txt").write_text("2\n", encoding="utf-8")

    operator = Catalog(root).load("identity")
    adapter = load_operator_adapter(
        operator.definition,
        oracle_path=operator.oracle_path,
    )

    assert adapter.reference(3) == 5
    assert operator.assets_digest.startswith("sha256:")


def test_v62_operator_asset_namespace_does_not_leak_between_catalogs(tmp_path):
    oracle = (
        'REFERENCE_DEVICE = "target"\n'
        "from assets.helper import OFFSET\n"
        "def run(x): return x + OFFSET\n"
    )
    roots = [
        _v62_catalog(tmp_path / "first", oracle=oracle),
        _v62_catalog(tmp_path / "second", oracle=oracle),
    ]
    for root, offset in zip(roots, (2, 5), strict=True):
        assets = root / "ops" / "identity" / "assets"
        assets.mkdir()
        (assets / "helper.py").write_text(
            f"OFFSET = {offset}\n", encoding="utf-8"
        )

    values = []
    for root in roots:
        operator = Catalog(root).load("identity")
        adapter = load_operator_adapter(
            operator.definition,
            oracle_path=operator.oracle_path,
        )
        values.append(adapter.reference(1))

    assert values == [3, 6]


def test_v62_operator_private_assets_change_benchmark_fingerprint(tmp_path):
    pytest.importorskip("torch")
    from kernelgen_server.evaluation.adapters.native import NativeEvaluationAdapter

    root = _v62_catalog(tmp_path)
    assets = root / "ops" / "identity" / "assets"
    assets.mkdir()
    helper = assets / "table.bin"
    helper.write_bytes(b"first")

    first_catalog = Catalog(root)
    first = NativeEvaluationAdapter(
        catalog=first_catalog,
        operator=first_catalog.load("identity"),
    ).inspect().benchmark_fingerprint
    helper.write_bytes(b"second")
    second_catalog = Catalog(root)
    second = NativeEvaluationAdapter(
        catalog=second_catalog,
        operator=second_catalog.load("identity"),
    ).inspect().benchmark_fingerprint

    assert first != second


def test_v62_catalog_rejects_extra_files_outside_assets(tmp_path):
    root = _v62_catalog(tmp_path)
    (root / "ops" / "identity" / "helper.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="under assets"):
        Catalog(root).load("identity")


def test_v62_catalog_allows_inactive_full_workload_archives(tmp_path):
    root = _v62_catalog(tmp_path)
    operator = root / "ops" / "identity"
    (operator / "timing.jsonl").write_text(
        json.dumps(
            {
                "name": "two",
                "inputs": {"x": {"type": "scalar", "value": 2}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for phase in ("correctness", "timing"):
        (operator / f"{phase}_full.jsonl").write_text(
            (operator / f"{phase}.jsonl").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    loaded = Catalog(root).load("identity")

    assert len(loaded.correctness_workloads) == 1
    assert len(loaded.timing_workloads) == 1


def test_v62_catalog_rejects_symlinked_assets(tmp_path):
    root = _v62_catalog(tmp_path)
    assets = root / "ops" / "identity" / "assets"
    assets.mkdir()
    target = tmp_path / "external.bin"
    target.write_bytes(b"external")
    (assets / "external.bin").symlink_to(target)

    with pytest.raises(ValueError, match="symlinks"):
        Catalog(root).load("identity")
