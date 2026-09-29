"""Freeze one selected operator without modifying the caller's Catalog."""

import hashlib
import shutil
import tempfile
from pathlib import Path
from contextlib import contextmanager

from kernelgen_client import Catalog
from kernelgen_client.operator_bundles import pack_operator_bundle
from kernelgen.framework.workflow import WorkflowResult
from kernelgen.data._atomic import atomic_write_json


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workflow_result(data, files=(), *, state="SUCCEEDED", message="Workflow call completed"):
    return WorkflowResult(state=state, message=message,
                          output={**data, "artifacts": {str(Path(p).resolve()): file_digest(Path(p)) for p in files}})


def selected_operator(catalog_path, operator):
    catalog = Catalog(catalog_path)
    if catalog.api_version != "v6.2" or catalog.evaluator != "native":
        raise ValueError("Catalog optimization requires a v6.2 native Catalog")
    selected = catalog.load(operator)
    if not selected.correctness_workloads or not selected.timing_workloads:
        raise ValueError("optimization requires both correctness and timing workloads")
    root = catalog.root / "ops" / selected.relative
    if root.is_symlink() or not root.resolve().is_relative_to(catalog.root):
        raise ValueError("operator directory must be inside the Catalog")
    return root


@contextmanager
def operator_bundle_source(catalog_path, operator):
    catalog = Catalog(catalog_path)
    if catalog.evaluator != "flaggems":
        yield selected_operator(catalog_path, operator)
        return
    from kernelgen_client.operator_bundles import GemsDefinitionSource
    source = GemsDefinitionSource.model_validate(catalog.manifest.get("definition_source"))
    selected = catalog.load(operator)
    if selected.definition.api_version != "v6.0":
        raise ValueError("Gems Definition input requires api_version=v6.0")
    definition = catalog.root / "definitions" / f"{operator}.json"
    if definition.is_symlink() or not definition.resolve().is_relative_to(catalog.root):
        raise ValueError("Definition must be inside the Catalog")
    with tempfile.TemporaryDirectory(prefix="kg-gems-definition-") as directory:
        root = Path(directory)
        shutil.copyfile(definition, root / "definition.json")
        atomic_write_json(root / "adapter.json", source.model_dump(mode="json"))
        yield root


def catalog_identity(catalog_path, operator):
    with operator_bundle_source(catalog_path, operator) as root, tempfile.TemporaryFile() as output:
        digest, _ = pack_operator_bundle(root, output)
    return hashlib.sha256((Path(catalog_path) / "manifest.json").read_bytes() + digest.encode()).hexdigest()


def catalog_result(catalog_path, operator_root, workspace, source_evidence=()):
    archive = workspace / "operator.tar"
    with archive.open("w+b") as output:
        digest, _ = pack_operator_bundle(operator_root, output)
    files = [p for p in catalog_path.rglob("*") if p.is_file()]
    return workflow_result({"catalog": str(catalog_path), "operator_dir": str(operator_root),
                         "bundle_id": "sha256:" + digest,
                         "source_evidence": [str(p) for p in source_evidence]}, [*files, archive])


def snapshot_catalog(inp, workspace):
    from kernelgen.workflows.catalog_extract_review import review_link_path
    if Catalog(inp.catalog_path).evaluator == "flaggems":
        catalog = workspace / "catalog"
        definition = catalog / "definitions" / f"{inp.operator}.json"
        definition.parent.mkdir(parents=True)
        shutil.copy2(inp.catalog_path / "manifest.json", catalog / "manifest.json")
        with operator_bundle_source(inp.catalog_path, inp.operator) as source:
            destination = workspace / "definition-bundle"
            shutil.copytree(source, destination)
            shutil.copyfile(source / "definition.json", definition)
        if catalog_identity(catalog, inp.operator) != inp.catalog_sha256:
            raise ValueError("catalog changed while preparing the snapshot")
        return catalog_result(catalog, destination, workspace)
    source = selected_operator(inp.catalog_path, inp.operator)
    catalog = workspace / "catalog"
    destination = catalog / source.relative_to(inp.catalog_path)
    shutil.copytree(source, destination, symlinks=True)
    shutil.copy2(inp.catalog_path / "manifest.json", catalog / "manifest.json")
    if catalog_identity(catalog, inp.operator) != inp.catalog_sha256:
        raise ValueError("catalog changed while preparing the snapshot")
    result = catalog_result(catalog, destination, workspace)
    link = review_link_path(inp.catalog_path)
    if link.is_file():
        frozen_link = review_link_path(catalog)
        shutil.copy2(link, frozen_link)
        result.output['artifacts'][str(frozen_link.resolve())] = file_digest(frozen_link)
    return result
