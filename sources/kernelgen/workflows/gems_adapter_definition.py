"""Build a pure Gems adapter Catalog from a pinned checkout's original pytest.

This workflow reads Python source, never imports Gems or executes its tests on
the Agent host. KGS collects and runs the original suites on the target device.
"""

from pathlib import Path
import hashlib
import json
import re
import subprocess
import tempfile

from pydantic import BaseModel, ConfigDict, Field

from kernelgen.agents.extractor.flaggems import extract_flaggems_definition
from kernelgen.agents.extractor.flaggems.source_inventory import collect_source_inventory, clear_source_inventory_cache
from kernelgen.data._atomic import atomic_write_json
from kernelgen.framework.workflow import Workflow
from kernelgen_client.flaggems_discovery import _discover_suite, _suite_marker_index
from kernelgen_client.operator_bundles import GemsDefinitionSource


class GemsAdapterDefinitionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flaggems_repo: Path
    pytest_path: Path
    operator: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


class GemsAdapterDefinitionOutput(BaseModel):
    operator: str
    catalog_path: Path
    definition_path: Path
    source_revision: str
    source_files: dict[str, str]
    definition_sha256: str


def _clean_revision(repo: Path) -> str:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("flaggems_repo must be the checkout root")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Definition extraction requires a clean, committed Gems checkout")
    return git("rev-parse", "HEAD")


class GemsAdapterDefinitionWorkflow(Workflow):
    """Export ABI plus provenance; no Native reference/workload translation."""

    name = "gems_adapter_definition"
    InputModel = GemsAdapterDefinitionInput
    OutputModel = GemsAdapterDefinitionOutput

    def __init__(self, *, cwd="."):
        self.root = Path(cwd).expanduser().resolve()

    def _execute(self, inp):
        repo = inp.flaggems_repo.expanduser().resolve()
        if self.root.is_relative_to(repo):
            raise ValueError("Definition workspace must be outside the Gems checkout")
        revision = _clean_revision(repo)
        clear_source_inventory_cache()
        _suite_marker_index.cache_clear()
        supplied = inp.pytest_path.expanduser()
        pytest_path = (supplied if supplied.is_absolute() else repo / supplied).resolve()
        if not pytest_path.is_relative_to(repo / "tests") or not pytest_path.is_file():
            raise ValueError("pytest_path must be an existing correctness test inside Gems tests/")
        operator = inp.operator or pytest_path.stem.removeprefix("test_")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", operator):
            raise ValueError("pytest does not identify a safe operator name; supply operator explicitly")
        # Reuse KGS discovery so supplementary suites cannot silently disappear.
        correctness, _ = _discover_suite(repo, "tests", operator)
        performance, _ = _discover_suite(repo, "benchmark", operator)
        if pytest_path not in correctness or not performance:
            raise ValueError("pytest does not identify one testable operator; supply operator with matching correctness and benchmark suites")
        inventory = collect_source_inventory(str(repo), operator)
        definition = extract_flaggems_definition(repo, operator).model_dump(mode="json", exclude_unset=True)
        definition["api_version"] = "v6.0"
        files = set((*correctness, *performance))
        if inventory.implementation_file.is_file():
            files.add(inventory.implementation_file)
        digests = {}
        for path in sorted(files):
            if path.is_symlink() or not path.resolve().is_relative_to(repo):
                raise ValueError("Definition sources must be regular files inside the checkout")
            relative = path.relative_to(repo).as_posix()
            subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", relative],
                           check=True, stdout=subprocess.DEVNULL)
            digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if _clean_revision(repo) != revision:
            raise ValueError("Gems checkout changed during Definition extraction")
        source = GemsDefinitionSource(source_revision=revision, source_files=digests)
        manifest = {"api_version": "v6.0", "name": "gems-pytest-definition", "evaluator": "flaggems",
                    "layout": "flat", "benchmark_level": "core", "definition_source": source.model_dump(mode="json")}
        self.root.mkdir(parents=True, exist_ok=True)
        catalog = self.root / "catalog"
        # A repeated identical export is safe; changed source needs a new workspace.
        if catalog.exists():
            if (json.loads((catalog / "manifest.json").read_text()) != manifest or
                    json.loads((catalog / "definitions" / f"{operator}.json").read_text()) != definition):
                raise ValueError("Definition workspace already contains a different input")
        else:
            with tempfile.TemporaryDirectory(prefix=".definition-", dir=self.root) as staging:
                prepared = Path(staging) / "catalog"
                atomic_write_json(prepared / "manifest.json", manifest)
                atomic_write_json(prepared / "definitions" / f"{operator}.json", definition)
                prepared.rename(catalog)
        path = catalog / "definitions" / f"{operator}.json"
        return {"operator": operator, "catalog_path": catalog, "definition_path": path,
                "source_revision": revision, "source_files": digests,
                "definition_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
