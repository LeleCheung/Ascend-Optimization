"""Generic Agent-backed extraction for one FlagGems Native v6.2 operator."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
from typing import Any, Callable

from pydantic import BaseModel, Field, model_validator

from kernelgen.agents.extractor.flaggems.v62_agent import (
    FlagGemsV62ExtractorAgent,
    FlagGemsV62ExtractorInput,
    FlagGemsV62ExtractorOutput,
    build_flaggems_v62_accuracy_coverage,
    persist_flaggems_v62_extraction,
)
from kernelgen.agents.extractor.flaggems.source_inventory import collect_source_inventory
from kernelgen.agents.extractor.flaggems.source_profile import PROFILES
from kernelgen.framework.workflow import Workflow
from kernelgen.data._atomic import atomic_write_json
from kernelgen.framework.local_state import file_lock
from kernelgen.framework.run_control import RunCancelled, RunState, WorkspaceRunControl
from kernelgen.agents.extractor.flaggems.case_collection import prepare_case_list
from kernelgen.workflows.catalog_extract_review import CatalogExtractionBlocked, CatalogReviewRequired, review_attempt, publish_review_link


class CatalogExtractInput(BaseModel):
    operator: str = Field(min_length=1)
    flaggems_repo: str | None = None
    pr_url: str | None = None
    case_list_path: str | None = None
    max_review_rounds: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def one_source(self):
        if bool(self.flaggems_repo) == bool(self.pr_url):
            raise ValueError("provide exactly one of flaggems_repo or pr_url")
        if self.pr_url:
            from kernelgen.agents.extractor.flaggems.pr_source import pull_request_number
            pull_request_number(self.pr_url)
        return self


class CatalogExtractOutput(BaseModel):
    operator: str
    extraction: FlagGemsV62ExtractorOutput
    accuracy_coverage: dict[str, Any]
    case_list_path: str = ""
    catalog_path: Path
    operator_dir: Path
    source_evidence: list[Path]
    review_path: Path | None = None


class CatalogExtractWorkflow(Workflow):
    """Run the generic V6.2 extractor without mutating a shared Catalog."""

    name = "flaggems_v62_agent_extract"
    InputModel = CatalogExtractInput
    OutputModel = CatalogExtractOutput

    def __init__(
        self,
        *,
        cwd: str = ".",
        runtime_factory: Callable[[str], Any] | None = None,
    ):
        self._cwd = Path(cwd).resolve()
        self._runtime_factory = runtime_factory

    @classmethod
    def bind(
        cls,
        path: str,
        runtime_factory: Callable[[str], Any],
    ) -> "CatalogExtractWorkflow":
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp: CatalogExtractInput) -> dict[str, Any]:
        if self._runtime_factory is None:
            raise RuntimeError("flaggems_v62_agent_extract requires a runtime_factory")
        self._cwd.mkdir(parents=True, exist_ok=True)
        control = WorkspaceRunControl(self._cwd, source=self.name)
        control.checkpoint("BEFORE_CATALOG_EXTRACT")
        with file_lock(self._cwd / ".catalog-extract.lock"):
            control.update_progress(state=RunState.RUNNING, stage='EXTRACTING', progress_kind='basic')
            try:
                result = self._extract(inp)
                control.update_progress(state=RunState.SUCCEEDED,stage='COMPLETED',message='Catalog review accepted')
                return result
            except RunCancelled:
                control.acknowledge_cancellation(stage='CANCELLED')
                raise
            except CatalogReviewRequired as exc:
                control.update_progress(state=RunState.PENDING,stage='WAITING_REVIEW',message=str(exc))
                raise
            except CatalogExtractionBlocked as exc:
                try:
                    control.checkpoint('AFTER_BLOCKER_REPORT')
                except RunCancelled:
                    control.acknowledge_cancellation(stage='CANCELLED')
                    raise
                atomic_write_json(self._cwd / 'review-loop.json', dict(
                    state='BLOCKED', report=str(exc.report_path)))
                control.update_progress(state=RunState.PENDING, stage='BLOCKED', message=str(exc))
                raise
            except Exception as exc:
                control.update_progress(state=RunState.FAILED,stage='FAILED',message=str(exc))
                raise

    def _extract(self, inp):
        if (self._cwd / "catalog").exists() or (self._cwd / "extraction.json").exists():
            raise ValueError("extraction artifacts already exist; preserve them and use a new workspace")
        if inp.pr_url:
            from kernelgen.agents.extractor.flaggems.pr_source import prepare_pull_request, verify_checkout
            from kernelgen.agents.extractor.flaggems.pr_agent import FlagGemsPRExtractorAgent, FlagGemsPRExtractorInput, extraction_input

            source_root = self._cwd.resolve() / "source"
            source = prepare_pull_request(inp.pr_url, source_root)
            repo = source_root / "head"
        else:
            repo = Path(inp.flaggems_repo).expanduser().resolve()
        cases = (Path(inp.case_list_path).expanduser().resolve() if inp.case_list_path else
                 prepare_case_list(repo, inp.operator, self._cwd / "case_collection", self._runtime_factory))
        digest = hashlib.sha256(cases.read_bytes()).hexdigest()
        plan = {"operator": inp.operator, "source_root": str(repo), "pr_url": inp.pr_url,
                "case_list_sha256": digest, "max_review_rounds": inp.max_review_rounds}
        plan_path = self._cwd / "catalog-extract-input.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("extraction input changed; use a new workspace")
        atomic_write_json(plan_path, plan)
        if inp.pr_url:
            selected = FlagGemsPRExtractorInput(operator=inp.operator, source_workspace=source_root,
                                                case_list_path=cases)
            agent_input = extraction_input(selected)
            agent = FlagGemsPRExtractorAgent()
        else:
            agent_input = FlagGemsV62ExtractorInput(
                operator=inp.operator, flaggems_repo=str(repo), case_list_path=str(cases))
            selected = agent_input
            agent = FlagGemsV62ExtractorAgent()
        inventory = collect_source_inventory(str(repo), inp.operator)
        evidence = [inventory.implementation_file, *inventory.test_files, *inventory.benchmark_files, cases]
        # Review needs the actual helper/defaults, not guesses from calls such as to_reference().
        evidence.extend(path for relative in (
            'tests/accuracy_utils.py', 'tests/conftest.py', 'benchmark/base.py',
        ) if (path := repo / relative).is_file())
        evidence.extend(path for profile in PROFILES
                        if (path := profile.package_root(repo) / 'testing/__init__.py').is_file())
        # The source's vendor policy governs conditional dtype lists and oracle
        # precision, not the machine used to collect the timing list.
        for profile in PROFILES:
            backend = profile.package_root(repo) / 'runtime/backend'
            evidence.extend(path for relative in ('backend_utils.py', 'device_finder.py')
                            if (path := backend / relative).is_file())
            evidence.extend(sorted(backend.glob('_*/__init__.py')))
        if inp.pr_url:
            evidence.append(source_root / "source.json")
        hashes = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in evidence}
        control = WorkspaceRunControl(self._cwd, source=self.name)
        runtime = self._runtime_factory(str(self._cwd / 'extractor'))
        feedback = ''
        for iteration in range(1, inp.max_review_rounds + 1):
            control.checkpoint('BEFORE_EXTRACTION_REVISION')
            attempt = self._cwd / 'attempts' / f'{iteration:02d}'
            attempt.mkdir(parents=True, exist_ok=False)
            control.update_progress(stage='EXTRACTING' if iteration == 1 else 'REVISING',
                                    message=f'Extraction/review round {iteration}/{inp.max_review_rounds}')
            if iteration == 1:
                extraction = agent.run(selected.model_dump(mode='python', exclude_defaults=True), runtime)
            else:
                extraction = agent.continue_session(feedback, runtime)
            control.checkpoint('AFTER_EXTRACTION_REVISION')
            if hashes != {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in evidence}:
                raise ValueError('source evidence or case list changed during extraction')
            if inp.pr_url:
                verify_checkout(source_root / 'head', source.head_sha, source.remote)
                verify_checkout(source_root / 'base', source.base_sha, source.remote)
            if extraction.blocker is not None:
                extraction.blocker.validate_evidence(evidence)
                report_path = attempt / 'extraction.json'
                atomic_write_json(report_path, extraction.model_dump(mode='json'))
                raise CatalogExtractionBlocked(report_path)
            coverage = build_flaggems_v62_accuracy_coverage(agent_input, extraction)
            catalog = attempt / 'catalog'
            persisted = persist_flaggems_v62_extraction(agent_input, extraction, catalog)
            if inp.pr_url:
                from kernelgen.agents.extractor.flaggems.source_profile import repository_profile
                manifest = catalog / 'manifest.json'
                binding = json.loads(manifest.read_text())
                binding.update(framework=repository_profile(source.repository).framework,
                    framework_repository=source.remote,framework_branch=f'refs/pull/{source.number}/head',
                    framework_revision=source.head_sha)
                atomic_write_json(manifest,binding)
            atomic_write_json(attempt/'accuracy_coverage.json',coverage)
            atomic_write_json(attempt/'extraction.json',extraction.model_dump(mode='json'))
            control.update_progress(stage='REVIEWING', message=f'Reviewing extraction round {iteration}')
            accepted, review_path, feedback = review_attempt(inp.operator,catalog,
                [*evidence,attempt/'accuracy_coverage.json'],attempt/'review',self._runtime_factory)
            control.checkpoint('AFTER_EXTRACTION_REVIEW')
            atomic_write_json(self._cwd/'review-loop.json',dict(
                state='SUCCEEDED' if accepted else 'NEEDS_FIX',round=iteration,max_rounds=inp.max_review_rounds,
                review=str(review_path)))
            if accepted:
                break
        else:
            raise CatalogReviewRequired(f'Catalog review still requires changes after {inp.max_review_rounds} rounds: {review_path}')
        destination = self._cwd/'catalog'
        relative = persisted.operator_root.relative_to(catalog)
        shutil.copytree(catalog,destination)
        catalog = destination
        publish_review_link(catalog, inp.operator, review_path)
        atomic_write_json(self._cwd / "accuracy_coverage.json", coverage)
        atomic_write_json(self._cwd / "extraction.json", extraction.model_dump(mode="json"))
        return {
            "operator": inp.operator,
            "extraction": extraction.model_dump(mode="python"),
            "accuracy_coverage": coverage,
            "case_list_path": str(cases),
            "catalog_path": catalog,
            "operator_dir": catalog / relative,
            "source_evidence": list(dict.fromkeys(Path(p).resolve() for p in evidence)),
            "review_path": review_path,
        }


__all__ = [
    "CatalogExtractInput",
    "CatalogExtractOutput",
    "CatalogExtractWorkflow",
]
