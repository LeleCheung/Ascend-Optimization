"""Adapter wrapper around the simplified native Workload engine."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from kernelgen_server.evaluation.engine import EvaluationEngine
from kernelgen_server.evaluation.loader import materialize_implementation
from kernelgen_server.profiling.models import (
    ProfileCommand,
    ProfileOptions,
    ProfileRequest,
    ProfileTarget,
)
from kernelgen_server.protocol.schema import (
    AdapterCapabilities,
    AdapterCase,
    AdapterManifest,
    BoundEvaluateRequest,
    CandidateContract,
    CaseList,
    PreflightResult,
)
from kernelgen_server.protocol.workload_call import definition_signature
from kernelgen_server.runtime.backend import runtime_device_type
from kernelgen_server.runtime.environment import bind_backend_device

from .base import EvaluatorAdapter


def _format_preflight_errors(per_workload: dict[str, dict[str, object]]) -> str:
    sections = []
    for case_id, value in per_workload.items():
        error = value.get("error", "")
        if error:
            sections.append(f"[{case_id}]\n{error}")
        elif value.get("status") == "SKIP":
            sections.append(f"[{case_id}] SKIP: {value.get('reason', '')}")
    return "\n".join(sections)


class NativeEvaluationAdapter(EvaluatorAdapter):
    kind = "native"

    def _fingerprint(self) -> str:
        payload = {
            "reference_policy": (
                "v6.2-whole-request-oracle-selection-timing-perf-only-"
                "custom-valid-file-backed/v4"
                if self.operator.definition.api_version == "v6.2"
                else "v6.0-single-reference/v1"
            ),
            "framework": self.catalog.framework,
            "framework_repository": self.catalog.framework_repository,
            "framework_branch": self.catalog.framework_branch,
            "framework_revision": self.catalog.framework_revision,
            "oracle_assets": self.operator.assets_digest,
            "definition": self.operator.definition.model_dump(mode="json"),
            "correctness": [
                item.model_dump(mode="json")
                for item in self.operator.correctness_workloads
            ],
            "timing": [
                item.model_dump(mode="json") for item in self.operator.timing_workloads
            ],
        }
        if self.operator.definition.source_policy_id:
            from kernelgen_server.runtime.source_policy import policy_snapshot
            payload["source_policy"] = policy_snapshot(self.backend)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def inspect(self) -> AdapterManifest:
        fingerprint = self._fingerprint()
        cases = [
            AdapterCase(
                case_id=workload.name,
                ordinal=index,
                dtype=(
                    str(workload.inputs.get("dtype"))
                    if workload.inputs.get("dtype") is not None
                    else None
                ),
                shape=workload.inputs.get("shape"),
                params={
                    key: value
                    for key, value in workload.inputs.items()
                    if key not in {"dtype", "shape"}
                },
            )
            for index, workload in enumerate(self.operator.timing_workloads)
        ]
        case_list = CaseList(
            adapter_kind="native",
            operator=self.operator.definition.name,
            benchmark_fingerprint=fingerprint,
            cases=cases,
        )
        return AdapterManifest(
            kind="native",
            benchmark_fingerprint=fingerprint,
            candidate_contract=CandidateContract(
                signature=str(definition_signature(self.operator.definition))
            ),
            capabilities=AdapterCapabilities(preflight=True, profile=True),
            case_list=case_list,
        )

    def _native_request(self, request: BoundEvaluateRequest):
        return self.catalog.request(
            self.operator.record_id,
            request.implementation,
            request.settings,
        )

    def preflight(self, request: BoundEvaluateRequest) -> PreflightResult:
        if self.device is None:
            raise RuntimeError("native preflight requires a device")
        raw = EvaluationEngine(
            self.device,
            self.device_string,
            self.backend,
            oracle_path=self.operator.oracle_path,
        ).preflight(self._native_request(request))
        per_workload = raw.get("per_workload", {})
        return PreflightResult(
            status=raw["status"] if raw["status"] in {"PASSED", "FAILED"} else "RUNTIME_ERROR",
            stage=raw.get("stage", "complete" if raw["status"] == "PASSED" else "candidate_smoke"),
            log="\n".join(filter(None, (raw.get("log", ""), _format_preflight_errors(per_workload)))),
            is_hack=raw.get("is_hack", False),
            hack_reason=raw.get("hack_reason", ""),
            benchmark_fingerprint=self._fingerprint(),
            num_cases=len(per_workload),
        )

    def evaluate(self, request: BoundEvaluateRequest):
        if self.device is None:
            raise RuntimeError("native evaluation requires a device")
        return EvaluationEngine(
            self.device,
            self.device_string,
            self.backend,
            oracle_path=self.operator.oracle_path,
        ).evaluate(self._native_request(request))

    def build_profile_command(
        self,
        request: ProfileRequest,
        options: ProfileOptions,
        artifact_dir: Path,
        device: str,
    ) -> ProfileCommand:
        manifest = self.inspect()
        if request.benchmark_fingerprint != manifest.benchmark_fingerprint:
            raise ValueError("benchmark fingerprint changed after inspect")
        workloads = {
            workload.name: workload for workload in self.operator.timing_workloads
        }
        try:
            workload = workloads[request.case_id]
        except KeyError as exc:
            raise ValueError(f"unknown native timing case_id: {request.case_id}") from exc
        target = ProfileTarget(
            evaluation_id=request.evaluation_id,
            implementation=request.implementation,
            definition=self.operator.definition,
            workload=workload,
            expected_backend=request.expected_backend,
            oracle_path=(
                str(self.operator.oracle_path)
                if self.operator.oracle_path is not None
                else None
            ),
        )
        data_dir = artifact_dir / "target"
        data_dir.mkdir(parents=True, exist_ok=False)
        candidate_root = materialize_implementation(
            request.implementation,
            data_dir / "candidate",
        )
        completion_marker = data_dir / "runner-completed"
        (data_dir / "target.json").write_text(
            target.model_dump_json(exclude_unset=True, by_alias=True),
            encoding="utf-8",
        )
        env: dict[str, str] = {}
        inherited_env = dict(os.environ)
        env.update(bind_backend_device(inherited_env, self.backend, device))
        if self.catalog.framework_root is not None:
            python_paths = [str(self.catalog.framework_root / "src")]
            if os.environ.get("PYTHONPATH"):
                python_paths.append(os.environ["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(python_paths)
        return ProfileCommand(
            argv=(
                sys.executable,
                "-m",
                "kernelgen_server.profiling.runner",
                "--data-dir",
                str(data_dir),
                "--device",
                f"{runtime_device_type(self.backend)}:0",
                "--warmup",
                str(options.warmup),
                "--iterations",
                str(options.iterations),
                "--implementation-source-root",
                str(candidate_root),
            ),
            cwd=str(self.catalog.root),
            env=env,
            source_roots=(str(candidate_root),),
            completion_marker_path=str(completion_marker),
        )


__all__ = ["NativeEvaluationAdapter"]
