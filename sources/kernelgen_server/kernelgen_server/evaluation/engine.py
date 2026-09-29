"""Core evaluator for the V6 protocol."""

from __future__ import annotations

import gc
import inspect
import random
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from ..protocol.schema import (
    CorrectnessResult,
    EvaluateRequest,
    EvaluateResponse,
    ReferenceDevice,
    TimingResult,
    Workload,
    WorkloadStatus,
)
from ..protocol.version import KERNELGEN_API_VERSION
from ..runtime.device import Device
from .call import Call
from .effects import (
    check_aliases,
    compare_outputs_and_mutations,
    named_outputs,
    resolve_effects,
)
from .hack_detection import log_graylist_hits
from .candidate_admission import check_candidate_admission
from .loader import load_implementation, load_operator_adapter
from .workload_runtime import make_call, make_cpu_bases, workload_context
from ..runtime.source_policy import SourcePolicyError, skip_reason, target_source_policy, validate_selected_policy
from .pytree import clone, to_cpu
from .result import aggregate_evaluation_response


def _normalize_verdict(value: Any) -> tuple[bool, str, Dict[str, Any]]:
    if isinstance(value, bool):
        return value, "", {}
    if isinstance(value, dict):
        passed = value.get("passed")
        if not isinstance(passed, bool):
            raise TypeError("validate verdict mapping must contain boolean 'passed'")
        message = value.get("message", "")
        metrics = value.get("metrics", {})
        if not isinstance(message, str) or not isinstance(metrics, dict):
            raise TypeError("validate verdict message/metrics have invalid types")
        return passed, message, metrics
    raise TypeError("validate must return bool or a verdict mapping")


def _seed_globals(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed % (2**32))
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)


def _output_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (tuple, list)) else [value]


@contextmanager
def _deterministic_correctness():
    """Temporarily stabilize cuDNN only around correctness invocations."""

    import torch

    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    cuda = getattr(torch, "cuda", None)
    available = False
    try:
        available = bool(
            cudnn is not None
            and cuda is not None
            and cuda.is_available()
            and cudnn.is_available()
        )
    except Exception:
        available = False
    if not available:
        yield
        return
    deterministic = cudnn.deterministic
    benchmark = cudnn.benchmark
    cudnn.deterministic = True
    cudnn.benchmark = False
    try:
        yield
    finally:
        cudnn.deterministic = deterministic
        cudnn.benchmark = benchmark


def _load_golden_output(request: EvaluateRequest, workload: Workload) -> Any:
    if workload.output_path is None:
        raise ValueError("workload.output_path is required")
    if not request.definition.outputs:
        raise ValueError("output_path cannot represent a void operator")
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise RuntimeError(
            "safetensors is required for file-backed workloads"
        ) from exc

    with safe_open(workload.output_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = set(request.definition.outputs) - keys
        if missing:
            raise ValueError(
                "output safetensors file is missing Definition output keys: "
                f"{sorted(missing)}"
            )
        values = [handle.get_tensor(name) for name in request.definition.outputs]
    return values[0] if len(values) == 1 else tuple(values)


class EvaluationEngine:
    def __init__(
        self,
        device: Device,
        device_string: str,
        backend: str = "unknown",
        oracle_path: str | Path | None = None,
    ) -> None:
        self.device = device
        self.device_string = device_string
        self.backend = backend
        self.oracle_path = oracle_path

    def _release_device_memory(self) -> None:
        """Release references left by one completed workload/trial."""

        gc.collect()
        self.device.empty_cache()

    def _make_calls(
        self,
        operator: Any,
        request: EvaluateRequest,
        workload: Workload,
    ) -> tuple[
        Call,
        inspect.BoundArguments,
        Call,
        inspect.BoundArguments,
        dict[str, Any],
    ]:
        reference_device = (
            "cpu"
            if request.definition.reference_device == ReferenceDevice.CPU
            else self.device_string
        )
        cpu_bases = make_cpu_bases(workload)
        _seed_globals(workload.seed)
        reference_call, reference_bound = make_call(
            request.definition,
            operator,
            workload,
            cpu_bases,
            reference_device,
        )
        _seed_globals(workload.seed)
        candidate_call, candidate_bound = make_call(
            request.definition,
            operator,
            workload,
            cpu_bases,
            self.device_string,
        )
        return (
            reference_call,
            reference_bound,
            candidate_call,
            candidate_bound,
            cpu_bases,
        )

    @staticmethod
    def _reference_function(operator: Any, *, timing: bool) -> Any:
        function = operator.timing_reference if timing else operator.reference
        phase = "timing" if timing else "correctness"
        if function is None:
            raise RuntimeError(f"selected oracle has no {phase} entrypoint")
        return function

    def _smoke_reference(
        self,
        operator: Any,
        request: EvaluateRequest,
    ) -> None:
        """Run every requested oracle specialization once without candidate/timing."""

        reference_device = (
            "cpu"
            if request.definition.reference_device == ReferenceDevice.CPU
            else self.device_string
        )
        phases = (
            (False, request.correctness_workloads),
            (True, request.timing_workloads),
        )
        for timing, workloads in phases:
            if not workloads:
                continue
            for workload in workloads:
                if skip_reason(workload):
                    continue
                try:
                    self._smoke_reference_workload(
                        operator,
                        request,
                        workload,
                        timing=timing,
                        reference_device=reference_device,
                    )
                finally:
                    self._release_device_memory()

    def _smoke_reference_workload(
        self,
        operator: Any,
        request: EvaluateRequest,
        workload: Workload,
        *,
        timing: bool,
        reference_device: str,
    ) -> None:
        """Run one readiness call so its tensors die before outer cleanup."""

        if not timing and workload.output_path is not None:
            _load_golden_output(request, workload)
            return
        function = self._reference_function(operator, timing=timing)
        cpu_bases = make_cpu_bases(workload)
        _seed_globals(workload.seed)
        call, _ = make_call(
            request.definition,
            operator,
            workload,
            cpu_bases,
            reference_device,
        )
        _seed_globals(workload.seed)
        function(*call.args, **call.kwargs)
        self.device.synchronize(self.device_string)

    def _select_reference(self, request: EvaluateRequest) -> Any:
        """Select one oracle source for the whole formal evaluation."""

        validate_selected_policy()
        if request.definition.api_version == "v6.0":
            return load_operator_adapter(
                request.definition, "primary", self.oracle_path
            )
        try:
            primary = load_operator_adapter(
                request.definition, "primary", self.oracle_path
            )
            self._smoke_reference(primary, request)
            # Readiness may mutate inputs and module-level state.  Formal
            # evaluation starts with a freshly imported oracle module.
            return load_operator_adapter(
                request.definition, "primary", self.oracle_path
            )
        except Exception as primary_error:
            if isinstance(primary_error, SourcePolicyError):
                raise
            try:
                fallback = load_operator_adapter(
                    request.definition, "torch_fallback", self.oracle_path
                )
                self._smoke_reference(fallback, request)
                return load_operator_adapter(
                    request.definition, "torch_fallback", self.oracle_path
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    "primary reference readiness failed: "
                    f"{type(primary_error).__name__}: {primary_error}; "
                    "Torch fallback readiness failed: "
                    f"{type(fallback_error).__name__}: {fallback_error}"
                ) from fallback_error

    def _runtime_error_response(
        self,
        request: EvaluateRequest,
        message: str,
        *,
        reference_source: str = "primary",
    ) -> EvaluateResponse:
        correctness = {
            workload.name: CorrectnessResult(
                status=WorkloadStatus.RUNTIME_ERROR,
                message=message,
            )
            for workload in request.correctness_workloads
        }
        timing = {
            workload.name: TimingResult(
                status=WorkloadStatus.RUNTIME_ERROR,
                message=message,
            )
            for workload in request.timing_workloads
        }
        return aggregate_evaluation_response(
            request,
            correctness,
            timing,
            device=self.device_string,
            backend=self.backend,
            reference_source=reference_source,
        )

    def _correctness_gate(
        self,
        operator: Any,
        candidate: Any,
        request: EvaluateRequest,
        workload: Workload,
        *,
        timing_reference: bool = False,
    ) -> CorrectnessResult:
        (
            reference_call,
            reference_bound,
            candidate_call,
            candidate_bound,
            cpu_bases,
        ) = self._make_calls(operator, request, workload)
        reference_before = clone(dict(reference_bound.arguments))
        candidate_before = clone(dict(candidate_bound.arguments))
        effects = resolve_effects(request.definition, reference_bound)

        reference_output = None
        candidate_output = None
        reference_error: Exception | None = None
        candidate_error: Exception | None = None
        with _deterministic_correctness():
            if workload.output_path is not None:
                try:
                    reference_output = _load_golden_output(request, workload)
                    # A file-backed golden cannot carry storage aliasing.
                    # Restore declared output-to-input relationships in the
                    # synthetic reference state for in-place operators.
                    reference_outputs = named_outputs(
                        request.definition, reference_output
                    )
                    represented_mutations = set(
                        effects.returns_alias_of.values()
                    )
                    missing_mutations = (
                        set(effects.mutates) - represented_mutations
                    )
                    if missing_mutations:
                        raise ValueError(
                            "output_path cannot represent mutated parameters that "
                            "are not returned by alias: "
                            f"{sorted(missing_mutations)}"
                        )
                    for (
                        output_name,
                        parameter_name,
                    ) in effects.returns_alias_of.items():
                        if parameter_name in effects.mutates:
                            reference_bound.arguments[parameter_name] = (
                                reference_outputs[output_name]
                            )
                except Exception as exc:
                    reference_error = exc
            else:
                reference_function = self._reference_function(
                    operator, timing=timing_reference
                )
                try:
                    _seed_globals(workload.seed)
                    reference_output = reference_function(
                        *reference_call.args, **reference_call.kwargs
                    )
                except Exception as exc:  # expected-exception workloads include RuntimeError
                    reference_error = exc
            try:
                _seed_globals(workload.seed)
                candidate_output = candidate(
                    *candidate_call.args, **candidate_call.kwargs
                )
            except Exception as exc:  # see reference side above
                candidate_error = exc
        self.device.synchronize(self.device_string)

        if reference_error is not None or candidate_error is not None:
            error = reference_error or candidate_error
            side = "reference" if reference_error is not None else "candidate"
            return CorrectnessResult(
                status=WorkloadStatus.RUNTIME_ERROR,
                message=f"{side} raised unexpectedly:\n{type(error).__name__}: {error}",
            )

        reference_outputs = named_outputs(request.definition, reference_output)
        candidate_outputs = named_outputs(request.definition, candidate_output)
        alias_error = check_aliases(
            effects.returns_alias_of,
            candidate_outputs,
            candidate_bound,
            "candidate",
        )
        if workload.output_path is None:
            alias_error = check_aliases(
                effects.returns_alias_of,
                reference_outputs,
                reference_bound,
                "reference",
            ) or alias_error
        if alias_error:
            return CorrectnessResult(
                status=WorkloadStatus.INCORRECT_NUMERICAL,
                message=alias_error,
            )

        comparison = compare_outputs_and_mutations(
            request.definition,
            effects,
            reference_output,
            candidate_output,
            reference_before,
            candidate_before,
            reference_bound,
            candidate_bound,
            request.settings,
            workload,
            compare_declared_values=(
                request.api_version == "v6.0" or operator.validate is None
            ),
            compare_return_contract=(
                request.api_version == "v6.0"
                or not operator.validate_owns_return_contract
            ),
        )
        if not comparison.passed:
            return CorrectnessResult(
                status=WorkloadStatus.INCORRECT_NUMERICAL,
                max_absolute_error=comparison.max_absolute_error,
                max_relative_error=comparison.max_relative_error,
                matched_ratio=comparison.matched_ratio,
                message=comparison.message,
            )

        metrics: Dict[str, Any] = {}
        message = ""
        if operator.validate is not None:
            candidate_bound.apply_defaults()
            if request.api_version == "v6.2":
                inputs = {
                    parameter.name: candidate_bound.arguments[parameter.name]
                    for parameter in request.definition.parameters
                    if parameter.name in candidate_bound.arguments
                }
            else:
                inputs = [
                    candidate_bound.arguments[parameter.name]
                    for parameter in request.definition.parameters
                    if parameter.name in candidate_bound.arguments
                ]
            verdict = operator.validate(
                to_cpu(_output_list(reference_output)),
                to_cpu(_output_list(candidate_output)),
                to_cpu(inputs),
                workload_context(workload),
            )
            passed, message, metrics = _normalize_verdict(verdict)
            if not passed:
                return CorrectnessResult(
                    status=WorkloadStatus.INCORRECT_NUMERICAL,
                    max_absolute_error=comparison.max_absolute_error,
                    max_relative_error=comparison.max_relative_error,
                    matched_ratio=comparison.matched_ratio,
                    message=message,
                    metrics=metrics,
                )
        return CorrectnessResult(
            status=WorkloadStatus.PASSED,
            max_absolute_error=comparison.max_absolute_error,
            max_relative_error=comparison.max_relative_error,
            matched_ratio=comparison.matched_ratio,
            message=message,
            metrics=metrics,
        )

    def _timing_trial(
        self,
        operator: Any,
        candidate: Any,
        request: EvaluateRequest,
        workload: Workload,
    ) -> tuple[float, float]:
        reference_call, _, candidate_call, _, _ = self._make_calls(
            operator, request, workload
        )
        reference_invoke = (
            lambda call=reference_call: self._reference_function(
                operator, timing=True
            )(*call.args, **call.kwargs)
        )
        candidate_invoke = (
            lambda call=candidate_call: candidate(*call.args, **call.kwargs)
        )
        # Binding, compilation, and smoke launches all precede measured loops.
        reference_invoke()
        candidate_invoke()
        self.device.synchronize(self.device_string)
        reference_latency = self.device.time(
            reference_invoke,
            [],
            request.settings.warmup_ms,
            request.settings.benchmark_ms,
            self.device_string,
        )
        candidate_latency = self.device.time(
            candidate_invoke,
            [],
            request.settings.warmup_ms,
            request.settings.benchmark_ms,
            self.device_string,
        )
        return reference_latency, candidate_latency

    def _preflight_workload(
        self,
        operator: Any,
        candidate: Any,
        request: EvaluateRequest,
        workload: Workload,
    ) -> None:
        _, _, candidate_call, _, _ = self._make_calls(
            operator, request, workload
        )
        candidate(*candidate_call.args, **candidate_call.kwargs)
        self.device.synchronize(self.device_string)

    @target_source_policy
    def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        self.device.set_device(self.device_string)
        try:
            operator = self._select_reference(request)
        except Exception as exc:
            return self._runtime_error_response(
                request,
                str(exc),
            )
        try:
            candidate = load_implementation(request.implementation, request.definition)
        except Exception:
            return self._runtime_error_response(
                request,
                traceback.format_exc(),
                reference_source=operator.reference_source,
            )
        correctness: Dict[str, CorrectnessResult] = {}
        timing: Dict[str, TimingResult] = {}

        for workload in request.correctness_workloads:
            try:
                reason = skip_reason(workload)
                if reason:
                    correctness[workload.name] = CorrectnessResult(
                        status=WorkloadStatus.SKIP, message=reason)
                    continue
                result = self._correctness_gate(
                    operator,
                    candidate,
                    request,
                    workload,
                )
            except Exception:
                result = CorrectnessResult(
                    status=WorkloadStatus.RUNTIME_ERROR,
                    message=traceback.format_exc(),
                )
            finally:
                self._release_device_memory()
            correctness[workload.name] = result

        for workload in request.timing_workloads:
            try:
                reason = skip_reason(workload)
                if reason:
                    timing[workload.name] = TimingResult(status=WorkloadStatus.SKIP, message=reason)
                    continue
                # v6.2 follows the framework benchmark contract: correctness is
                # owned exclusively by correctness workloads. Timing workloads
                # only need to execute and produce valid latency measurements.
                # Keep the v6.0 gate unchanged for wire-level compatibility.
                if request.api_version == "v6.0":
                    try:
                        gate = self._correctness_gate(
                            operator,
                            candidate,
                            request,
                            workload,
                            timing_reference=True,
                        )
                    finally:
                        self._release_device_memory()
                    if gate.status != WorkloadStatus.PASSED:
                        timing[workload.name] = TimingResult(
                            status=gate.status,
                            message=(
                                "timing correctness/effects gate failed: "
                                f"{gate.message}"
                            ),
                        )
                        continue

                ref_latencies = []
                candidate_latencies = []
                for _ in range(request.settings.num_trials):
                    try:
                        reference_latency, candidate_latency = self._timing_trial(
                            operator, candidate, request, workload
                        )
                        ref_latencies.append(reference_latency)
                        candidate_latencies.append(candidate_latency)
                    finally:
                        self._release_device_memory()
                reference_latency = sum(ref_latencies) / len(ref_latencies)
                candidate_latency = sum(candidate_latencies) / len(candidate_latencies)
                result = TimingResult(
                    status=WorkloadStatus.PASSED,
                    latency_ms=candidate_latency,
                    reference_latency_ms=reference_latency,
                    speedup=reference_latency / candidate_latency,
                )
            except Exception:
                result = TimingResult(
                    status=WorkloadStatus.RUNTIME_ERROR,
                    message=traceback.format_exc(),
                )
            timing[workload.name] = result

        response = aggregate_evaluation_response(
            request,
            correctness,
            timing,
            device=self.device_string,
            backend=self.backend,
            reference_source=operator.reference_source,
        )
        return response

    @target_source_policy
    def preflight(self, request: EvaluateRequest) -> Dict[str, Any]:
        """ABI-check and smoke-launch every timing specialization once."""
        hack = check_candidate_admission(
            request.implementation, operator_name=request.definition.name
        )
        if hack.is_hack:
            return {
                "api_version": KERNELGEN_API_VERSION, "status": "FAILED",
                "stage": "candidate_admission", "is_hack": True,
                "hack_reason": hack.hack_reason, "log": hack.hack_reason,
                "per_workload": {},
            }
        self.device.set_device(self.device_string)
        log_graylist_hits(request.implementation, vendor=self.backend)
        validate_selected_policy()
        operator = load_operator_adapter(
            request.definition, oracle_path=self.oracle_path
        )
        candidate = load_implementation(request.implementation, request.definition)
        # Preflight is deliberately candidate-only.  Correctness workloads need
        # reference execution and assertions, which belong to evaluate().
        workloads = {workload.name: workload for workload in request.timing_workloads}
        results: Dict[str, Any] = {}
        for name, workload in workloads.items():
            try:
                reason = skip_reason(workload)
                if reason:
                    results[name] = {"status": "SKIP", "error": "", "reason": reason}
                    continue
                self._preflight_workload(
                    operator, candidate, request, workload
                )
                results[name] = {"status": "PASSED", "error": ""}
            except Exception:
                results[name] = {
                    "status": "RUNTIME_ERROR",
                    "error": traceback.format_exc(),
                }
            finally:
                self._release_device_memory()
        status = (
            "PASSED"
            if all(item["status"] in {"PASSED", "SKIP"} for item in results.values())
            else "RUNTIME_ERROR"
        )
        return {
            "api_version": KERNELGEN_API_VERSION,
            "status": status,
            "is_hack": hack.is_hack,
            "hack_reason": hack.hack_reason,
            "log": hack.log,
            "per_workload": results,
        }
