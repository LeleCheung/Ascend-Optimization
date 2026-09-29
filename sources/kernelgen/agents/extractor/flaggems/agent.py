"""Legacy model-driven FlagGems Definition/Workload extraction."""

from __future__ import annotations

import ast
import json as json_mod
from pathlib import Path
from typing import Any

from kernelgen.agents.extractor.flaggems.abi_validation import validate_run_abi
from kernelgen.agents.extractor.flaggems.guidance import (
    GENERAL_EXTRACTION_PRINCIPLES,
    OPERATOR_DEFINITION_SHELLS,
    OPERATOR_EXTRACTION_GUIDANCE,
    REFERENCE_DTYPE_GUIDANCE,
)
from kernelgen.agents.extractor.flaggems.models import (
    V6_GROUPS,
    FlagGemsExtractorInput,
    FlagGemsExtractorOutput,
)
from kernelgen.agents.extractor.flaggems.source_inventory import (
    FlagGemsSourceInventory,
    collect_source_inventory,
)
from kernelgen.agents.extractor.flaggems.source_validation import (
    benchmark_has_blas_column_major_control,
    pytest_requires_equal_nan,
    pytest_keyword_scalar_requirements,
    resolve_core_addmm_timing_profile,
    validate_core_addmm_timing,
    validate_core_blas_layout,
    validate_equal_nan_coverage,
    validate_keyword_scalar_coverage,
)
from kernelgen.framework.base import BaseAgent
from kernelgen.framework.contract import extract_json, render_contract


_GROUP_SYNONYMS = {
    "blas": "gemm",
    "matmul": "gemm",
    "linear_algebra": "linalg",
    "embedding": "indexing",
    "gather": "indexing",
    "nested": "shape",
    "view": "shape",
    "fused": "norm",
    "nn": "pointwise",
    "activation": "pointwise",
    "elementwise": "pointwise",
    "unary": "pointwise",
    "binary": "pointwise",
    "optimizer": "pointwise",
    "loss": "reduction",
    "pool": "pooling",
    "dropout": "pointwise",
    "distribution": "pointwise",
    "random": "pointwise",
}

_CASE_LIST_SCHEMA = "flaggems.benchmark-case-list/v2"


def _load_authoritative_cases(path: str, operator: str) -> list[dict[str, Any]]:
    report_path = Path(path).expanduser().resolve()
    try:
        report = json_mod.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json_mod.JSONDecodeError) as exc:
        raise ValueError(f"{operator}: cannot read --list-cases report: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != _CASE_LIST_SCHEMA:
        raise ValueError(f"{operator}: invalid --list-cases report schema")
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError(f"{operator}: --list-cases benchmarks must be a list")
    matches = [
        benchmark
        for benchmark in benchmarks
        if isinstance(benchmark, dict) and benchmark.get("op_name") == operator
    ]
    if not matches:
        raise ValueError(
            f"{operator}: --list-cases report has no matching benchmark"
        )
    cases: list[dict[str, Any]] = []
    for benchmark in matches:
        if (
            benchmark.get("schema_version") != _CASE_LIST_SCHEMA
            or benchmark.get("phase") != "timing"
            or benchmark.get("level") != "core"
        ):
            raise ValueError(f"{operator}: --list-cases benchmark is not core timing")
        benchmark_cases = benchmark.get("cases")
        if not isinstance(benchmark_cases, list):
            raise ValueError(f"{operator}: --list-cases cases must be a list")
        cases.extend(benchmark_cases)
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(
        not isinstance(case_id, str) or not case_id for case_id in case_ids
    ):
        raise ValueError(f"{operator}: every listed timing case needs a case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{operator}: listed timing case IDs must be unique")
    return cases


def _coerce_common_literals(payload: Any) -> None:
    """Normalize only unambiguous non-ABI vocabulary.

    ABI-bearing fields are deliberately not repaired. In particular, an old
    ``flaggems_`` name, a v5 API version, or a wrong parameter signature must fail
    validation instead of being silently rewritten.
    """

    if not isinstance(payload, dict):
        return
    for result in payload.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        definition = result.get("definition")
        if isinstance(definition, dict):
            device = definition.get("reference_device")
            if isinstance(device, str) and device.lower() in {"target", "cpu"}:
                definition["reference_device"] = device.lower()
        group = result.get("group")
        if isinstance(group, str) and group not in V6_GROUPS:
            normalized_group = _GROUP_SYNONYMS.get(group.lower())
            if normalized_group is not None:
                result["group"] = normalized_group


class FlagGemsExtractorAgent(BaseAgent):
    """Read/audit the retired translated-catalog format.

    New V6 flows must use ``extract_flaggems_adapter_spec``.
    """

    name = "flaggems_extractor"
    InputModel = FlagGemsExtractorInput
    OutputModel = FlagGemsExtractorOutput

    def preprocess(self, inp: FlagGemsExtractorInput, runtime) -> str:
        source = collect_source_inventory(inp.flaggems_repo, inp.operator)
        self._source_inventory = source
        authoritative_cases = (
            _load_authoritative_cases(inp.case_list_path, inp.operator)
            if inp.case_list_path is not None
            else None
        )
        self._authoritative_timing_cases = authoritative_cases

        labels = list(source.labels)
        bench_yaml_shapes = source.benchmark_shapes
        operator_symbols = list(source.operator_symbols)
        impl_file = source.implementation_file
        test_files = list(source.test_files)
        benchmark_files = list(source.benchmark_files)
        test_file = source.test_file
        bench_file = source.benchmark_file

        self._source_test_file = test_file
        self._source_bench_file = bench_file if bench_file.exists() else None
        self._source_test_files = test_files
        self._source_bench_files = benchmark_files
        implementation_status = source.implementation_status
        helper_context = source.helper_context

        role = self._role_for(runtime)
        context_parts = ["## Injected Context (from Python — authoritative)"]
        context_parts.append(f"- **Operator**: `{inp.operator}`")
        context_parts.append(
            f"- **Registered FlagGems symbol(s)**: `{operator_symbols}`"
        )
        context_parts.append(
            f"- **Implementation status**: {implementation_status}"
        )
        context_parts.append(
            f"- **Labels**: {labels if labels else '(not found in catalog)'}"
        )
        context_parts.append(
            f"- **Impl file**: `{impl_file}` "
            f"{'✓ exists' if impl_file.exists() else '✗ NOT FOUND'}"
        )
        if source.public_signatures:
            rendered_signatures = "\n".join(
                signature.source.rstrip() for signature in source.public_signatures
            )
            context_parts.append(
                "- **Deterministically resolved public Python signature(s)**: "
                "These are authoritative for Definition `name` and `parameters`; "
                "the postprocessor compares matching exports again.\n```python\n"
                f"{rendered_signatures}\n```"
            )
        context_parts.append(
            f"- **Test file**: `{test_file}` "
            f"{'✓ exists' if test_file.exists() else '✗ NOT FOUND'}"
        )
        if len(test_files) > 1:
            context_parts.append(
                "- **Additional pytest files marked for this operator**: "
                + ", ".join(f"`{path}`" for path in test_files[1:])
            )
        context_parts.append(
            f"- **Benchmark file**: `{bench_file}` "
            f"{'✓ exists' if bench_file.exists() else '✗ NOT FOUND'}"
        )
        if len(benchmark_files) > 1:
            context_parts.append(
                "- **Additional benchmark files marked for this operator**: "
                + ", ".join(f"`{path}`" for path in benchmark_files[1:])
            )
        if benchmark_files:
            context_parts.append(
                "- **Mandatory core timing profile**: The delivery runner is "
                "`tools/test-op.sh`, whose performance command is "
                "`pytest -s <benchmark-file> --level core --record log`. Reproduce "
                "that core execution sequence exactly: preserve dtype order, source "
                "shape order, duplicate records, input layout, explicit/omitted "
                "arguments, and skips. Do not sample, deduplicate, reorder, or add "
                "cases reached only by bare-pytest's default comprehensive level. "
                "At least one target-reference result must contain timing workloads; "
                "CPU-only oracle results remain correctness-only."
            )
            if authoritative_cases is not None:
                context_parts.append(
                    "- **Authoritative target `--list-cases` sequence**: Emit exactly "
                    "one timing workload per record, in this order. Copy each "
                    "`case_id` into Workload.name and materialize its dtype, shape, "
                    "params and layout by tracing the benchmark builder. The "
                    "postprocessor rejects any count/order drift.\n```json\n"
                    f"{json_mod.dumps(authoritative_cases, indent=2)}\n```"
                )
            if benchmark_has_blas_column_major_control(benchmark_files):
                context_parts.append(
                    "- **Mandatory core BLAS layout**: At `--level core`, "
                    "`BlasBenchmark.get_input_iter()` calls the input function only "
                    "with `b_column_major=False`. Emit only the contiguous/row-major "
                    "branch. `b_column_major` is a benchmark generator control, not "
                    "public ABI, and `generator_params.column_major=true` belongs "
                    "only to comprehensive timing. The postprocessor rejects it."
                )
            core_addmm_profile = resolve_core_addmm_timing_profile(
                inp.operator,
                benchmark_files,
                bench_yaml_shapes,
            )
            if core_addmm_profile is not None:
                context_parts.append(
                    "- **Deterministically resolved addmm_ core sequence**: Emit "
                    "exactly this profile. Source shape records remain distinct even "
                    "when dropping B produces duplicate tensor shapes. The "
                    "postprocessor checks every workload in order.\n```json\n"
                    f"{json_mod.dumps(core_addmm_profile.as_prompt_payload(), indent=2)}"
                    "\n```"
                )
        else:
            context_parts.append(
                "- **Timing source rule**: No benchmark exists. Every result must "
                "set `timing_workloads` to an empty list. Never infer timing cases "
                "from pytest, implementation code, Torch schema, or core_shapes.yaml."
            )
        if bench_yaml_shapes:
            context_parts.append(
                "- **Resolved core benchmark shapes**: This already applies "
                "benchmark-local `set_shapes()` precedence over the YAML class "
                "fallback.\n```json\n"
                f"{bench_yaml_shapes}\n```"
            )
        else:
            context_parts.append(
                f"- **Resolved core benchmark shapes**: (not statically resolved "
                f"for `{inp.operator}`; read the benchmark's `set_shapes()` and "
                "Benchmark base-class fallback)"
            )
        if helper_context:
            context_parts.append(
                "- **Resolved directly referenced helpers (authoritative; do not "
                "reopen/search these files)**:\n```python\n"
                f"{helper_context}\n```"
            )
        keyword_scalar_requirements = pytest_keyword_scalar_requirements(
            inp.operator,
            test_files,
            helper_context,
        )
        if keyword_scalar_requirements:
            context_parts.append(
                "- **Mandatory pytest keyword-value coverage**: Every listed value "
                "must occur in at least one correctness workload for that public "
                "keyword. This is deterministically traced from pytest "
                "parametrization and local assignments; the postprocessor rejects "
                "missing values.\n```json\n"
                f"{json_mod.dumps(keyword_scalar_requirements, indent=2)}\n```"
            )
        if pytest_requires_equal_nan(inp.operator, test_files):
            context_parts.append(
                "- **Mandatory NaN comparison semantics**: The resolved pytest "
                "explicitly passes `equal_nan=True`. Every correctness workload "
                "must set `tolerance.equal_nan=true`; the postprocessor rejects "
                "omissions."
            )

        context_parts.append(
            "- **Protocol**: v6.0 Definition + Workloads. Definition `parameters` "
            "is the single public ABI; Definition `name` is the real public callable "
            "without a `flaggems_` prefix; every Workload has a restricted `call`; "
            "mutation and aliasing are in `effects`; custom runtime values use "
            "`generator_params` and fixed `gen_inputs(ctx, device)`. Do not emit v5 "
            "inputs lists or configurable input/correctness entrypoint names."
        )
        context_parts.append(
            f"- **Candidate/reference dtype rule**: {REFERENCE_DTYPE_GUIDANCE}"
        )
        context_parts.append(
            "- **General extraction principles (mandatory)**:\n"
            f"{GENERAL_EXTRACTION_PRINCIPLES}"
        )
        guidance_name = (
            inp.operator[1:]
            if inp.operator.startswith("_") and not inp.operator.startswith("__")
            else inp.operator
        )
        operator_guidance = OPERATOR_EXTRACTION_GUIDANCE.get(
            inp.operator
        ) or OPERATOR_EXTRACTION_GUIDANCE.get(guidance_name)
        if operator_guidance:
            context_parts.append(
                f"- **Operator-specific extraction rule**: {operator_guidance}"
            )
        definition_shell = OPERATOR_DEFINITION_SHELLS.get(inp.operator)
        if definition_shell:
            context_parts.append(
                "- **Required structural shell**: Copy the following group and "
                "Definition fields exactly; add only the description and extracted "
                "workloads. Do not rename fields or rewrite the reference source.\n"
                f"```json\n{json_mod.dumps(definition_shell, indent=2)}\n```"
            )

        context_block = "\n".join(context_parts)
        params_block = ""
        if inp.default_params:
            params_block = (
                "\n\nDefault parameters provided by user:\n"
                f"```json\n{json_mod.dumps(inp.default_params, indent=2)}\n```\n"
                "Use these only where they agree with the source tests."
            )

        contract = render_contract(self.OutputModel)
        return "\n\n".join(
            filter(
                None,
                [
                    role,
                    context_block,
                    params_block,
                    f"--- OUTPUT CONTRACT ---\n{contract}",
                ],
            )
        )

    def postprocess(self, raw: str, runtime) -> FlagGemsExtractorOutput:
        payload = extract_json(raw)
        _coerce_common_literals(payload)
        output = self.OutputModel.model_validate(payload)

        source: FlagGemsSourceInventory | None = getattr(
            self,
            "_source_inventory",
            None,
        )
        if source is None:
            raise ValueError(
                "extractor source context is missing; preprocess must resolve pytest "
                "before postprocess"
            )
        test_file = source.test_file
        benchmark_files = list(source.benchmark_files)
        keyword_scalar_requirements = pytest_keyword_scalar_requirements(
            source.operator,
            source.test_files,
            source.helper_context,
        )
        for result in output.results:
            matching_signatures = [
                signature
                for signature in source.public_signatures
                if signature.public_name == result.definition.name
            ]
            if source.public_signatures and not matching_signatures:
                expected = sorted(
                    signature.public_name for signature in source.public_signatures
                )
                raise ValueError(
                    f"{result.definition.name}: Definition name does not match the "
                    f"deterministically resolved FlagGems public callable {expected}"
                )
            for signature in matching_signatures:
                signature_module = ast.parse(signature.source)
                function = next(
                    node
                    for node in signature_module.body
                    if isinstance(node, ast.FunctionDef)
                )
                validate_run_abi(
                    result.definition.name,
                    result.definition.parameters,
                    function,
                    "FlagGems public source",
                )
            if not result.correctness_workloads:
                raise ValueError(
                    f"{result.definition.name}: pytest-backed extraction requires "
                    "at least one correctness workload"
                )
            validate_keyword_scalar_coverage(
                result,
                keyword_scalar_requirements,
            )
            validate_equal_nan_coverage(
                result,
                pytest_requires_equal_nan(source.operator, source.test_files),
            )
            description = result.definition.description.lower()
            if "no pytest" in description or (
                "pytest" in description and "not found" in description
            ):
                raise ValueError(
                    f"{result.definition.name}: description contradicts resolved "
                    f"pytest {test_file}"
                )

        target_timing = [
            workload
            for result in output.results
            if result.definition.reference_device == "target"
            for workload in result.timing_workloads
        ]
        all_timing = [
            workload
            for result in output.results
            for workload in result.timing_workloads
        ]
        authoritative_cases: list[dict[str, Any]] | None = getattr(
            self,
            "_authoritative_timing_cases",
            None,
        )
        if authoritative_cases is not None:
            if len(all_timing) != len(authoritative_cases):
                raise ValueError(
                    f"{source.operator}: extracted {len(all_timing)} timing "
                    f"workloads but --list-cases returned {len(authoritative_cases)}"
                )
            for workload, case in zip(all_timing, authoritative_cases, strict=True):
                workload.name = case["case_id"]
                expected_dtype = case.get("dtype")
                if not isinstance(expected_dtype, str):
                    continue
                expected_dtype = expected_dtype.removeprefix("torch.")
                extracted_dtypes = {
                    spec.dtype
                    for spec in workload.inputs.values()
                    if spec.dtype is not None
                }
                if extracted_dtypes and expected_dtype not in extracted_dtypes:
                    raise ValueError(
                        f"{source.operator}/{workload.name}: listed dtype "
                        f"{expected_dtype} not in extracted inputs "
                        f"{sorted(extracted_dtypes)}"
                    )
            renamed = [workload.name for workload in all_timing]
            correctness_names = {
                workload.name
                for result in output.results
                for workload in result.correctness_workloads
            }
            if len(renamed) != len(set(renamed)) or correctness_names & set(renamed):
                raise ValueError(
                    f"{source.operator}: authoritative timing case IDs are not "
                    "globally unique across extracted workloads"
                )
        if not benchmark_files and all_timing:
            raise ValueError("timing workloads require a resolved FlagGems benchmark")
        if benchmark_files and not target_timing:
            target_results = [
                result
                for result in output.results
                if result.definition.reference_device == "target"
            ]
            if target_results:
                raise ValueError(
                    "resolved FlagGems benchmark requires at least one timing "
                    "workload for a target-reference result"
                )
        if benchmark_files:
            has_blas_column_major_control = benchmark_has_blas_column_major_control(
                benchmark_files
            )
            core_addmm_profile = resolve_core_addmm_timing_profile(
                source.operator,
                benchmark_files,
                source.benchmark_shapes,
            )
            for result in output.results:
                description = result.definition.description.lower()
                if "no benchmark" in description or (
                    "benchmark" in description and "not found" in description
                ):
                    raise ValueError(
                        f"{result.definition.name}: description contradicts "
                        f"resolved benchmark files {benchmark_files}"
                    )
                if has_blas_column_major_control and result.timing_workloads:
                    validate_core_blas_layout(result)
                if core_addmm_profile is not None and result.timing_workloads:
                    validate_core_addmm_timing(result, core_addmm_profile)
        return output
