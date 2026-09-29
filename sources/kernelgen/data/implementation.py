"""Authoritative implementation-language profiles for KernelGen tasks.

Agent roles describe the optimization process.  A profile supplies the
language-specific source contract that is injected into each task and enforced
by the MCP/eval path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ImplementationLanguage(str, Enum):
    """Implementation languages currently supported end-to-end by KernelGen."""

    TRITON = "triton"


@dataclass(frozen=True)
class ImplementationProfile:
    language: ImplementationLanguage
    dsl: str
    source_format: str
    entry_point: str
    kernel_marker: str
    launch_form: str
    wrapper_policy: str
    forbidden_fallback: str


_PROFILES = {
    ImplementationLanguage.TRITON: ImplementationProfile(
        language=ImplementationLanguage.TRITON,
        dsl="triton.language",
        source_format="self-contained Python module",
        entry_point="main.py::run",
        kernel_marker="@triton.jit",
        launch_form="kernel[grid](...)",
        wrapper_policy=(
            "output allocation (for example torch.empty/torch.empty_like); "
            "tensor metadata reads; zero-copy view/layout operations such as "
            "view, unsqueeze, squeeze, transpose, permute, and narrow; "
            "scalar/grid calculation and runtime device queries; JIT kernel "
            "launch; and preflight-allowlisted, output-independent state "
            "maintenance/data movement after device-kernel launches, such as "
            "torch.cat(..., out=<existing reference-mutated input>). Any exact "
            "API form remains subject to preflight_kernel."
        ),
        forbidden_fallback=(
            "framework or host-side computation that produces, or contributes "
            "to, the returned numerical output (for example torch.matmul, "
            "torch.softmax, framework convolution/norm/activation/reduction); "
            "cross-call tensor caches; and tensor data movement not explicitly "
            "allowed by preflight. This is not a blanket ban on all torch APIs."
        ),
    ),
}


def get_implementation_profile(
    language: ImplementationLanguage | str,
) -> ImplementationProfile:
    normalized = ImplementationLanguage(language)
    return _PROFILES[normalized]


def render_implementation_profile(
    language: ImplementationLanguage | str,
    *,
    evaluator_kind: str | None = None,
) -> str:
    profile = get_implementation_profile(language)
    return (
        "<implementation_profile>\n"
        f"Language: {profile.language.value}\n"
        f"DSL: {profile.dsl}\n"
        f"Source format: {profile.source_format}\n"
        f"Entry point: {profile.entry_point}\n"
        f"Kernel marker: {profile.kernel_marker}\n"
        f"Launch form: {profile.launch_form}\n"
        f"Wrapper policy: {profile.wrapper_policy}\n"
        f"Forbidden fallback: {profile.forbidden_fallback}\n"
        "Candidate admission: preflight_kernel owns the source-policy gate, ABI and compile/smoke checks. "
        "Correct only the candidate after rejection; do not fabricate a measured round or request BLOCK for a candidate policy violation.\n"
        "Protected state: never modify Torch source/registrations, baseline or global operators, "
        "compiler/JIT internals, pytest/reference code, tolerances, workloads, timing or measurement state. "
        "This applies to imports as well as run(); temporarily patching and restoring state is also forbidden.\n"
        "Environment ownership: never call empty_cache() or change Torch's shared memory-allocation policy "
        "(for example _set_allocator_settings, change_current_allocator, or set_per_process_memory_fraction), "
        "write/delete/update process environment variables, write runtime files, install compiler shims, "
        "or launch subprocesses from the submitted candidate. The evaluator owns process cleanup and environment setup. "
        "Read-only environment/device/shape/stride/dtype queries, ordinary tensor allocation/release, "
        "and legal per-kernel launch settings remain allowed. A triton.set_allocator callback that only "
        "provides required kernel workspace using ordinary allocation is allowed; it must not change "
        "Torch allocation policy, clear caches, or modify evaluation state. "
        "You may write candidate and diagnostic scripts "
        "in the authorized workspace, but must not use diagnostics to patch the protected environment. "
        "Reference, compiler and injection defects belong to their maintainers; preserve a reproducible failure "
        "instead of compensating inside the candidate.\n"
        + (
            "Gems adapter protection: never modify Gems source, registration or testing APIs; "
            "never install override_registered_op/override_gems_op contexts, even temporarily.\n"
            if evaluator_kind == "flaggems" else ""
        )
        + "Do not use zero-grid/dummy kernels or unrelated side-stream work to satisfy kernel requirements. "
        "The Server-maintained metadata policy may exempt reviewed lift_fresh identity and unsqueeze_ inplace-view implementations "
        "from the JIT/launch requirement; never self-declare an exemption or change alias/input semantics.\n"
        "</implementation_profile>"
    )


def _target_key(target_hardware: str) -> str:
    return "".join(
        character
        for character in str(target_hardware or "").lower()
        if character.isalnum()
    )


def render_target_compatibility_rules(
    language: ImplementationLanguage | str,
    target_hardware: str,
) -> str:
    """Render narrowly scoped cross-device portability requirements."""

    if ImplementationLanguage(language) is not ImplementationLanguage.TRITON:
        return ""

    target = _target_key(target_hardware)
    if "910b" in target:
        return (
            "<target_compatibility>\n"
            "Ascend 910B portability requirement:\n"
            "- The active development/evaluation service may report Ascend910B4 "
            "or Ascend910B3, but the required submission target is Ascend910B2.\n"
            "- Do not hard-code a physical AI Core, Vector Core, SM, or resident-"
            "program count observed on the development device. Numeric tile sizes "
            "derived from the workload are allowed; physical-core launch constants "
            "are not.\n"
            "- When launch concurrency depends on physical resources, query "
            "driver.active.utils.get_device_properties(torch.npu.current_device()) "
            "on the target. Use num_aicore for Cube/tl.dot/CV kernels and "
            "num_vectorcore for pure Vector kernels, and cap the grid by available "
            "work. A task-derived grid does not require a physical-core query.\n"
            "- Persistent kernels must stride by tl.num_programs(0), the actual "
            "launched program count, rather than a B4/B3-derived constant.\n"
            "- Avoid B4/B3-only APIs, compiler options, layouts, or resource "
            "assumptions unless their B2 support is verified. Keep a B2-safe "
            "alternative when a target-specific optimization is retained.\n"
            "</target_compatibility>"
        )

    if "c500" in target or "c550" in target:
        return (
            "<target_compatibility>\n"
            "MetaX C500 portability requirement:\n"
            "- Code developed or optimized on C550 must remain source-correct and "
            "performance-portable to C500.\n"
            "- Even when C500 and C550 report compatible XCORE and warp targets, "
            "do not assume identical clocks, power limits, occupancy, cache "
            "behavior, or optimal launch parameters.\n"
            "- Do not hard-code C550 physical compute-unit counts, resident program "
            "counts, clock rates, cache capacities, or occupancy assumptions. "
            "When launch concurrency depends on hardware, query target runtime "
            "properties and cap the grid by available work. Persistent kernels must "
            "stride by tl.num_programs(0).\n"
            "- Keep the algorithm shared, but allow C500/C550-specific BLOCK sizes, "
            "num_warps, num_stages, pipeline, scenario, and persistent-grid "
            "settings. Include the device family in dispatch and autotune-cache "
            "keys; never reuse a C550 autotune winner as C500 performance evidence.\n"
            "- Do not perform expensive autotuning inside the timed run() path. "
            "Prefer mechanisms that transfer across devices, such as less memory "
            "traffic, coalesced access, less redundant computation, and fewer "
            "spills.\n"
            "</target_compatibility>"
        )

    return ""
