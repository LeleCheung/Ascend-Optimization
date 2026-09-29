# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Standalone implementation runner used by vendor profiler commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..evaluation.loader import load_implementation, load_operator_adapter
from ..evaluation.workload_runtime import make_call, make_cpu_bases
from ..runtime.device import configure_device, get_device
from ..runtime.source_policy import (
    source_policy_scope,
    skip_reason,
    validate_selected_policy,
)
from .capture import capture_scope, prepare_profile_compiler
from .models import ProfileTarget

RUNNER_COMPLETION_MARKER = "runner-completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--implementation-source-root", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    completion_marker = data_dir / RUNNER_COMPLETION_MARKER
    completion_marker.unlink(missing_ok=True)
    target = ProfileTarget.model_validate_json(
        (data_dir / "target.json").read_text(encoding="utf-8")
    )
    backend = target.expected_backend
    prepare_profile_compiler(backend)
    configure_device(backend, timing_strategy="triton")
    device = get_device(backend)
    device.set_device(args.device)

    with source_policy_scope(backend, target.definition.source_policy_id):
        validate_selected_policy()
        reason = skip_reason(target.workload)
        if reason:
            raise ValueError(
                f"cannot profile an inapplicable source workload: {reason}"
            )
        _run_profile(args, target, device, completion_marker)


def _run_profile(args, target, device, completion_marker):
    backend = target.expected_backend

    operator = load_operator_adapter(
        target.definition,
        oracle_path=target.oracle_path,
    )
    candidate = load_implementation(
        target.implementation,
        target.definition,
        source_root=args.implementation_source_root,
    )
    generated, _ = make_call(
        target.definition,
        operator,
        target.workload,
        make_cpu_bases(target.workload),
        args.device,
    )

    def invoke() -> None:
        candidate(*generated.args, **generated.kwargs)

    for _ in range(args.warmup):
        invoke()
    device.synchronize(args.device)
    with capture_scope(backend):
        for _ in range(args.iterations):
            invoke()
        device.synchronize(args.device)
    completion_marker.write_text("completed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
