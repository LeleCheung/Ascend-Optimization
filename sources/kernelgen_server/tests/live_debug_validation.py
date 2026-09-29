#!/usr/bin/env python3
"""Validate Debug Job and status APIs against a running KernelGen Server."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from kernelgen_server import DebugJobRequest, DebugSourceFile
from kernelgen_server.client import (
    download_debug_artifacts,
    status,
    submit_debug_job,
)


CUDA_COMPATIBLE_BACKENDS = {
    "cuda",
    "hygon",
    "iluvatar",
    "kunlunxin",
    "metax",
    "thead",
}
VISIBILITY_VARIABLES = {
    "cuda": ["CUDA_VISIBLE_DEVICES"],
    "npu": ["ASCEND_RT_VISIBLE_DEVICES"],
    "musa": ["MUSA_VISIBLE_DEVICES", "MTHREADS_VISIBLE_DEVICES"],
    "mlu": ["MLU_VISIBLE_DEVICES"],
    "hygon": [
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ],
    "metax": ["CUDA_VISIBLE_DEVICES"],
    "iluvatar": ["CUDA_VISIBLE_DEVICES"],
    "kunlunxin": ["CUDA_VISIBLE_DEVICES"],
    "thead": ["CUDA_VISIBLE_DEVICES"],
}
def validate(
    server_url: str,
    expected_backend: str,
    expected_device_count: int,
    expected_workers: int | None,
) -> dict[str, Any]:
    service = status(server_url)
    assert service["backend"] == expected_backend, service
    assert len(service["devices"]) == expected_device_count, service
    if expected_workers is not None:
        assert service["workers"] == expected_workers, service
    assert service["debug"]["enabled"] is True, service
    assert service["debug"]["security_boundary"] is False, service
    assert service["target"]["backend"] == (
        "ascend" if expected_backend == "npu" else expected_backend
    )
    assert "software" in service

    script = """
import json
import os
from pathlib import Path

backend = os.environ["KGS_BACKEND"]
if backend == "npu":
    import torch_npu
elif backend == "musa":
    import torch_musa
elif backend == "mlu":
    import torch_mlu
import torch

names = [
    "ASCEND_RT_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "MLU_VISIBLE_DEVICES",
    "MTHREADS_VISIBLE_DEVICES",
    "MUSA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
]
payload = {
    "assigned": os.environ["KGS_ASSIGNED_DEVICE"],
    "backend": backend,
    "catalog_exists": Path(os.environ["KGS_CATALOG_ROOT"]).is_dir(),
    "device": os.environ["KGS_DEVICE"],
    "device_value": torch.ones(1, device=os.environ["KGS_DEVICE"]).cpu().item(),
    "visibility": {name: os.environ[name] for name in names if name in os.environ},
}
output = Path(os.environ["KGS_DEBUG_ARTIFACTS"]) / "result.json"
output.write_text(json.dumps(payload), encoding="utf-8")
print("debug-live-ok")
"""
    completed = submit_debug_job(
        DebugJobRequest(
            command=["{python}", "debug.py"],
            files=[DebugSourceFile(path="debug.py", content=script)],
            timeout_seconds=30,
        ),
        server_url,
    )
    assert completed.status == "SUCCEEDED", completed
    assert completed.stdout.splitlines()[0] == "debug-live-ok", completed
    assert completed.device in service["devices"], completed
    assert len(completed.artifacts) == 1, completed

    with tempfile.TemporaryDirectory(prefix="kgs-live-debug-") as output_dir:
        downloaded = download_debug_artifacts(
            completed,
            output_dir,
            server_url,
        )
        payload = json.loads(downloaded["a001"].read_text(encoding="utf-8"))
    expected_runtime_device = (
        "cuda:0"
        if expected_backend in CUDA_COMPATIBLE_BACKENDS
        else f"{expected_backend}:0"
    )
    assert payload["backend"] == expected_backend, payload
    assert payload["device"] == expected_runtime_device, payload
    assert payload["device_value"] == 1.0, payload
    assert payload["catalog_exists"] is True, payload
    expected_visibility = VISIBILITY_VARIABLES[expected_backend]
    assert expected_visibility[0] in payload["visibility"], payload
    for name, value in payload["visibility"].items():
        if name in expected_visibility:
            assert value and "," not in value, payload

    timed_out = submit_debug_job(
        DebugJobRequest(
            command=["{python}", "-c", "import time; time.sleep(30)"],
            timeout_seconds=1,
        ),
        server_url,
    )
    assert timed_out.status == "TIMEOUT", timed_out

    return {
        "status": "PASSED",
        "backend": expected_backend,
        "devices": service["devices"],
        "workers": service["workers"],
        "target": service["target"],
        "software": service["software"],
        "debug_job": {
            "device": completed.device,
            "visibility": payload["visibility"],
            "artifacts": len(completed.artifacts),
        },
        "timeout": timed_out.status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--expected-backend", required=True)
    parser.add_argument("--expected-device-count", required=True, type=int)
    parser.add_argument("--expected-workers", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate(
        args.server,
        args.expected_backend,
        args.expected_device_count,
        args.expected_workers,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
