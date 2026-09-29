"""Stateless MCP adapter for KernelGen Server."""

from __future__ import annotations

from typing import Any, Dict, Optional

from kernelgen.data.catalog import DEFAULT_CATALOG_NAME


def evaluate_only(
    kernel_path: str,
    definition: str,
    *,
    catalog_name: str = DEFAULT_CATALOG_NAME,
    target_hardware: str = "",
    server_url: str = "",
    entry_point: str = "main.py::run",
    destination_passing_style: bool = False,
    language: Optional[str] = None,
    warmup_ms: int = 1000,
    benchmark_ms: int = 100,
    num_trials: int = 1,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """Evaluate one kernel without writing optimization or ledger state."""
    try:
        from kernelgen.tools.kernelgen_server_adapter import evaluate_kernel_file
    except Exception as exc:  # noqa: BLE001
        message = f"cannot import KernelGen Server adapter: {exc}"
        return {"status": "ERROR", "log": message, "error": message}

    return evaluate_kernel_file(
        kernel_path=kernel_path,
        definition_name=definition,
        catalog_name=catalog_name,
        target_hardware=target_hardware,
        server_url=server_url,
        entry_point=entry_point,
        destination_passing_style=destination_passing_style,
        language=language,
        warmup_ms=warmup_ms,
        benchmark_ms=benchmark_ms,
        num_trials=num_trials,
        timeout_seconds=timeout_seconds,
    )
