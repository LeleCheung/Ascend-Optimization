"""Unit tests for the eval_only tool + its MCP registration.

Pure Python, no server/GPU needed — exercises the guard paths and registration:

    cd /data/akg_kernel_bench_lite
    PYTHONPATH=.:./flashinfer-bench python3 kernelgen/tests/test_eval_only.py
"""

import asyncio
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "flashinfer-bench"),
)

from kernelgen.tools.eval_only import evaluate_only
from kernelgen.mcp_server.contract import MCP_TOOL_NAMES, MCP_CLAUDE_TOOL_NAMES


def test_registered_in_contract():
    assert "eval_only" in MCP_TOOL_NAMES
    assert "mcp__kernelgen__eval_only" in MCP_CLAUDE_TOOL_NAMES


def test_registered_in_live_server():
    import kernelgen.mcp_server.server as s
    names = {t.name for t in asyncio.run(s.mcp.list_tools())}
    assert "eval_only" in names, f"eval_only missing from {sorted(names)}"
    parameters = inspect.signature(s.eval_only).parameters
    assert "catalog_name" in parameters
    assert "trace_root" not in parameters
    assert "trace_set_key" not in parameters


def test_guard_missing_catalog(tmp_path):
    kernel = tmp_path / "main.py"
    kernel.write_text("def run(x): return x", encoding="utf-8")
    r = evaluate_only(
        kernel_path=str(kernel),
        definition="x",
        catalog_name="not-installed-catalog",
        server_url="http://x",
    )
    assert r["status"] == "ERROR"
    assert "catalog" in r["log"].lower()


def test_guard_missing_server():
    r = evaluate_only(
        kernel_path="/nope.py",
        definition="x",
        catalog_name="flaggems-v5",
        server_url="",
    )
    assert r["status"] == "ERROR"
    assert "server_url" in r["log"]


def test_guard_missing_kernel_file(tmp_path=None):
    r = evaluate_only(kernel_path="/definitely/not/here.py", definition="x",
                      catalog_name="flaggems-v5", server_url="http://x")
    assert r["status"] == "ERROR"
    assert "kernel file not found" in r["log"]


def test_adapter_delegates_to_shared_evaluator(monkeypatch):
    from kernelgen.tools import kernelgen_server_adapter

    calls = []
    monkeypatch.setattr(
        kernelgen_server_adapter,
        "evaluate_kernel_file",
        lambda **kwargs: calls.append(kwargs) or {"status": "PASSED"},
    )

    result = evaluate_only(
        kernel_path="candidate.py",
        definition="op",
        catalog_name="flaggems-v5",
        server_url="http://eval",
    )

    assert result["status"] == "PASSED"
    assert len(calls) == 1
    assert calls[0]["definition_name"] == "op"
    assert calls[0]["catalog_name"] == "flaggems-v5"
    assert calls[0]["language"] is None
    assert calls[0]["destination_passing_style"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
