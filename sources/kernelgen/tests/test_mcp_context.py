"""Tests for workspace-bound configuration used by the KernelGen MCP server."""

import sys
import tempfile
import json
from pathlib import Path

import pytest

from kernelgen.data.tool_context import (
    ToolContext,
    load_tool_context,
    resolve_workspace_file,
)


def test_context_round_trip(tmp_path):
    written = ToolContext(
        definition="gelu",
        target_hardware="H100",
        eval_server_url="http://eval:8000",
        catalog_name="flaggems-v5",
        destination_passing_style=False,
        eval_tolerance_mode="fixed",
        eval_atol=0.123,
        eval_rtol=0.456,
    ).write(tmp_path)

    assert written == tmp_path / ".kernelgen" / "tool-context.json"
    loaded = load_tool_context(tmp_path)
    assert loaded.definition == "gelu"
    assert loaded.implementation_language.value == "triton"
    assert loaded.eval_server_url == "http://eval:8000"
    assert loaded.catalog_name == "flaggems-v5"
    assert loaded.destination_passing_style is False
    assert loaded.eval_tolerance_mode == "fixed"
    assert loaded.eval_atol == 0.123
    assert loaded.eval_rtol == 0.456
    assert loaded.eval_timeout_seconds == 1500
    assert loaded.eval_transport_timeout_seconds == 1800
    serialized = json.loads(written.read_text(encoding="utf-8"))
    assert "trace_root" not in serialized
    assert "trace_set_key" not in serialized


@pytest.mark.parametrize("legacy_field", ["trace_root", "trace_set_key"])
def test_context_rejects_removed_trace_fields(legacy_field):
    with pytest.raises(ValueError, match=legacy_field):
        ToolContext(
            definition="gelu",
            target_hardware="H100",
            **{legacy_field: "legacy"},
        )


def test_context_rejects_language_without_an_end_to_end_profile():
    try:
        ToolContext(
            definition="gelu",
            target_hardware="H100",
            implementation_language="tilelang",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported implementation language was accepted")


def test_workspace_file_rejects_escape(tmp_path):
    (tmp_path / "tmp").mkdir()
    kernel = tmp_path / "tmp" / "main.py"
    kernel.write_text("def run(): pass")
    assert resolve_workspace_file(tmp_path, "tmp/main.py") == kernel.resolve()

    for unsafe in ("../outside.py", str(kernel.resolve())):
        try:
            resolve_workspace_file(tmp_path, unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe path accepted: {unsafe}")


def test_missing_context_has_actionable_error(tmp_path):
    try:
        load_tool_context(tmp_path)
    except RuntimeError as exc:
        assert "tool-context.json" in str(exc)
    else:
        raise AssertionError("missing context should fail")


if __name__ == "__main__":
    import inspect
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            if "tmp_path" in inspect.signature(test).parameters:
                with tempfile.TemporaryDirectory() as directory:
                    test(Path(directory))
            else:
                test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
