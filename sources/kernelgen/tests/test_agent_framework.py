"""Unit tests for the agent framework (ADR-3 task #5).

Pure Python, no torch/GPU/LLM (BaseAgent.run is exercised via FakeRuntime):

    cd /data/akg_kernel_bench_lite
    python -m pytest tests/test_agent_framework.py -v
    # or:
    python tests/test_agent_framework.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional


from pydantic import BaseModel, Field  # noqa: E402

from kernelgen.framework import (  # noqa: E402
    BaseAgent,
    AgentContractError,
    FakeRuntime,
    render_contract,
    extract_json,
)


# --- contract: render_contract -------------------------------------------

class _Out(BaseModel):
    status: str = Field(description="PASSED / FAILED")
    score: float = Field(description="the speedup")
    note: Optional[str] = Field(default=None, description="optional note")


def test_render_contract_lists_fields_and_required():
    block = render_contract(_Out)
    assert "status:" in block and "PASSED / FAILED" in block
    assert "score:" in block and "the speedup" in block
    assert "[required]" in block            # status/score required
    assert "[optional]" in block            # note optional


class _AliasedOut(BaseModel):
    internal_name: int = Field(alias="public_name")


def test_render_contract_uses_json_field_alias():
    block = render_contract(_AliasedOut)
    assert "public_name: int" in block
    assert "internal_name:" not in block


class _NestedLeaf(BaseModel):
    value: int


class _NestedOut(BaseModel):
    items: List[_NestedLeaf]
    mapping: Dict[str, _NestedLeaf]
    optional: _NestedLeaf | None = None


def test_render_contract_recurses_through_containers_and_pep604_union():
    block = render_contract(_NestedOut)
    assert block.count("# _NestedLeaf fields:") == 3
    assert block.count("value:") == 3


class _ConstrainedOut(BaseModel):
    kind: Literal["reference", "method"]
    phases: List[Literal["initial", "post_error"]]
    identifier: str = Field(
        min_length=4,
        max_length=80,
        pattern=r"^kg:(reference|method):",
    )
    rank: int = Field(gt=0, le=10)


def test_render_contract_preserves_literal_values_and_field_constraints():
    block = render_contract(_ConstrainedOut)
    assert "kind: Literal['reference', 'method']" in block
    assert "phases: List[Literal['initial', 'post_error']]" in block
    assert "min_length=4" in block
    assert "max_length=80" in block
    assert "pattern='^kg:(reference|method):'" in block
    assert "gt=0" in block
    assert "le=10" in block


# --- contract: extract_json ----------------------------------------------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_and_think():
    raw = '<think>reasoning {bogus}</think> preamble\n```json\n{"a": 2, "b": "x"}\n```\ntail'
    assert extract_json(raw) == {"a": 2, "b": "x"}


def test_extract_json_fenced_with_nested_code_fence_in_string():
    payload = {
        "candidate_experience": "concise",
        "candidate_detailed": "before\n```python\nx = 1\n```\nafter",
        "skip_reason": "",
    }
    raw = f"```json\n{json.dumps(payload)}\n```"
    assert extract_json(raw) == payload


def test_extract_json_last_json_fence_wins():
    raw = '```json\n{"a": 1}\n```\n```json\n{"a": 2}\n```'
    assert extract_json(raw) == {"a": 2}


def test_extract_json_tries_earlier_fence_when_latest_is_invalid():
    raw = '```json\n{"a": 1}\n```\n```json\n{"a": invalid}\n```'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_braces_fallback():
    assert extract_json('noise {"a": 3} more') == {"a": 3}

def test_extract_json_repairs_trailing_commas_outside_strings():
    raw = '```json\n{"items":[{"text":"keep ,} and ,] inside strings","value":1,},],}\n```'
    assert extract_json(raw) == {
        "items": [{"text": "keep ,} and ,] inside strings", "value": 1}]}


def test_extract_json_invalid_raises():
    for bad in ["", "no json here", "[1,2,3]"]:
        try:
            extract_json(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


# --- BaseAgent.run via FakeRuntime ---------------------------------------

class _In(BaseModel):
    x: int


class _Agent(BaseAgent):
    name = "demo"
    md_path = ""                       # no role file for the test
    InputModel = _In
    OutputModel = _Out


def test_run_happy_path():
    rt = FakeRuntime(['```json\n{"status":"PASSED","score":1.2}\n```'])

    out = _Agent().run({"x": 1}, rt)
    assert isinstance(out, _Out)
    assert out.status == "PASSED" and out.score == 1.2
    # Input and output contract are dynamic prompt content. Provider tool
    # registration belongs to the native agent definition, not BaseAgent.
    assert "OUTPUT CONTRACT" in rt.calls[0]["prompt"]
    assert "Available tools" not in rt.calls[0]["prompt"]
    assert "custom_tools" not in rt.calls[0]


def test_run_repairs_then_succeeds():
    # first reply misses required 'score' -> ValidationError -> repair -> 2nd ok
    rt = FakeRuntime([
        '{"status":"PASSED"}',                          # invalid: missing score
        '```json\n{"status":"PASSED","score":1.5}\n```' # valid
    ])
    # max_retries is now internal (_MAX_RETRIES=2)
    out = _Agent().run({"x": 1}, rt)
    assert out.score == 1.5
    assert len(rt.calls) == 2
    # the repair note was appended to the 2nd prompt
    assert "FAILED VALIDATION" in rt.calls[1]["prompt"]
    assert '{"status":"PASSED"}' in rt.calls[1]["prompt"]
    assert "trailing commas" in rt.calls[1]["prompt"]



def test_run_allows_two_repairs():
    rt = FakeRuntime([
        "not json",
        '{"status":"PASSED"}',
        '{"status":"PASSED","score":2.0}',
    ])

    out = _Agent().run({"x": 1}, rt)

    assert out.score == 2.0
    assert len(rt.calls) == 3
    assert '{"status":"PASSED"}' in rt.calls[2]["prompt"]
def test_run_repairs_output_in_same_provider_session():
    class _ResumableRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def invoke(self, prompt, *, model="inherit", agent=None):
            self.calls.append(("invoke", prompt))
            self.last_session_id = "session-1"
            return '{"status":"PASSED"}'

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append(("resume", prompt))
            return '{"status":"PASSED","score":1.5}'

    rt = _ResumableRuntime()
    out = _Agent().run({"x": 1}, rt)

    assert out.score == 1.5
    assert [kind for kind, _ in rt.calls] == ["invoke", "resume"]
    assert "PREVIOUS OUTPUT FAILED VALIDATION" in rt.calls[1][1]
    assert "AUTHORITATIVE OUTPUT CONTRACT" in rt.calls[1][1]
    assert "score: float" in rt.calls[1][1]


def test_run_exhausts_repairs_raises():
    rt = FakeRuntime(["not json"] * 3)
    try:
        _Agent().run({"x": 1}, rt)
    except AgentContractError:
        assert len(rt.calls) == 3
    else:
        raise AssertionError("should raise AgentContractError after repairs")


def test_run_bad_input_raises_before_invoke():
    rt = FakeRuntime([])               # no replies; must not be called

    try:
        _Agent().run({"x": "not-an-int-and-not-coercible"}, rt)
    except Exception:
        assert rt.calls == []          # failed at input validation, before invoke
    else:
        raise AssertionError("bad input should raise")


def test_run_loads_role_md(tmp_path):
    md = tmp_path / "role.md"
    md.write_text("YOU ARE A DEMO AGENT.")

    class _RoleAgent(_Agent):
        md_path = str(md)

    rt = FakeRuntime(['{"status":"PASSED","score":1.0}'])

    _RoleAgent().run({"x": 1}, rt)
    assert "YOU ARE A DEMO AGENT." in rt.calls[0]["prompt"]


def test_native_runtime_rejects_missing_definition():
    class _NativeRuntime(FakeRuntime):
        supports_native_agents = True

    try:
        _Agent().run({"x": 1}, _NativeRuntime([]))
    except RuntimeError as exc:
        assert "agent role definition is missing" in str(exc)
    else:
        raise AssertionError("missing native agent should not silently lose its role")


def test_repository_agents_do_not_keep_duplicate_role_md():
    repo_root = Path(__file__).resolve().parents[1]
    legacy_roles = list((repo_root / "agents").rglob("role.md"))
    assert legacy_roles == []


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            if "tmp_path" in inspect.signature(t).parameters:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
