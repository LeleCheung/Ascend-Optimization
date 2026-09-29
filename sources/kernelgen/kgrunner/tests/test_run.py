"""Tests for kgrunner.run() API."""

import json
import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from kgrunner import run
from kgrunner.agent import AgentDef, IOField, ResourceRequirements
from kgrunner.codeagent import CodeAgentConfig


@pytest.fixture
def tmp_workspace(tmp_path):
    return tmp_path / "workspace"


@pytest.fixture
def log_dir(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.fixture
def mock_cc_success(tmp_path):
    """Create a mock CC binary that outputs a successful JSONL result."""
    script = tmp_path / "mock_cc"
    script.write_text('''#!/bin/bash
# Write JSONL to stdout (which is redirected to a file by spawn_agent)
echo '{"type":"system","subtype":"init","session_id":"test-session-001"}'
echo '{"type":"result","result":"```json\\n{\\"status\\":\\"success\\",\\"operator\\":\\"relu\\",\\"kernel_path\\":\\"triton_relu.py\\",\\"test_passed\\":true}\\n```"}'
exit 0
''')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def mock_cc_fail(tmp_path):
    """Create a mock CC binary that outputs a failed result."""
    script = tmp_path / "mock_cc_fail"
    script.write_text('''#!/bin/bash
echo '{"type":"system","subtype":"init","session_id":"test-session-002"}'
echo '{"type":"result","result":"```json\\n{\\"status\\":\\"failed\\",\\"error\\":\\"compilation error\\"}\\n```"}'
exit 0
''')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def mock_cc_api_error(tmp_path):
    """Create a mock CC binary that simulates API stream error."""
    script = tmp_path / "mock_cc_api_error"
    script.write_text('''#!/bin/bash
echo '{"type":"system","subtype":"init","session_id":"test-session-003"}'
echo '{"type":"result","result":"API Error: Unexpected EOF"}'
exit 1
''')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def mock_cc_timeout(tmp_path):
    """Create a mock CC binary that hangs (for timeout testing)."""
    script = tmp_path / "mock_cc_timeout"
    script.write_text('''#!/bin/bash
echo '{"type":"system","subtype":"init","session_id":"test-session-004"}'
sleep 60
''')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def prompt_template(tmp_path):
    """Create a simple prompt template."""
    tpl = tmp_path / "prompt.md"
    tpl.write_text("Generate kernel for {{operator}}")
    return tpl


@pytest.fixture
def sample_agent(prompt_template):
    return AgentDef(
        name="test_agent",
        version="1.0.0",
        description="test",
        prompt_template=prompt_template,
        resources=ResourceRequirements(gpu=0, workspace=False),
        inputs=[IOField(name="operator", type="string", required=True)],
        outputs=[
            IOField(name="status", type="string", required=True),
            IOField(name="operator", type="string", required=True),
            IOField(name="kernel_path", type="path", optional=True),
            IOField(name="test_passed", type="bool", optional=True),
        ],
    )


class TestRunCallable:
    def test_callable_receives_input(self):
        def my_fn(input, gpu_id=None, workspace=None):
            return {"status": "success", "value": input["x"] * 2}

        result = run(my_fn, {"x": 5})
        assert result == {"status": "success", "value": 10}

    def test_callable_receives_gpu_and_workspace(self, tmp_path):
        def my_fn(input, gpu_id=None, workspace=None):
            return {"gpu": gpu_id, "ws": str(workspace)}

        result = run(my_fn, {}, gpu=3, workspace=tmp_path)
        assert result["gpu"] == 3
        assert result["ws"] == str(tmp_path)

    def test_callable_exception_propagates(self):
        def bad_fn(input, gpu_id=None, workspace=None):
            raise ValueError("broken")

        with pytest.raises(ValueError, match="broken"):
            run(bad_fn, {})


class TestRunAgent:
    def test_success(self, sample_agent, mock_cc_success, log_dir, tmp_path):
        config = CodeAgentConfig(bin=mock_cc_success)
        result = run(
            sample_agent,
            {"operator": "relu"},
            agent_config=config,
            workspace=tmp_path,
            log_dir=log_dir,
        )
        assert result["status"] == "success"
        assert result["operator"] == "relu"
        assert result["kernel_path"] == "triton_relu.py"
        assert result["test_passed"] is True

    def test_validates_missing_required_input(self, sample_agent, mock_cc_success, log_dir, tmp_path):
        config = CodeAgentConfig(bin=mock_cc_success)
        with pytest.raises(ValueError, match="missing required input"):
            run(
                sample_agent,
                {},  # missing 'operator'
                agent_config=config,
                workspace=tmp_path,
                log_dir=log_dir,
            )

    def test_retry_on_failure(self, sample_agent, mock_cc_fail, mock_cc_success, log_dir, tmp_path):
        call_count = {"n": 0}
        original_bin = None

        # First call fails, second succeeds
        fail_script = mock_cc_fail
        success_script = mock_cc_success

        # Use a wrapper script that switches behavior
        wrapper = tmp_path / "wrapper_cc"
        wrapper.write_text(f'''#!/bin/bash
COUNT_FILE="{tmp_path}/call_count"
if [ ! -f "$COUNT_FILE" ]; then
    echo "0" > "$COUNT_FILE"
fi
COUNT=$(cat "$COUNT_FILE")
COUNT=$((COUNT + 1))
echo "$COUNT" > "$COUNT_FILE"
if [ "$COUNT" -eq 1 ]; then
    exec {fail_script} "$@"
else
    exec {success_script} "$@"
fi
''')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)

        config = CodeAgentConfig(bin=str(wrapper))
        result = run(
            sample_agent,
            {"operator": "relu"},
            agent_config=config,
            workspace=tmp_path,
            log_dir=log_dir,
            max_retries=2,
        )
        assert result["status"] == "success"

    def test_exhausted_retries_raises(self, sample_agent, mock_cc_fail, log_dir, tmp_path):
        config = CodeAgentConfig(bin=mock_cc_fail)
        with pytest.raises(RuntimeError, match="exhausted"):
            run(
                sample_agent,
                {"operator": "relu"},
                agent_config=config,
                workspace=tmp_path,
                log_dir=log_dir,
                max_retries=1,
            )

    def test_timeout_kills_process(self, sample_agent, mock_cc_timeout, log_dir, tmp_path):
        config = CodeAgentConfig(bin=mock_cc_timeout)
        with pytest.raises(RuntimeError, match="exhausted"):
            run(
                sample_agent,
                {"operator": "relu"},
                agent_config=config,
                workspace=tmp_path,
                log_dir=log_dir,
                timeout=1,
                max_retries=0,
            )

    def test_resume_on_api_stream_error(self, sample_agent, log_dir, tmp_path):
        """First call hits API stream error, resume succeeds."""
        # Mock CC that checks --resume flag to decide behavior
        script = tmp_path / "mock_cc_resume"
        script.write_text('''#!/bin/bash
# If --resume is in args, output success; otherwise output API error
RESUME=false
for arg in "$@"; do
    if [ "$arg" = "--resume" ]; then
        RESUME=true
        break
    fi
done

if [ "$RESUME" = "true" ]; then
    echo '{"type":"system","subtype":"init","session_id":"test-session-resumed"}'
    echo '{"type":"result","result":"```json\\n{\\"status\\":\\"success\\",\\"operator\\":\\"relu\\",\\"kernel_path\\":\\"triton_relu.py\\",\\"test_passed\\":true}\\n```"}'
    exit 0
else
    echo '{"type":"system","subtype":"init","session_id":"test-session-003"}'
    echo '{"type":"result","result":"API Error: Unexpected EOF"}'
    exit 1
fi
''')
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        config = CodeAgentConfig(bin=str(script))
        result = run(
            sample_agent,
            {"operator": "relu"},
            agent_config=config,
            workspace=tmp_path,
            log_dir=log_dir,
            max_resumes=3,
        )
        assert result["status"] == "success"
        assert result["operator"] == "relu"

    def test_resume_exhausted_then_retry(self, sample_agent, log_dir, tmp_path):
        """API errors exhaust max_resumes, falls through to retry, still fails."""
        script = tmp_path / "mock_cc_always_api_error"
        script.write_text('''#!/bin/bash
echo '{"type":"system","subtype":"init","session_id":"test-session-always-error"}'
echo '{"type":"result","result":"API Error: Unexpected EOF"}'
exit 1
''')
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        config = CodeAgentConfig(bin=str(script))
        with pytest.raises(RuntimeError, match="exhausted"):
            run(
                sample_agent,
                {"operator": "relu"},
                agent_config=config,
                workspace=tmp_path,
                log_dir=log_dir,
                max_resumes=2,
                max_retries=1,
            )

    def test_custom_output_parser(self, sample_agent, log_dir, tmp_path):
        """User-provided output_parser overrides default JSON extraction."""
        # Mock CC that outputs YAML-like text (not JSON code block)
        script = tmp_path / "mock_cc_yaml"
        script.write_text('''#!/bin/bash
echo '{"type":"system","subtype":"init","session_id":"test-session-yaml"}'
echo '{"type":"result","result":"status: success\\noperator: relu\\nkernel_path: triton_relu.py\\ntest_passed: true"}'
exit 0
''')
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        def yaml_parser(text: str) -> dict | None:
            result = {}
            for line in text.strip().split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    val = val.strip()
                    if val == "true":
                        val = True
                    elif val == "false":
                        val = False
                    result[key.strip()] = val
            return result if result else None

        config = CodeAgentConfig(bin=str(script))
        result = run(
            sample_agent,
            {"operator": "relu"},
            agent_config=config,
            workspace=tmp_path,
            log_dir=log_dir,
            output_parser=yaml_parser,
        )
        assert result["status"] == "success"
        assert result["operator"] == "relu"
