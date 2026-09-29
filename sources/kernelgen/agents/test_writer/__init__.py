"""TestWriterAgent: generate new-spec FlagGems correctness + benchmark tests.

Given a ``torch.ops.aten`` operator, this agent:
  1. Enumerates overloads (torch.ops.aten + native_functions.yaml cross-check)
  2. Collects reference shapes / similar-operator examples from FlagGems
  3. Asks the LLM to produce the two test files (new-spec style:
     ``resolve_gems_op`` / ``gems_op=`` / two-phase GenericBenchmark)
  4. Verifies them ref-vs-ref in-process: ``override_gems_op(op, torch.ops.aten.<op>)``
     then ``pytest.main`` — candidate is the torch native op, so a pass proves
     the test files themselves are correct
  5. On failure, feeds pytest output back to the LLM (append_repair) and retries
     up to ``max_verify_retries``
  6. Writes exactly two files into the FlagGems checkout:
     ``tests/test_<op>.py`` + ``benchmark/test_<op>.py`` (existing → skipped)

Git operations (commit/push/PR) are intentionally outside this agent.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from kernelgen.framework.base import AgentContractError, BaseAgent
from kernelgen.framework.contract import append_repair, extract_json, render_contract

from .overloads import format_overloads, operator_exists, query_native_entries
from .references import build_reference_context


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------

class TestWriterInput(BaseModel):
    operator: str                          # torch.ops.aten operator, e.g. "clamp_max"
    flaggems_dir: str = ""                 # FlagGems checkout root (where tests/ lives)
    max_verify_retries: int = Field(default=5, ge=1, le=20)


class TestWriterOutput(BaseModel):
    operator: str
    correctness_test: str = ""             # full source of tests/test_<op>.py
    benchmark_test: str = ""               # full source of benchmark/test_<op>.py
    files_written: List[str] = Field(default_factory=list)
    status: str = Field(default="", description="WRITTEN / SKIPPED_EXISTS / FAILED / NO_VALID_OUTPUT")
    summary: str = ""


# ---------------------------------------------------------------------------
# Verification (ref-vs-ref via override + subprocess pytest)
# ---------------------------------------------------------------------------
#
# Verification runs in a FRESH subprocess (never pytest.main in the KernelGen
# process). Reasons:
#   * isolation: the subprocess sys.path contains only the FlagGems checkout and
#     flag_gems -- NOT the KernelGen repo root, so its ``tests`` package cannot
#     shadow FlagGems' ``tests``/``benchmark`` packages. This removes the need
#     for any sys.path hack inside the generated test files.
#   * state: repeated in-process pytest.main leaked GPU memory and cached stale
#     conftest/Config singletons across runs (the earlier 38/72 false
#     failures). A fresh process per operator has no such accumulation.
#   * override_gems_op is process-local, so the candidate override is installed
#     inside the subprocess before pytest.main runs there.

_VERIFY_DRIVER = r"""import sys
import torch
import flag_gems
import flag_gems.testing as testing
import pytest
from contextlib import ExitStack

operator = sys.argv[1]
run_benchmark = sys.argv[2] == "1"
correctness = sys.argv[3]
benchmark = sys.argv[4]

packet = getattr(torch.ops.aten, operator, None)
overrides = [(operator, packet)] if packet is not None else []
if packet is not None:
    for name in packet.overloads():
        if name == "out":
            out_fn = getattr(packet, "out")
            overrides.append((operator + ".out", out_fn))
            overrides.append((operator + "_out", out_fn))

with ExitStack() as stack:
    for name, fn in overrides:
        stack.enter_context(testing.override_gems_op(name, fn))
    code = pytest.main(
        ["tests/" + correctness, "-q", "-x", "--no-header",
         "-p", "no:cacheprovider", "--import-mode=importlib"]
    )
    if code != 0:
        sys.exit(code)
    if run_benchmark:
        code = pytest.main(
            ["benchmark/" + benchmark, "-q", "-x", "--no-header",
             "-p", "no:cacheprovider", "--import-mode=importlib",
             "--warmup", "0", "--iter", "1"]
        )
        sys.exit(code)
sys.exit(0)
"""


def _verify_files(
    flaggems_dir: Path,
    operator: str,
    correctness_src: str,
    benchmark_src: str,
    *,
    run_benchmark: bool = True,
    timeout_s: int = 900,
) -> tuple[bool, str]:
    """Verify generated tests ref-vs-ref in a fresh subprocess.

    The candidate is overridden to ``torch.ops.aten.<op>`` (torch native) inside
    the subprocess, so a pytest pass proves the test files themselves are
    correct -- independent of whether flag_gems implements the op.

    Returns (passed, combined_log).
    """
    # 1. Stage the two files in a temp checkout so we never touch FlagGems.
    with tempfile.TemporaryDirectory(prefix="tw_verify_") as tmp:
        tmp_root = Path(tmp)
        (tmp_root / "tests").mkdir()
        (tmp_root / "benchmark").mkdir()
        import shutil

        def _copytree_contents(src: Path, dst: Path) -> None:
            for item in src.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dst / item.name)

        _copytree_contents(flaggems_dir / "tests", tmp_root / "tests")
        _copytree_contents(flaggems_dir / "benchmark", tmp_root / "benchmark")

        correctness_name = f"test_{operator}.py"
        benchmark_name = f"test_{operator}.py"
        (tmp_root / "tests" / correctness_name).write_text(correctness_src, encoding="utf-8")
        (tmp_root / "benchmark" / benchmark_name).write_text(benchmark_src, encoding="utf-8")

        # 2. Run the driver in a fresh subprocess. PYTHONPATH carries ONLY the
        #    FlagGems src (for flag_gems) and the staged temp root (for the
        #    tests/benchmark packages) -- never the KernelGen repo root.
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(flaggems_dir / "src"), str(tmp_root)]
        )
        proc = subprocess.run(
            [sys.executable, "-c", _VERIFY_DRIVER, operator,
             "1" if run_benchmark else "0", correctness_name, benchmark_name],
            capture_output=True, text=True,
            cwd=str(tmp_root), env=env, timeout=timeout_s,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, combined


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestWriterAgent(BaseAgent):
    """Generate new-spec FlagGems tests for a torch.ops.aten operator.

    Neutral role: ``.kernelgen/agents/kernel-test-writer.md``.
    """

    name = "test_writer"
    InputModel = TestWriterInput
    OutputModel = TestWriterOutput

    def preprocess(self, inp: TestWriterInput, runtime) -> str:
        role = self._role_for(runtime)

        task_block = (
            "<task>\n"
            f"Operator: {inp.operator}\n"
            "Target: write tests/test_<op>.py + benchmark/test_<op>.py "
            "following the new KernelGen integration spec AND the regular-operator "
            "test spec (value ranges / shape levels / broadcast / backward / "
            "negative cases, via tests/test_utils.py helpers).\n"
            "If the target test file already exists, REWRITE the whole file: keep "
            "spec-compliant tests, migrate randn-based value tests to the "
            "value-range framework, add missing dimensions.\n"
            "VERY IMPORTANT: the files must be written into the FlagGems checkout "
            "at the path given below, NOT into the agent workspace and NOT into "
            "any other FlagGems clone.\n"
            "</task>\n\n"
            f"<target_flaggems_dir>\n{inp.flaggems_dir}\n</target_flaggems_dir>"
        )

        overloads = format_overloads(inp.operator)
        native = query_native_entries(inp.operator)
        native_block = (
            f"<native_functions>\n{native}\n</native_functions>" if native else ""
        )

        flaggems_dir = Path(inp.flaggems_dir)
        references = ""
        if flaggems_dir.exists():
            references = build_reference_context(flaggems_dir, inp.operator)
            existing = self._read_existing_files(flaggems_dir, inp.operator)
            if existing:
                references += "\n\n" + existing

        contract = render_contract(self.OutputModel)
        return "\n\n".join(filter(None, [
            role,
            task_block,
            overloads,
            native_block,
            references,
            f"--- OUTPUT CONTRACT ---\n{contract}",
        ]))

    @staticmethod
    def _read_existing_files(flaggems_dir: Path, operator: str) -> str:
        """Inject the existing test file(s) so the agent can rewrite them."""
        blocks = []
        for sub, pattern in (("tests", "test_*.py"), ("benchmark", "test_*.py")):
            p = flaggems_dir / sub / f"test_{operator}.py"
            if p.exists():
                blocks.append(f"== EXISTING {sub}/{p.name} ==\n```python\n{p.read_text(encoding='utf-8')}\n```")
        return "\n\n".join(blocks)

    def postprocess(self, raw: str, runtime) -> TestWriterOutput:
        data = extract_json(raw)
        out = TestWriterOutput.model_validate(data)
        out.operator = data.get("operator") or out.operator
        return out

    # -- execution -------------------------------------------------------

    def _execute(self, inp: TestWriterInput) -> TestWriterOutput:
        from pydantic import ValidationError

        # 0. Pre-check: the operator must exist.
        if not operator_exists(inp.operator):
            return TestWriterOutput(
                operator=inp.operator,
                status="FAILED",
                summary=f"Operator '{inp.operator}' not found in torch.ops.aten or native_functions.yaml",
            )

        # flaggems_dir must be explicit (an InputModel field). Never fall back to
        # env/relative paths — the agent subprocess runs from an isolated
        # workspace and could otherwise resolve to a stale ~/FlagGems clone.
        flaggems_dir = Path(inp.flaggems_dir)
        if not flaggems_dir.exists() or not (flaggems_dir / "tests").exists():
            raise ValueError(
                f"flaggems_dir invalid or missing tests/: {flaggems_dir}"
            )

        target_test = flaggems_dir / "tests" / f"test_{inp.operator}.py"
        target_bench = flaggems_dir / "benchmark" / f"test_{inp.operator}.py"
        # REWRITE mode: existing files are injected into the prompt and the
        # agent outputs the full rewritten source. No SKIPPED_EXISTS short
        # circuit — the whole point is to bring existing tests up to spec.

        runtime = self._runtime
        prompt = self.preprocess(inp, runtime)
        last_err: Optional[Exception] = None
        last_output: Optional[TestWriterOutput] = None

        for attempt in range(1, inp.max_verify_retries + 1):
            try:
                raw = self._invoke_runtime(runtime, prompt)
                output = self.postprocess(raw, runtime)
            except (ValidationError, ValueError) as e:
                last_err = e
                prompt = append_repair(prompt, e)
                continue

            if not output.correctness_test or not output.benchmark_test:
                last_err = ValueError("Output missing correctness_test or benchmark_test source")
                prompt = append_repair(prompt, last_err)
                continue

            # Verify ref-vs-ref.
            print(
                f"  [verify] {inp.operator} attempt {attempt}/{inp.max_verify_retries}: "
                f"correctness={len(output.correctness_test)}c, "
                f"benchmark={len(output.benchmark_test)}c",
                flush=True,
            )
            passed, log = _verify_files(
                flaggems_dir, inp.operator,
                output.correctness_test, output.benchmark_test,
            )
            if passed:
                return self._write_files(inp, output, flaggems_dir)

            last_err = ValueError(f"verification failed (attempt {attempt}):\n{log[:2000]}")
            print(f"  [verify] {inp.operator} FAILED attempt {attempt}: {log[:300]}", flush=True)
            prompt = append_repair(prompt, last_err)
            last_output = output

        # Exhausted retries: return last output, not written.
        if last_output is not None:
            last_output.status = "FAILED"
            last_output.summary = f"verification failed after {inp.max_verify_retries} attempts"
            return last_output
        raise AgentContractError(
            f"{self.name}: no valid output after {inp.max_verify_retries} attempts: {last_err}"
        )

    # -- file writing ----------------------------------------------------

    def _write_files(
        self,
        inp: TestWriterInput,
        output: TestWriterOutput,
        flaggems_dir: Path,
    ) -> TestWriterOutput:
        """Write the two verified test files into the FlagGems checkout.

        REWRITE mode: existing files are overwritten with the agent's rewritten
        source (verified before writing).
        """
        test_path = flaggems_dir / "tests" / f"test_{inp.operator}.py"
        bench_path = flaggems_dir / "benchmark" / f"test_{inp.operator}.py"

        written: List[str] = []
        existed = test_path.exists() or bench_path.exists()

        test_path.write_text(output.correctness_test, encoding="utf-8")
        written.append(str(test_path))
        bench_path.write_text(output.benchmark_test, encoding="utf-8")
        written.append(str(bench_path))

        output.files_written = written
        output.status = "REWRITTEN" if existed else "WRITTEN"
        output.summary = (
            f"{'rewrote' if existed else 'wrote'} {len(written)} files: "
            f"tests/test_{inp.operator}.py, benchmark/test_{inp.operator}.py"
        )
        print(f"  [write] {inp.operator}: {'rewrote' if existed else 'wrote'} {len(written)} files", flush=True)
        return output