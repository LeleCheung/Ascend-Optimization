"""Bounded source-only export for KG review; never imports or runs test code."""

from pathlib import Path

from kernelgen_server.protocol.schema import SourceFile, TestReviewSources

MAX_REVIEW_SOURCE_BYTES = 8 * 1024 * 1024
MAX_REVIEW_SOURCE_FILES = 128


def _read_sources(root, paths):
    root = Path(root).resolve()
    files = []
    total = 0
    for path in sorted(set(paths)):
        path = Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("test review source must be a regular file inside its source root")
        if len(files) >= MAX_REVIEW_SOURCE_FILES:
            raise ValueError("test review source file limit exceeded")
        with path.open("rb") as stream:
            raw = stream.read(MAX_REVIEW_SOURCE_BYTES - total + 1)
        total += len(raw)
        if total > MAX_REVIEW_SOURCE_BYTES:
            raise ValueError("test review source byte limit exceeded")
        files.append(SourceFile(path=path.relative_to(root).as_posix(), content=raw.decode("utf-8")))
    return files


def export_test_sources(catalog, operator):
    if operator.evaluator == "flaggems":
        from .adapters.flaggems.adapter import FlagGemsEvaluationAdapter
        # The same source/revision/suite checks used during inspect, without
        # executing pytest --list-cases or importing the target framework.
        adapter = FlagGemsEvaluationAdapter(catalog=catalog, operator=operator)
        assets = adapter._assets()
        paths = [*assets.correctness, *assets.performance]
        for name in ("conftest.py", "tests/conftest.py", "tests/accuracy_utils.py",
                     "tests/test_utils.py", "tests/utils.py", "benchmark/base.py",
                     "benchmark/consts.py", "benchmark/cases.py", "benchmark/utils.py",
                     "benchmark/core_shapes.yaml", "benchmark/profile_hook.py",
                     "benchmark/conftest.py", "benchmark/performance_utils.py",
                     "benchmark/generated_operator_utils.py", "benchmark/reference.py"):
            path = assets.root / name
            if path.exists():
                paths.append(path)
        paths.extend((assets.root / "src/flag_gems/testing").rglob("*.py"))
        evidence = TestReviewSources(framework_revision=assets.revision,
                                     files=_read_sources(assets.root, paths))
        if adapter._assets().revision != assets.revision:
            raise ValueError("Gems checkout changed while exporting review evidence")
        return evidence

    # The contract already includes oracle and workloads. Export companion
    # Python helpers, when present, without exposing data archives or executables.
    oracle = operator.oracle_path
    if oracle is None:
        return TestReviewSources(files=[SourceFile(path="reference.py", content=operator.definition.reference or "")])
    return TestReviewSources(files=_read_sources(oracle.parent, oracle.parent.rglob("*.py")))
