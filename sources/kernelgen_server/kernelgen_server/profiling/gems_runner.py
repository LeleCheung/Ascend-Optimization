"""Run FlagGems pytest while keeping KGS capture in the pytest process."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import pytest

from kernelgen_server.runtime.flaggems import flaggems_vendor

from .capture import capture_scope, prepare_profile_compiler


class ProfilePlugin:
    def __init__(self, backend: str, case_id: str):
        self.backend = backend
        self.case_id = case_id
        self.completed = 0

    @pytest.hookimpl
    @contextmanager
    def pytest_flaggems_profile_scope(self, backend: str, case_id: str):
        expected_backend = flaggems_vendor(self.backend)
        if backend != expected_backend or case_id != self.case_id or self.completed:
            raise RuntimeError(
                "profile capture request does not match the selected case"
            )
        with capture_scope(self.backend):
            yield
        self.completed += 1


def run(backend: str, case_id: str, marker: Path, pytest_args: list[str]) -> int:
    marker.unlink(missing_ok=True)
    plugin = ProfilePlugin(backend, case_id)
    prepare_profile_compiler(backend)
    status = int(pytest.main(pytest_args, plugins=[plugin]))
    if status == 0:
        if plugin.completed != 1:
            raise RuntimeError(
                "profile-only pytest did not complete exactly one capture"
            )
        marker.write_text("completed\n", encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--completion-marker", type=Path, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pytest_args = (
        args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    )
    raise SystemExit(
        run(args.backend, args.case_id, args.completion_marker, pytest_args)
    )


if __name__ == "__main__":
    main()
