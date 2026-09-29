import sys

import pytest

from kernelgen_server.api import cli


def test_cli_rejects_removed_npu_event_timing(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["kernelgen-server", "--timing", "npu-event"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


def test_cli_rejects_nonpositive_operator_bundle_limit(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["kernelgen-server", "--operator-bundle-max-bytes", "0"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
