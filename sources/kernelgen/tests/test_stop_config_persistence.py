"""Stop-policy persistence must not silently expand an explicit round budget."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from kernelgen.data import _atomic
from kernelgen.data.ledger import Ledger, STOP_CONFIG_FILENAME
from kernelgen.data.stop_policy import StopConfig
from kernelgen.tests.helpers import experiment_plan, round_conclusion


def test_absent_stop_config_preserves_default_without_creating_file(tmp_path):
    assert Ledger(tmp_path).stop_config == StopConfig()
    assert not (tmp_path / STOP_CONFIG_FILENAME).exists()


def test_stop_config_round_trip_keeps_all_fields_and_explicit_budget(tmp_path):
    config = StopConfig(early_stop_rounds=0, min_rounds=3, max_round=1, soft_stop_disabled=True)
    ledger = Ledger(tmp_path)
    ledger.set_stop_config(config)
    assert json.loads((tmp_path / STOP_CONFIG_FILENAME).read_text()) == asdict(config)
    reloaded = Ledger(tmp_path)
    assert reloaded.stop_config == config
    reloaded.record_eval({"status": "PASSED", "geo_mean": 1.25}, "code", experiment_plan(1))
    assert reloaded.finalize_round(1, round_conclusion(1)).code == "max_round_reached"


@pytest.mark.parametrize("content", ["", '{"max_round":', "null", "[]", '{"unexpected": 1}', '{"max_round": 0}', '{"max_round": "one"}', b"\xff"])
def test_invalid_present_config_fails_without_replacing_it(tmp_path, content):
    path = tmp_path / STOP_CONFIG_FILENAME
    raw = content if isinstance(content, bytes) else content.encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="invalid stop config") as error:
        Ledger(tmp_path)
    assert str(path) in str(error.value)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_stop_config_update_preserves_committed_state(tmp_path, monkeypatch, existing):
    ledger = Ledger(tmp_path)
    if existing:
        ledger.set_stop_config(StopConfig(max_round=1))
    old_config = ledger.stop_config
    destination = tmp_path / STOP_CONFIG_FILENAME
    old_content = destination.read_bytes() if existing else None

    def fail_replace(source, target):
        assert target == destination
        assert json.loads(Path(source).read_text())["max_round"] == 2
        raise OSError("interrupted atomic replacement")

    monkeypatch.setattr(_atomic.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        ledger.set_stop_config(StopConfig(max_round=2))
    assert ledger.stop_config == old_config
    assert Ledger(tmp_path).stop_config == old_config
    assert (destination.read_bytes() if destination.exists() else None) == old_content
    assert not list(tmp_path.glob("*.tmp"))
