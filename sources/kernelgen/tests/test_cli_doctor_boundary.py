from argparse import Namespace

import pytest

from kernelgen.cli import server


@pytest.mark.parametrize("remote_error", [None, "checkout is dirty", "unexpected commit"])
def test_remote_doctor_does_not_require_local_checkout(monkeypatch, capsys, remote_error):
    monkeypatch.setattr(server, "_load_config", lambda name: {"target": "remote", "kgs_release": "v6.3.2", "kgs_commit": "a" * 40, "protocol_version": "v6.2"})
    monkeypatch.setattr(server, "_active_process", lambda name: None)
    monkeypatch.setattr(server, "_validate_checkout", lambda *a, **k: pytest.fail("local checkout must not be checked"))
    def remote(config, action):
        assert action == "doctor"
        if remote_error:
            raise RuntimeError(remote_error)
    monkeypatch.setattr(server, "_remote_call_config", remote)
    assert server._command_doctor(Namespace(name="remote")) == (1 if remote_error else 0)
    assert (remote_error or "status: OK") in capsys.readouterr().out


def test_local_doctor_checks_recorded_checkout(monkeypatch, tmp_path):
    config = {"target": "local", "kgs_root": str(tmp_path), "kgs_commit": "a" * 40}
    monkeypatch.setattr(server, "_load_config", lambda name: config)
    monkeypatch.setattr(server, "_active_process", lambda name: None)
    def validate(root, **kwargs):
        assert root == tmp_path and kwargs["commit"] == config["kgs_commit"]
        raise RuntimeError("dirty checkout")
    monkeypatch.setattr(server, "_validate_checkout", validate)
    assert server._command_doctor(Namespace(name="local")) == 1
