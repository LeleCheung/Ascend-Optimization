"""Host-only tests for KGS-owned target process management."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import os

import pytest

from kernelgen_server import management as remote_server


@pytest.mark.parametrize("backend,expected", [("enflame", {"TOPS_VISIBLE_DEVICES": "7"}),
                                              ("cuda", {"CUDA_VISIBLE_DEVICES": "7"})])
def test_remote_start_isolates_enflame_visibility(tmp_path, monkeypatch, backend, expected):
    for variable in ("TOPS_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"):
        monkeypatch.setenv(variable, "0,1,2")
    captured = {}
    class Process:
        pid = 321
    monkeypatch.setattr(remote_server, "_prepare", lambda payload: (tmp_path, {}))
    monkeypatch.setattr(remote_server.subprocess, "Popen", lambda command, **kwargs: captured.update(command=command, **kwargs) or Process())
    monkeypatch.setattr(remote_server, "_process_start", lambda pid: "start")
    remote_server._start({**_payload(tmp_path), "backend": backend, "devices": ["7"]})
    visible = {k: v for k, v in captured["env"].items() if k in {n for names in remote_server.DEVICE_ENVIRONMENTS.values() for n in names}}
    assert visible == expected
    assert captured["command"][captured["command"].index("--backend") + 1] == backend


def test_enflame_requires_and_protects_vendor_runtime(tmp_path, monkeypatch):
    assert "torch_gcu" in remote_server._runtime_modules("enflame")
    assert "torch-gcu" in remote_server.PROTECTED_DISTRIBUTIONS
    before = {name: None for name in remote_server.PROTECTED_DISTRIBUTIONS}
    before["torch-gcu"] = "original"
    versions = iter([before, {**before, "torch-gcu": "changed"}])
    monkeypatch.setattr(remote_server, "_package_versions", lambda: next(versions))
    monkeypatch.setattr(remote_server, "_validate_checkout", lambda *args, **kwargs: None)
    monkeypatch.setattr(remote_server, "_validate_installed_server", lambda *args: None)
    with pytest.raises(RuntimeError, match="protected runtime packages: torch-gcu"):
        remote_server._prepare({**_payload(tmp_path), "backend": "enflame"})


def _payload(tmp_path) -> dict:
    return {
        "repository": "git@gitee.example:team/kernelgen_server.git",
        "release": "v6.2.4",
        "commit": "locked-commit",
        "backend": "npu",
        "devices": ["2", "3"],
        "timing": "auto",
        "port": 18080,
        "max_workers": 2,
        "kgs_root": str(tmp_path / "kgs"),
        "state_root": str(tmp_path / "state"),
    }


def test_remote_start_uses_loopback_and_records_pid_identity(tmp_path, monkeypatch):
    kgs_root = tmp_path / "kgs"
    kgs_root.mkdir()
    captured = {}

    class Process:
        pid = 321

    def popen(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return Process()

    monkeypatch.setattr(
        remote_server,
        "_prepare",
        lambda payload: (kgs_root, {"torch": "test"}),
    )
    monkeypatch.setattr(remote_server.subprocess, "Popen", popen)
    monkeypatch.setattr(remote_server, "_process_start", lambda pid: "start-id")

    result = remote_server._start(_payload(tmp_path))

    assert result["pid"] == 321
    assert result["process_start"] == "start-id"
    assert captured["command"][captured["command"].index("--host") + 1] == "127.0.0.1"
    assert captured["kwargs"]["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "2,3"
    process = json.loads(
        (tmp_path / "state" / "process.json").read_text(encoding="utf-8")
    )
    assert process["pid"] == 321
    assert process["process_start"] == "start-id"


def test_remote_start_exports_installed_flaggems_root(tmp_path, monkeypatch):
    kgs_root = tmp_path / "kgs"
    flaggems_root = tmp_path / "FlagGems"
    kgs_root.mkdir()
    flaggems_root.mkdir()
    payload = {
        **_payload(tmp_path),
        "flaggems": {
            "root": str(flaggems_root),
            "repository": "https://example.com/FlagGems.git",
            "branch": "pinned-branch",
            "commit": "flaggems-commit",
        },
    }
    captured = {}

    class Process:
        pid = 321

    monkeypatch.setattr(
        remote_server,
        "_prepare",
        lambda payload: (kgs_root, {"torch": "test"}),
    )
    monkeypatch.setattr(
        remote_server,
        "_validate_checkout",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        remote_server.subprocess,
        "Popen",
        lambda command, **kwargs: captured.update(kwargs) or Process(),
    )
    monkeypatch.setattr(remote_server, "_process_start", lambda pid: "start-id")

    remote_server._start(payload)

    assert captured["env"]["KGS_FLAGGEMS_ROOT"] == str(flaggems_root)


def test_deployment_environment_loads_shell_file_and_restores_process_env(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / "deployment.env.sh"
    env_file.write_text(
        "KG_DEPLOYMENT_TEST=from-file\n"
        "KG_DEPLOYMENT_RELATIVE=$(pwd)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KG_DEPLOYMENT_TEST", "original")

    with remote_server._deployment_environment({"env_file": str(env_file)}):
        assert os.environ["KG_DEPLOYMENT_TEST"] == "from-file"
        assert os.environ["KG_DEPLOYMENT_RELATIVE"] == str(tmp_path)

    assert os.environ["KG_DEPLOYMENT_TEST"] == "original"
    assert "KG_DEPLOYMENT_RELATIVE" not in os.environ


def test_deployment_environment_does_not_expose_shell_output_on_failure(tmp_path):
    env_file = tmp_path / "deployment.env.sh"
    env_file.write_text(
        "echo do-not-leak-this-value >&2\nfalse\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as raised:
        with remote_server._deployment_environment({"env_file": str(env_file)}):
            pass

    assert "do-not-leak-this-value" not in str(raised.value)


def test_remote_start_does_not_export_deployment_environment_to_kgs(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / "deployment.env.sh"
    env_file.write_text("KG_DEPLOYMENT_TEST=temporary\n", encoding="utf-8")
    payload = {**_payload(tmp_path), "env_file": str(env_file)}
    observed = {}
    captured = {}

    class Process:
        pid = 321

    @contextmanager
    def deployment_environment(payload):
        original = os.environ.get("KG_DEPLOYMENT_TEST")
        os.environ["KG_DEPLOYMENT_TEST"] = "temporary"
        try:
            yield
        finally:
            if original is None:
                os.environ.pop("KG_DEPLOYMENT_TEST", None)
            else:
                os.environ["KG_DEPLOYMENT_TEST"] = original

    def prepare_checkout(*args, **kwargs):
        observed["during_prepare"] = os.environ.get("KG_DEPLOYMENT_TEST")

    monkeypatch.delenv("KG_DEPLOYMENT_TEST", raising=False)
    monkeypatch.setattr(
        remote_server,
        "_deployment_environment",
        deployment_environment,
    )
    monkeypatch.setattr(remote_server, "_validate_checkout", prepare_checkout)
    monkeypatch.setattr(remote_server, "_validate_installed_server", lambda root: None)
    monkeypatch.setattr(
        remote_server,
        "_package_versions",
        lambda: {name: None for name in remote_server.PROTECTED_DISTRIBUTIONS},
    )
    monkeypatch.setattr(
        remote_server,
        "_validate_imports",
        lambda modules: {name: "test" for name in modules},
    )
    monkeypatch.setattr(remote_server, "_validate_server_import_root", lambda root: None)
    monkeypatch.setattr(
        remote_server,
        "_run",
        lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(
        remote_server.subprocess,
        "Popen",
        lambda command, **kwargs: captured.update(kwargs) or Process(),
    )
    monkeypatch.setattr(remote_server, "_process_start", lambda pid: "start-id")

    remote_server._start(payload)

    assert observed["during_prepare"] == "temporary"
    assert "KG_DEPLOYMENT_TEST" not in captured["env"]


def test_remote_stop_refuses_reused_or_different_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(
        remote_server,
        "_active_record",
        lambda path: {"pid": 999, "process_start": "actual-start"},
    )
    monkeypatch.setattr(
        remote_server.os,
        "killpg",
        lambda *args: pytest.fail("must not signal a mismatched process"),
    )
    payload = {
        **_payload(tmp_path),
        "expected_pid": 321,
        "expected_process_start": "expected-start",
    }

    with pytest.raises(RuntimeError, match="pid no longer matches"):
        remote_server._stop(payload)


def test_remote_logs_return_incremental_base64(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / "kernelgen-server.log").write_bytes(b"one\ntwo\nthree\n")
    monkeypatch.setattr(remote_server, "_active_record", lambda path: None)

    initial = remote_server._logs({**_payload(tmp_path), "lines": 2})
    incremental = remote_server._logs(
        {**_payload(tmp_path), "offset": len(b"one\ntwo\n")}
    )

    assert base64.b64decode(initial["data"]) == b"two\nthree\n"
    assert base64.b64decode(incremental["data"]) == b"three\n"
    assert initial["running"] is False


def test_remote_prepare_skips_install_when_expected_checkout_is_importable(
    tmp_path,
    monkeypatch,
):
    kgs_root = tmp_path / "kgs"
    payload = _payload(tmp_path)
    monkeypatch.setattr(remote_server, "_validate_checkout", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        remote_server,
        "_package_versions",
        lambda: {name: None for name in remote_server.PROTECTED_DISTRIBUTIONS},
    )
    monkeypatch.setattr(
        remote_server,
        "_validate_imports",
        lambda modules: {name: "test" for name in modules},
    )
    monkeypatch.setattr(remote_server, "_validate_server_import_root", lambda root: None)
    monkeypatch.setattr(remote_server, "_validate_installed_server", lambda root: None)
    monkeypatch.setattr(
        remote_server,
        "_run",
        lambda *args, **kwargs: pytest.fail("pip must not run on a normal restart"),
    )

    _, versions = remote_server._prepare(payload)

    assert versions["kernelgen_server"] == "test"


def test_remote_prepare_reinstalls_when_server_package_is_missing(tmp_path, monkeypatch):
    kgs_root = tmp_path / "kgs"
    payload = _payload(tmp_path)
    monkeypatch.setattr(remote_server, "_validate_checkout", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        remote_server,
        "_package_versions",
        lambda: {name: None for name in remote_server.PROTECTED_DISTRIBUTIONS},
    )
    commands = []
    monkeypatch.setattr(
        remote_server,
        "_validate_imports",
        lambda modules: {name: "test" for name in modules},
    )
    monkeypatch.setattr(remote_server, "_validate_server_import_root", lambda root: None)
    monkeypatch.setattr(
        remote_server,
        "_validate_installed_server",
        lambda root: (_ for _ in ()).throw(RuntimeError("missing")),
    )
    monkeypatch.setattr(
        remote_server,
        "_run",
        lambda command, **kwargs: commands.append(command) or "",
    )

    remote_server._prepare(payload)

    assert commands == [
        [
            remote_server.sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(kgs_root / "client"),
        ],
        [
            remote_server.sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            f"{kgs_root}[server]",
        ]
    ]


def test_remote_install_rejects_client_from_another_checkout(tmp_path, monkeypatch):
    expected = tmp_path / "expected-kgs"
    foreign = tmp_path / "foreign-client" / "kernelgen_client" / "__init__.py"
    server = expected / "kernelgen_server" / "__init__.py"
    monkeypatch.setattr(
        remote_server,
        "_run",
        lambda *_args, **_kwargs: f"kernelgen_client:{foreign}\nkernelgen_server:{server}\n",
    )

    with pytest.raises(RuntimeError, match="kernelgen_client imports from"):
        remote_server._validate_installed_server(expected)


def test_remote_install_flaggems_prepares_exact_checkout(tmp_path, monkeypatch):
    flaggems_root = tmp_path / "FlagGems"
    payload = {
        **_payload(tmp_path),
        "flaggems": {
            "root": str(flaggems_root),
            "repository": "https://example.com/FlagGems.git",
            "branch": "pinned-branch",
            "commit": "flaggems-commit",
        },
    }
    monkeypatch.setattr(remote_server, "_active_record", lambda path: None)
    prepared = []
    monkeypatch.setattr(
        remote_server,
        "_prepare_checkout",
        lambda root, **options: prepared.append((root, options)),
    )

    result = remote_server._install_flaggems(payload)

    assert result == {
        "root": str(flaggems_root.resolve()),
        "commit": "flaggems-commit",
    }
    assert prepared == [
        (
            flaggems_root.resolve(),
            {
                "repository": "https://example.com/FlagGems.git",
                "release": None,
                "branch": "pinned-branch",
                "commit": "flaggems-commit",
            },
        )
    ]


def test_remote_install_flaggems_uses_deployment_environment(tmp_path, monkeypatch):
    env_file = tmp_path / "deployment.env.sh"
    env_file.write_text("HTTPS_PROXY=http://proxy.invalid:8080\n", encoding="utf-8")
    payload = {
        **_payload(tmp_path),
        "env_file": str(env_file),
        "flaggems": {
            "root": str(tmp_path / "FlagGems"),
            "repository": "https://example.com/FlagGems.git",
            "branch": "pinned-branch",
            "commit": "flaggems-commit",
        },
    }
    monkeypatch.setattr(remote_server, "_active_record", lambda path: None)
    observed = {}
    monkeypatch.setattr(
        remote_server,
        "_prepare_checkout",
        lambda *args, **kwargs: observed.update(
            https_proxy=os.environ.get("HTTPS_PROXY")
        ),
    )

    remote_server._install_flaggems(payload)

    assert observed["https_proxy"] == "http://proxy.invalid:8080"


def test_remote_doctor_does_not_require_catalog_or_framework(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    monkeypatch.setattr(remote_server, "_validate_checkout", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        remote_server,
        "_validate_imports",
        lambda modules: {name: "test" for name in modules},
    )
    monkeypatch.setattr(remote_server, "_validate_server_import_root", lambda root: None)

    result = remote_server._doctor(payload)

    assert result["ok"] is True
    assert result["versions"]["kernelgen_server"] == "test"
