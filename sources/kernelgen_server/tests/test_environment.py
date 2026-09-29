from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import pytest

from kernelgen_server.runtime import environment
from kernelgen_server.profiling.environment import (
    bind_logical_device as profiling_bind_logical_device,
)


def test_module_version_uses_worker_import_order_and_ignores_vendor_stdout(monkeypatch):
    imported = []

    def import_module(name):
        imported.append(name)
        if name == "triton":
            assert imported == ["torch", "triton"]
        print("vendor diagnostic", end="")
        return SimpleNamespace(__version__="3.5.1")

    def run(argv, **kwargs):
        assert kwargs["timeout"] == 60
        output = StringIO()
        with monkeypatch.context() as child:
            child.setattr(environment.importlib, "import_module", import_module)
            with redirect_stdout(output):
                exec(argv[-1], {})
                print("path string is NULL", end="")
        return SimpleNamespace(returncode=0, stdout=output.getvalue())

    monkeypatch.setattr(environment.subprocess, "run", run)
    assert environment._module_version("triton") == "3.5.1"


@pytest.mark.parametrize("stdout,returncode", [
    ('KGS_MODULE_VERSION="3.5.1"', 1),
    ("vendor diagnostic only", 0),
    ("KGS_MODULE_VERSION=invalid", 0),
    ("KGS_MODULE_VERSION=123", 0),
    ('KGS_MODULE_VERSION="unknown"', 0),
])
def test_module_version_does_not_infer_missing_or_failed_import(monkeypatch, stdout, returncode):
    monkeypatch.setattr(environment.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout=stdout, returncode=returncode))
    assert environment._module_version("triton") == ""


def test_module_version_timeout_is_unknown(monkeypatch):
    def timeout(*args, **kwargs):
        raise environment.subprocess.TimeoutExpired("probe", 60)
    monkeypatch.setattr(environment.subprocess, "run", timeout)
    assert environment._module_version("triton") == ""


def test_device_metadata_probe_allows_vendor_cold_import(monkeypatch):
    observed = {}

    class Result:
        returncode = 0
        stdout = (
            'KGS_ENV_JSON={"devices":[{"logical_device":"cuda:0",'
            '"name":"device"}],"metadata":{}}\n'
        )

    def fake_run(*args, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        return Result()

    monkeypatch.setattr(environment.subprocess, "run", fake_run)

    result = environment._probe_environment("metax", ["cuda:0"])
    assert result["devices"][0]["name"] == "device"
    assert observed["timeout"] == 60


def test_profiling_environment_reexports_runtime_binding():
    assert profiling_bind_logical_device is environment.bind_logical_device


def test_distribution_names_for_module_uses_package_metadata(monkeypatch):
    monkeypatch.setattr(
        environment.importlib.metadata,
        "packages_distributions",
        lambda: {"triton": ["triton", "triton_ascend", "triton"]},
        raising=False,
    )

    assert environment._distribution_names_for_module("triton") == (
        "triton",
        "triton_ascend",
    )


def test_distribution_names_for_module_supports_old_metadata_api(monkeypatch):
    class FakeDistribution:
        files = ("triton/__init__.py",)
        metadata = {"Name": "FlagTree"}

        @staticmethod
        def read_text(filename):
            return ""

    monkeypatch.setattr(
        environment.importlib.metadata,
        "packages_distributions",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        environment.importlib.metadata,
        "distributions",
        lambda: (FakeDistribution(),),
    )

    assert environment._distribution_names_for_module("triton") == ("FlagTree",)


def test_bind_backend_device_narrows_every_configured_hygon_variable():
    child_env = {
        "HIP_VISIBLE_DEVICES": "3,5",
        "ROCR_VISIBLE_DEVICES": "3,5",
        "CUDA_VISIBLE_DEVICES": "3,5",
    }

    selected = environment.bind_backend_device(child_env, "hygon", "cuda:1")

    assert selected == {
        "HIP_VISIBLE_DEVICES": "5",
        "ROCR_VISIBLE_DEVICES": "5",
        "CUDA_VISIBLE_DEVICES": "5",
    }
    assert all(child_env[name] == "5" for name in selected)


def test_bind_backend_device_narrows_both_musa_variables():
    child_env = {
        "MUSA_VISIBLE_DEVICES": "2,4",
        "MTHREADS_VISIBLE_DEVICES": "2,4",
    }

    selected = environment.bind_backend_device(child_env, "musa", "musa:0")

    assert selected == {
        "MUSA_VISIBLE_DEVICES": "2",
        "MTHREADS_VISIBLE_DEVICES": "2",
    }


def test_bind_backend_device_narrows_both_metax_variables():
    child_env = {
        "CUDA_VISIBLE_DEVICES": "0,2,4,6",
        "MACA_VISIBLE_DEVICES": "0,2,4,6",
    }

    selected = environment.bind_backend_device(child_env, "metax", "cuda:2")

    assert selected == {
        "CUDA_VISIBLE_DEVICES": "4",
        "MACA_VISIBLE_DEVICES": "4",
    }


def test_environment_info_reports_target_and_software(monkeypatch):
    monkeypatch.setattr(
        environment,
        "_probe_environment",
        lambda backend, devices: {
            "devices": [
                {
                    "logical_device": "npu:0",
                    "name": "Ascend 910B",
                    "compute_capability": "observed-architecture",
                }
            ],
            "metadata": {
                "runtime_version": "8.1",
                "driver_version": "24.1",
                "sources": {
                    "software.runtime_version": "/runtime/version.info",
                    "software.driver_version": "/driver/version.info",
                },
            },
        },
    )
    monkeypatch.setattr(
        environment,
        "_distribution_names_for_module",
        lambda name: ("triton", "triton_ascend"),
    )
    package_versions = {
        "triton": "3.5.0",
        "triton_ascend": "3.2.2",
    }
    monkeypatch.setattr(
        environment,
        "_package_version",
        lambda *names: next(
            (package_versions[name] for name in names if name in package_versions),
            "",
        ),
    )
    monkeypatch.setattr(environment, "_module_version", lambda name: "3.2.0")
    result = environment.environment_info("npu", ["npu:0"])

    assert result["target"] == {
        "backend": "ascend",
        "vendor": "huawei",
        "device": "Ascend 910B",
        "architecture": "observed-architecture",
        "devices": [
            {
                "logical_device": "npu:0",
                "name": "Ascend 910B",
                "compute_capability": "observed-architecture",
            }
        ],
    }
    assert result["software"] == {
        "language": "triton",
        "language_version": "3.2.0",
        "compiler": "triton-ascend",
        "compiler_version": "3.2.2",
        "runtime": "cann",
        "runtime_version": "8.1",
        "driver_version": "24.1",
    }
    assert result["metadata"]["complete"] is True
    assert result["metadata"]["missing"] == []
    assert result["metadata"]["sources"]["target.architecture"] == (
        "device runtime properties"
    )
    assert result["metadata"]["sources"]["software.runtime_version"] == (
        "/runtime/version.info"
    )


def test_environment_info_does_not_confuse_language_and_compiler_versions(
    monkeypatch,
):
    monkeypatch.setattr(
        environment,
        "_probe_environment",
        lambda backend, devices: {"devices": [], "metadata": {}},
    )
    monkeypatch.setattr(
        environment,
        "_distribution_names_for_module",
        lambda name: (),
    )
    monkeypatch.setattr(environment, "_package_version", lambda *names: "")
    monkeypatch.setattr(environment, "_module_version", lambda name: "3.6.0-vendor")

    result = environment.environment_info("iluvatar", ["cuda:0"])

    assert result["software"]["language_version"] == "3.6.0-vendor"
    assert result["software"]["compiler"] == "triton"
    assert result["software"]["compiler_version"] == ""
    assert "software.compiler_version" in result["metadata"]["missing"]
    assert "software.compiler_version" not in result["metadata"]["sources"]


def test_environment_info_reports_discovered_vendor_provider(monkeypatch):
    monkeypatch.setattr(
        environment,
        "_probe_environment",
        lambda backend, devices: {"devices": [], "metadata": {}},
    )
    monkeypatch.setattr(
        environment,
        "_distribution_names_for_module",
        lambda name: ("FlagTree",),
    )
    monkeypatch.setattr(
        environment,
        "_package_version",
        lambda *names: "0.6.0+iluvatar3.6" if "FlagTree" in names else "",
    )
    monkeypatch.setattr(environment, "_module_version", lambda name: "3.6.0")

    result = environment.environment_info("iluvatar", ["cuda:0"])

    assert result["software"]["language_version"] == "3.6.0"
    assert result["software"]["compiler"] == "flagtree"
    assert result["software"]["compiler_version"] == "0.6.0+iluvatar3.6"


def test_environment_info_does_not_backfill_language_from_compiler(monkeypatch):
    monkeypatch.setattr(
        environment,
        "_probe_environment",
        lambda backend, devices: {"devices": [], "metadata": {}},
    )
    monkeypatch.setattr(
        environment,
        "_compiler_info",
        lambda backend: ("flagtree", "0.6.0"),
    )
    monkeypatch.setattr(environment, "_module_version", lambda name: "")

    result = environment.environment_info("enflame", ["gcu:0"])

    assert result["software"]["language_version"] == ""
    assert result["software"]["compiler_version"] == "0.6.0"
    assert "software.language_version" in result["metadata"]["missing"]
    assert "software.language_version" not in result["metadata"]["sources"]


def test_environment_info_preserves_observed_device_identity(monkeypatch):
    monkeypatch.setattr(environment, "_module_version", lambda name: "")
    monkeypatch.setattr(
        environment,
        "_compiler_info",
        lambda backend: ("triton", ""),
    )

    cases = (
        ("enflame", "ZIXIAOC200"),
        ("hygon", "BW3000"),
        ("kunlunxin", "P800 OAM"),
    )
    for backend, observed_name in cases:
        monkeypatch.setattr(
            environment,
            "_probe_environment",
            lambda backend, devices, observed_name=observed_name: {
                "devices": [
                    {"logical_device": "cuda:0", "name": observed_name}
                ],
                "metadata": {},
            },
        )

        result = environment.environment_info(backend, ["cuda:0"])

        assert result["target"]["device"] == observed_name
        assert result["target"]["devices"][0]["name"] == observed_name
        assert "raw_name" not in result["target"]["devices"][0]


def test_environment_info_adds_display_name_without_replacing_identity(monkeypatch):
    monkeypatch.setattr(environment, "_module_version", lambda name: "")
    monkeypatch.setattr(
        environment,
        "_compiler_info",
        lambda backend: ("flagtree", ""),
    )
    monkeypatch.setattr(
        environment,
        "_probe_environment",
        lambda backend, devices: {
            "devices": [
                {"logical_device": "gcu:0", "name": "ZIXIAOC200"}
            ],
            "metadata": {},
        },
    )

    result = environment.environment_info("enflame", ["gcu:0"])

    assert result["target"]["device"] == "ZIXIAOC200"
    assert result["target"]["devices"][0]["name"] == "ZIXIAOC200"
    assert result["target"]["display_name"] == "S60"
    assert result["metadata"]["sources"]["target.display_name"] == (
        "static display mapping"
    )


def test_environment_info_does_not_infer_architecture_from_model(monkeypatch):
    monkeypatch.setattr(environment, "_module_version", lambda name: "3.2.0")
    monkeypatch.setattr(
        environment,
        "_compiler_info",
        lambda backend: ("triton-ascend", "3.2.2"),
    )
    monkeypatch.setattr(
        environment,
        "_probe_environment",
        lambda backend, devices: {
            "devices": [
                {
                    "logical_device": "npu:0",
                    "name": "Ascend 910B",
                    "compute_capability": "",
                }
            ],
            "metadata": {},
        },
    )

    result = environment.environment_info("npu", ["npu:0"])

    assert result["target"]["architecture"] == ""
    assert result["metadata"]["complete"] is False
    assert "target.architecture" in result["metadata"]["missing"]


def test_environment_info_explicit_overrides_win_and_report_source(monkeypatch):
    monkeypatch.setattr(environment, "_module_version", lambda name: "3.6.0")
    monkeypatch.setattr(
        environment,
        "_compiler_info",
        lambda backend: ("triton", "3.6.0"),
    )
    monkeypatch.setattr(
        environment,
        "_probe_environment",
        lambda backend, devices: {
            "devices": [
                {
                    "logical_device": "cuda:0",
                    "name": "Observed accelerator",
                    "compute_capability": "observed-arch",
                }
            ],
            "metadata": {
                "architecture": "cli-arch",
                "runtime_version": "runtime-probe",
                "driver_version": "driver-probe",
                "sources": {
                    "target.architecture": "vendor-cli",
                    "software.runtime_version": "vendor-cli",
                    "software.driver_version": "vendor-cli",
                },
            },
        },
    )
    monkeypatch.setenv("KGS_TARGET_ARCHITECTURE", "configured-arch")
    monkeypatch.setenv("KGS_RUNTIME_VERSION", "configured-runtime")
    monkeypatch.setenv("KGS_DRIVER_VERSION", "configured-driver")

    result = environment.environment_info("metax", ["cuda:0"])

    assert result["target"]["architecture"] == "configured-arch"
    assert result["software"]["runtime_version"] == "configured-runtime"
    assert result["software"]["driver_version"] == "configured-driver"
    assert result["metadata"]["sources"]["target.architecture"] == (
        "environment:KGS_TARGET_ARCHITECTURE"
    )
    assert result["metadata"]["sources"]["software.runtime_version"] == (
        "environment:KGS_RUNTIME_VERSION"
    )


def test_unavailable_probe_values_remain_missing(monkeypatch):
    monkeypatch.setattr(environment, "_module_version", lambda name: "3.6.0")
    monkeypatch.setattr(
        environment,
        "_compiler_info",
        lambda backend: ("triton", "3.6.0"),
    )
    monkeypatch.setattr(
        environment,
        "_probe_environment",
        lambda backend, devices: {
            "devices": [
                {
                    "logical_device": "cuda:0",
                    "name": "Observed accelerator",
                    "compute_capability": "Not Found",
                }
            ],
            "metadata": {
                "runtime_version": "Not Found",
                "driver_version": "N/A",
            },
        },
    )

    result = environment.environment_info("kunlunxin", ["cuda:0"])
    assert result["target"]["architecture"] == ""
    assert result["software"]["runtime_version"] == ""
    assert result["software"]["driver_version"] == ""
    assert {
        "target.architecture",
        "software.runtime_version",
        "software.driver_version",
    }.issubset(result["metadata"]["missing"])
