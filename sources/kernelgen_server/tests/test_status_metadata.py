from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from kernelgen_server.backends.ascend import device as ascend_device
from kernelgen_server.backends.cambricon import device as cambricon_device
from kernelgen_server.backends.enflame import device as enflame_device
from kernelgen_server.backends.kunlunxin import device as kunlunxin_device
from kernelgen_server.runtime.device import (
    clean_status_value,
    device_info_from_properties,
    status_json_from_output,
    status_json_value,
    status_value_from_label,
    status_xml_value,
)


def test_clean_status_value_rejects_vendor_unavailable_sentinels():
    assert clean_status_value("Not Found") == ""
    assert clean_status_value("N/A") == ""
    assert clean_status_value("unknown error") == ""
    assert clean_status_value("3.7.0.38") == "3.7.0.38"


def test_status_parsers_accept_vendor_cli_formats():
    payload = status_json_from_output(
        'mx-smi version: 1.0\n{"versions":{"maca_version":"3.7.0.38"}}'
    )

    assert status_json_value(payload, "maca_version") == "3.7.0.38"
    assert status_value_from_label("Driver Version: 3.3.5-server", "Driver Version") == (
        "3.3.5-server"
    )
    assert status_xml_value(
        "<?xml version='1.0'?><root><product_architecture>KL3</product_architecture>"
        "<cuda_version>Not Found</cuda_version></root>",
        "product_architecture",
    ) == "KL3"
    assert status_xml_value(
        "<root><cuda_version>Not Found</cuda_version></root>",
        "cuda_version",
    ) == ""


def test_device_properties_use_observed_architecture_without_model_mapping():
    properties = SimpleNamespace(
        name="Vendor Model X",
        gcnArchName="Vendor Model X",
        major=7,
        minor=1,
        L2_cache_size=1024,
        multi_processor_count=16,
        total_memory=4096,
    )

    result = device_info_from_properties(0, "Vendor Model X", properties)

    assert result.name == "Vendor Model X"
    assert result.compute_capability == "7.1"
    assert result.l2_cache_size == 1024
    assert result.num_sm == 16


def test_device_properties_preserve_runtime_isa_string():
    properties = SimpleNamespace(
        name="Vendor Device",
        gcn_arch_name="gfx936:sramecc+:xnack-",
        major=9,
        minor=3,
    )

    result = device_info_from_properties(0, "Vendor Device", properties)

    assert result.compute_capability == "gfx936:sramecc+:xnack-"


def test_enflame_tops_runtime_integer_version_format():
    assert enflame_device._decode_tops_version(1090) == "1.9"
    assert enflame_device._decode_tops_version(12040) == "12.4"
    assert enflame_device._decode_tops_version(0) == ""


def test_enflame_status_prefers_tops_runtime_api(monkeypatch):
    monkeypatch.setattr(enflame_device, "_tops_runtime_version", lambda: "1.9")
    monkeypatch.setattr(
        enflame_device,
        "_efsmi_device_record",
        lambda detail, index: {"Ver": "1.9.10"} if detail == "DRIVER" else {},
    )

    result = enflame_device.EnflameDevice().status_metadata()

    assert result["runtime_version"] == "1.9"
    assert result["driver_version"] == "1.9.10"
    assert result["sources"]["software.runtime_version"] == (
        "libtopsrt.so.1:topsRuntimeGetVersion"
    )


def test_cambricon_status_prefers_cndev_driver_api(monkeypatch):
    monkeypatch.setattr(cambricon_device, "_cndev_driver_version", lambda: "6.2.15")
    monkeypatch.setattr(cambricon_device, "torch_runtime_version", lambda name: "4.4.1")

    result = cambricon_device.CambriconDevice().status_metadata()

    assert result["runtime_version"] == "4.4.1"
    assert result["driver_version"] == "6.2.15"
    assert result["sources"]["software.driver_version"] == (
        "libcndev.so:cndevGetDriverVersion"
    )


def test_cambricon_cnmon_driver_fallback_normalizes_prefix(monkeypatch):
    monkeypatch.setattr(
        cambricon_device,
        "run_status_command",
        lambda argv: "Card 0\n    Driver : v6.2.15\n",
    )

    assert cambricon_device._cnmon_driver_version() == "6.2.15"


def test_ascend_status_uses_cann_compiler_architecture(monkeypatch):
    monkeypatch.setattr(
        ascend_device,
        "_compiler_architecture",
        lambda soc_name: "dav-c220-cube",
    )
    monkeypatch.setattr(
        ascend_device.NpuDevice,
        "device_name",
        lambda self, device: "Ascend910B4-1",
    )

    result = ascend_device.NpuDevice().status_metadata()

    assert result["architecture"] == "dav-c220-cube"
    assert result["sources"]["target.architecture"] == (
        "tbe.common.platform.get_soc_spec(COMPILER_ARCH)"
    )


@pytest.mark.parametrize("home,toolkit,selected", [
    ("/custom/cann", "/other/cann", "/custom/cann"),
    (None, "/custom/cann", "/custom/cann"),
    ("", "/custom/cann", "/custom/cann"),
    (None, None, "/usr/local/Ascend/ascend-toolkit/latest"),
])
def test_ascend_runtime_version_follows_selected_toolkit(monkeypatch, home, toolkit, selected):
    for name, value in (("ASCEND_HOME_PATH", home), ("ASCEND_TOOLKIT_HOME", toolkit)):
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, value)
    expected = selected + "/share/info/runtime/version.info"
    monkeypatch.setattr(ascend_device, "_version_from_info_file",
                        lambda path: "9.1.0-beta.3" if str(path) == expected else "")
    monkeypatch.setattr(ascend_device, "_compiler_architecture", lambda _: "")
    monkeypatch.setattr(ascend_device.NpuDevice, "device_name", lambda *_: "")

    result = ascend_device.NpuDevice().status_metadata()
    assert result["runtime_version"] == "9.1.0-beta.3"
    assert result["sources"]["software.runtime_version"] == expected


def test_ascend_missing_selected_version_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("ASCEND_HOME_PATH", "/missing/cann")
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", "/old/cann")
    monkeypatch.setattr(ascend_device, "_version_from_info_file",
                        lambda path: "" if str(path).startswith("/missing/") else "9.0.0")
    monkeypatch.setattr(ascend_device, "_compiler_architecture", lambda _: "")
    monkeypatch.setattr(ascend_device.NpuDevice, "device_name", lambda *_: "")
    result = ascend_device.NpuDevice().status_metadata()
    assert result["runtime_version"] == ""
    assert "software.runtime_version" not in result["sources"]


def test_kunlunxin_status_does_not_replace_missing_xpu_runtime(monkeypatch):
    monkeypatch.setattr(
        kunlunxin_device,
        "run_status_command",
        lambda argv: (
            "<root>"
            "<driver_version>515.58</driver_version>"
            "<cuda_version>Not Found</cuda_version>"
            "<product_architecture>KL3</product_architecture>"
            "</root>"
        ),
    )

    result = kunlunxin_device.KunlunxinDevice().status_metadata()

    assert result["architecture"] == "KL3"
    assert result["runtime_version"] == ""
    assert result["driver_version"] == "515.58"
