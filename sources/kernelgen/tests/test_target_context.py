"""TargetContext resolution and strict Server device authority."""

import pytest

from kernelgen.data.target_context import (
    TargetContextError,
    build_target_context,
    normalize_device,
)
from kernelgen.knowledge.context import (
    build_target_context as build_knowledge_target_context,
)
from kernelgen.knowledge.publishing.materialize import _validate_publish_target
from kernelgen.workflows import knowledge_bridge


def _status(device: str | None) -> dict:
    return {
        "status": "ok",
        "backend": "npu",
        "target": {
            "backend": "ascend",
            "vendor": "huawei",
            "architecture": "DAV_2201",
            "device": device,
            "capabilities": ["triton"],
        },
        "software": {
            "language": "triton",
            "language_version": "3.2.0",
            "compiler": "bisheng",
            "compiler_version": "1",
            "runtime": "cann",
            "runtime_version": "9.0.0",
            "driver_version": "25.2.0",
        },
        "metadata": {"complete": True, "missing": []},
    }


def test_known_ascend_aliases_normalize_to_one_device():
    assert normalize_device("910B") == "Ascend910B"
    assert normalize_device("Ascend 910B") == "Ascend910B"
    assert normalize_device("Ascend910B") == "Ascend910B"


def test_metax_c500_alias_normalizes_to_c500():
    assert normalize_device("C500") == "C500"
    assert normalize_device("MetaX C500") == "C500"


@pytest.mark.parametrize(
    ("reported", "canonical"),
    [
        ("NVIDIA A100-SXM4-40GB", "A100"),
        ("BI-V150", "BI-V150"),
        ("Hygon BW1000", "BW1000"),
        ("MUSA S5000", "S5000"),
        ("Kunlunxin P800", "P800"),
        ("PPU ZW810E", "ZW810E"),
        ("Enflame S60", "S60"),
        ("MetaX C550", "C550"),
        ("Cambricon MLU590", "MLU590"),
    ],
)
def test_production_device_aliases_have_stable_identity(reported, canonical):
    assert normalize_device(reported) == canonical


def test_server_device_is_authoritative_and_preserves_context():
    context = build_target_context(
        target_hardware="910B",
        implementation_language="triton",
        service_status=_status("Ascend 910B"),
    )

    assert context.device == "Ascend910B"
    assert context.schema_version == "1.0"
    assert context.backend == "ascend"
    assert context.architecture == "DAV_2201"
    assert context.software.compiler == "bisheng"
    assert context.metadata is not None
    assert context.metadata.complete is True
    assert context.metadata.missing == []
    assert context.source == "eval_service"


def test_empty_input_uses_server_device_without_fallback():
    context = build_target_context(
        target_hardware="",
        implementation_language="triton",
        service_status=_status("Ascend 910B"),
    )

    assert context.device == "Ascend910B"


def test_missing_server_device_is_rejected_even_with_input():
    with pytest.raises(TargetContextError, match="TARGET_DEVICE_MISSING"):
        build_target_context(
            target_hardware="Ascend910B",
            implementation_language="triton",
            service_status=_status(None),
        )


def test_missing_server_backend_is_rejected():
    status = _status("Ascend 910B")
    status["backend"] = ""
    status["target"]["backend"] = ""

    with pytest.raises(TargetContextError, match="TARGET_BACKEND_MISSING"):
        build_target_context(
            target_hardware="Ascend910B",
            implementation_language="triton",
            service_status=status,
        )


def test_input_and_server_device_mismatch_is_rejected():
    with pytest.raises(TargetContextError, match="TARGET_HARDWARE_MISMATCH"):
        build_target_context(
            target_hardware="H100",
            implementation_language="triton",
            service_status=_status("Ascend 910B"),
        )


def test_language_mismatch_is_rejected_instead_of_dropping_software():
    with pytest.raises(TargetContextError, match="TARGET_LANGUAGE_MISMATCH"):
        build_target_context(
            target_hardware="Ascend910B",
            implementation_language="cuda",
            service_status=_status("Ascend 910B"),
        )


def test_incomplete_server_metadata_is_preserved_and_blocks_publication():
    status = _status("Ascend 910B")
    status["metadata"] = {
        "complete": False,
        "missing": ["software.runtime_version"],
    }
    context = build_knowledge_target_context(
        target_hardware="Ascend910B",
        implementation_language="triton",
        service_status=status,
    )

    assert context.metadata is not None
    assert context.metadata.complete is False
    assert context.metadata.missing == ["software.runtime_version"]
    with pytest.raises(
        ValueError,
        match="complete Server status metadata: software.runtime_version",
    ):
        _validate_publish_target(context)


def test_inconsistent_server_metadata_is_rejected():
    status = _status("Ascend 910B")
    status["metadata"] = {
        "complete": True,
        "missing": ["software.runtime_version"],
    }

    with pytest.raises(TargetContextError, match="TARGET_METADATA_INCONSISTENT"):
        build_target_context(
            target_hardware="Ascend910B",
            implementation_language="triton",
            service_status=status,
        )


def test_knowledge_bridge_uses_shared_status_client(monkeypatch):
    from kernelgen.tools import kernelgen_server_adapter

    calls = []

    def get_service_status(server_url, *, timeout):
        calls.append((server_url, timeout))
        return _status("Ascend910B4-1")

    monkeypatch.setattr(
        kernelgen_server_adapter,
        "get_service_status",
        get_service_status,
    )

    status = knowledge_bridge._service_status("http://remote-server:18306")

    assert status["target"]["device"] == "Ascend910B4-1"
    assert calls == [("http://remote-server:18306", 45)]
