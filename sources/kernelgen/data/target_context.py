"""Authoritative target identity derived from KernelGen Server status."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TargetContextError(ValueError):
    """Stable target-resolution failure exposed by workflow and tool boundaries."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


class SoftwareContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = ""
    language_version: str = ""
    compiler: str = ""
    compiler_version: str = ""
    runtime: str = ""
    runtime_version: str = ""
    library: str = ""
    library_version: str = ""
    driver_version: str = ""


class TargetMetadata(BaseModel):
    """Server-reported completeness used by publication gates."""

    model_config = ConfigDict(extra="forbid")

    complete: bool = False
    missing: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_consistency(self) -> "TargetMetadata":
        if self.complete and self.missing:
            raise ValueError("complete target metadata cannot list missing fields")
        return self


class TargetContext(BaseModel):
    """Resolved target facts used by Coder, ledger, Distiller, and future KB."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    backend: str = Field(min_length=1)
    vendor: str = ""
    architecture: str = ""
    device: str = Field(min_length=1)
    capabilities: list[str] = Field(default_factory=list)
    software: SoftwareContext = Field(default_factory=SoftwareContext)
    metadata: TargetMetadata | None = None
    source: Literal["eval_service"] = "eval_service"


_DEVICE_ALIASES = (
    ("ascend910b", "Ascend910B"),
    ("910b", "Ascend910B"),
    ("ascend910a", "Ascend910A"),
    ("910a", "Ascend910A"),
    ("biv150", "BI-V150"),
    ("biv100", "BI-V100"),
    ("a100", "A100"),
    ("a800", "A800"),
    ("h100", "H100"),
    ("h200", "H200"),
    ("h800", "H800"),
    ("v100", "V100"),
    ("l40", "L40"),
    ("l20", "L20"),
    ("rtx4090", "RTX4090"),
    ("4090", "RTX4090"),
    ("rtx3090", "RTX3090"),
    ("3090", "RTX3090"),
    ("s5000", "S5000"),
    ("mlu590", "MLU590"),
    ("c500", "C500"),
    ("c550", "C550"),
    ("bw1000", "BW1000"),
    ("p800", "P800"),
    ("zw810e", "ZW810E"),
    ("zw810", "ZW810E"),
    ("s60", "S60"),
)


def _identity_key(value: Any) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def normalize_device(value: Any) -> str:
    """Normalize known device aliases while preserving unknown exact identities."""

    raw = str(value or "").strip()
    key = _identity_key(raw)
    for marker, canonical in _DEVICE_ALIASES:
        if marker in key:
            return canonical
    return raw


def _normalize_backend(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "npu": "ascend",
        "cann": "ascend",
        "ascend_npu": "ascend",
        "gpu": "cuda",
        "nvidia": "cuda",
    }.get(normalized, normalized)


def build_target_context(
    *,
    target_hardware: str,
    implementation_language: str,
    service_status: dict[str, Any],
) -> TargetContext:
    """Build the jiabei-style TargetContext with strict Server device authority."""

    reported_target = (
        service_status.get("target")
        if isinstance(service_status.get("target"), dict)
        else {}
    )
    reported_device = str(reported_target.get("device") or "").strip()
    if not reported_device:
        raise TargetContextError(
            "TARGET_DEVICE_MISSING",
            "KernelGen Server /status.target.device is required; "
            "target_hardware fallback is disabled",
        )

    actual_device = normalize_device(reported_device)
    requested_device = normalize_device(target_hardware)
    if requested_device and _identity_key(requested_device) != _identity_key(actual_device):
        raise TargetContextError(
            "TARGET_HARDWARE_MISMATCH",
            f"input target_hardware {target_hardware!r} does not match "
            f"KernelGen Server device {reported_device!r}",
        )

    reported_software = (
        service_status.get("software")
        if isinstance(service_status.get("software"), dict)
        else {}
    )
    reported_language = str(reported_software.get("language") or "")
    if (
        reported_language
        and _identity_key(reported_language)
        != _identity_key(implementation_language)
    ):
        raise TargetContextError(
            "TARGET_LANGUAGE_MISMATCH",
            f"implementation language {implementation_language!r} does not match "
            f"KernelGen Server language {reported_language!r}",
        )

    raw_metadata = service_status.get("metadata")
    metadata = None
    if raw_metadata is not None:
        if not isinstance(raw_metadata, dict):
            raise TargetContextError(
                "TARGET_METADATA_INVALID",
                "KernelGen Server /status.metadata must be an object",
            )
        raw_complete = raw_metadata.get("complete", False)
        raw_missing = raw_metadata.get("missing", [])
        if not isinstance(raw_complete, bool) or not isinstance(raw_missing, list):
            raise TargetContextError(
                "TARGET_METADATA_INVALID",
                "KernelGen Server /status.metadata requires boolean complete "
                "and list missing fields",
            )
        missing = sorted(
            {str(item).strip() for item in raw_missing if str(item).strip()}
        )
        if raw_complete and missing:
            raise TargetContextError(
                "TARGET_METADATA_INCONSISTENT",
                "KernelGen Server reports complete metadata while listing "
                "missing fields: " + ", ".join(missing),
            )
        metadata = TargetMetadata(complete=raw_complete, missing=missing)

    backend = _normalize_backend(
        reported_target.get("backend") or service_status.get("backend")
    )
    if not backend:
        raise TargetContextError(
            "TARGET_BACKEND_MISSING",
            "KernelGen Server /status target/backend identity is required",
        )

    return TargetContext(
        backend=backend,
        vendor=str(reported_target.get("vendor") or ""),
        architecture=str(reported_target.get("architecture") or ""),
        device=actual_device,
        capabilities=sorted(
            {
                str(item)
                for item in reported_target.get("capabilities", [])
                if str(item)
            }
        ),
        software=SoftwareContext(
            language=implementation_language,
            language_version=str(reported_software.get("language_version") or ""),
            compiler=str(reported_software.get("compiler") or ""),
            compiler_version=str(reported_software.get("compiler_version") or ""),
            runtime=str(reported_software.get("runtime") or ""),
            runtime_version=str(reported_software.get("runtime_version") or ""),
            library=str(reported_software.get("library") or ""),
            library_version=str(reported_software.get("library_version") or ""),
            driver_version=str(reported_software.get("driver_version") or ""),
        ),
        metadata=metadata,
    )
