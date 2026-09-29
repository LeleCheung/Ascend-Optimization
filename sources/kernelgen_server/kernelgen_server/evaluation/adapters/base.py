"""Common evaluator-adapter lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from kernelgen_server.catalog import Catalog, OperatorData
from kernelgen_server.profiling.models import (
    ProfileCommand,
    ProfileOptions,
    ProfileRequest,
)
from kernelgen_server.protocol.schema import (
    AdapterManifest,
    BoundEvaluateRequest,
    EvaluateResponse,
    PreflightResult,
    ReferenceRequest,
    ReferenceResult,
)


class EvaluatorAdapter(ABC):
    kind: str
    version = "1"

    def __init__(
        self,
        *,
        catalog: Catalog,
        operator: OperatorData,
        device: Any | None = None,
        device_string: str = "",
        backend: str = "",
    ) -> None:
        self.catalog = catalog
        self.operator = operator
        self.device = device
        self.device_string = device_string
        self.backend = backend

    @abstractmethod
    def inspect(self) -> AdapterManifest: ...

    @abstractmethod
    def preflight(self, request: BoundEvaluateRequest) -> PreflightResult: ...

    @abstractmethod
    def evaluate(self, request: BoundEvaluateRequest) -> EvaluateResponse: ...

    def reference(self, request: ReferenceRequest) -> ReferenceResult:
        return ReferenceResult(status="UNSUPPORTED", log="benchmark core reference-only requires a Gems adapter")

    def build_profile_command(
        self,
        request: ProfileRequest,
        options: ProfileOptions,
        artifact_dir: Any,
        device: str,
    ) -> ProfileCommand:
        raise NotImplementedError(f"{self.kind} adapter does not support profiling")


__all__ = ["EvaluatorAdapter"]
