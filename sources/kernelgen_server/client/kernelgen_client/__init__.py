"""Hardware-neutral client-side models and helpers for KernelGen Server."""

from .protocol.version import KERNELGEN_API_VERSION, KERNELGEN_SUPPORTED_API_VERSIONS
from .protocol.schema import (
    AdapterManifest, BoundEvaluateRequest, Definition, Effects, EvaluateRequest,
    EvaluateResponse, EvaluationSettings, EvaluatorBinding, Implementation,
    InspectRequest, OperatorContract, Parameter, ReferenceRequest,
    ReferenceResult, SourceFile, Workload,
)
from .catalog import Catalog, OperatorData
from .debug.models import DebugArtifact, DebugJob, DebugJobRequest, DebugSourceFile
from .operator_bundles import OperatorBundleInfo

__version__ = "0.1.0"

__all__ = [
    "AdapterManifest", "BoundEvaluateRequest", "Catalog", "DebugArtifact",
    "DebugJob", "DebugJobRequest", "DebugSourceFile", "Definition", "Effects",
    "EvaluateRequest", "EvaluateResponse", "EvaluationSettings", "EvaluatorBinding",
    "Implementation", "InspectRequest", "KERNELGEN_API_VERSION",
    "KERNELGEN_SUPPORTED_API_VERSIONS", "OperatorContract", "OperatorBundleInfo",
    "OperatorData", "Parameter", "ReferenceRequest", "ReferenceResult",
    "SourceFile", "Workload", "__version__",
]
