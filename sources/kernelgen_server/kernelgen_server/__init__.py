"""Hardware-neutral kernel evaluation service."""

from .protocol.version import KERNELGEN_API_VERSION, KERNELGEN_SERVER_VERSION
from .protocol.schema import (
    AdapterManifest,
    BoundEvaluateRequest,
    Definition,
    Effects,
    EvaluateRequest,
    EvaluateResponse,
    EvaluationSettings,
    EvaluatorBinding,
    Implementation,
    InspectRequest,
    OperatorContract,
    ReferenceRequest,
    ReferenceResult,
    Parameter,
    SourceFile,
    Workload,
)
from .catalog import Catalog, OperatorData, builtin_catalog_path
from .debug.jobs import DebugArtifact, DebugJob, DebugJobRequest, DebugSourceFile
from .operator_bundles import OperatorBundleInfo

__version__ = KERNELGEN_SERVER_VERSION

__all__ = [
    "AdapterManifest",
    "BoundEvaluateRequest",
    "Catalog",
    "DebugArtifact",
    "DebugJob",
    "DebugJobRequest",
    "DebugSourceFile",
    "Definition",
    "Effects",
    "EvaluateRequest",
    "EvaluateResponse",
    "EvaluationSettings",
    "EvaluatorBinding",
    "Implementation",
    "InspectRequest",
    "OperatorContract",
    "ReferenceRequest",
    "ReferenceResult",
    "KERNELGEN_API_VERSION",
    "KERNELGEN_SERVER_VERSION",
    "OperatorData",
    "OperatorBundleInfo",
    "Parameter",
    "SourceFile",
    "Workload",
    "__version__",
    "builtin_catalog_path",
]
