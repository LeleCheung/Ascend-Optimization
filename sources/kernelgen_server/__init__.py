"""Make a source checkout importable when its parent is on ``sys.path``.

KernelGen's editable install exposes the shared repository parent.  Point this
top-level package at the actual implementation directory so it resolves exactly
like the installed ``kernelgen_server`` package.
"""

from pathlib import Path

_IMPLEMENTATION = Path(__file__).resolve().parent / "kernelgen_server"
__path__.insert(0, str(_IMPLEMENTATION))

from .protocol.version import KERNELGEN_API_VERSION, KERNELGEN_SERVER_VERSION
from .catalog import (
    Catalog,
    OperatorData,
    builtin_catalog_path,
)
from .debug.jobs import DebugArtifact, DebugJob, DebugJobRequest, DebugSourceFile
from .operator_bundles import OperatorBundleInfo
from .schema import (
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
    Parameter,
    ReferenceRequest,
    ReferenceResult,
    SourceFile,
    Workload,
)

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
