"""KernelGen Server release and public API versions."""

from typing import Literal, TypeAlias


KERNELGEN_API_VERSION = "v6.2"
KERNELGEN_SUPPORTED_API_VERSIONS = ("v6.0", "v6.2")

ApiVersion: TypeAlias = Literal["v6.0", "v6.2"]

__all__ = [
    "ApiVersion",
    "KERNELGEN_API_VERSION",
    "KERNELGEN_SUPPORTED_API_VERSIONS",
]
