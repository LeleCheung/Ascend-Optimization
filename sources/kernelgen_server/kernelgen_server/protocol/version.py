"""Compatibility import for the shared protocol version source."""

from kernelgen_client.protocol.version import (
    ApiVersion,
    KERNELGEN_API_VERSION,
    KERNELGEN_SUPPORTED_API_VERSIONS,
)

KERNELGEN_SERVER_VERSION = "v6.5.0"

__all__ = ["ApiVersion", "KERNELGEN_API_VERSION", "KERNELGEN_SERVER_VERSION", "KERNELGEN_SUPPORTED_API_VERSIONS"]
