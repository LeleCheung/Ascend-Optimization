"""Compatibility module alias for the pure HTTP client."""

import sys

from kernelgen_client import http as _http

sys.modules[__name__] = _http
