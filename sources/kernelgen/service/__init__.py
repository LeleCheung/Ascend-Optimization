"""HTTP service exposing the shared ``kernelgen.cli.api`` contract.

The service is a thin FastAPI layer over :mod:`kernelgen.cli.api`; it submits,
inspects, lists, and cancels the same local runs the ``kg`` CLI manages. Run
``python -m kernelgen.service`` to start it.
"""

from kernelgen.service.app import create_app

__all__ = ["create_app"]
