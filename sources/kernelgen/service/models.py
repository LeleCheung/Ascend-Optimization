"""Request/response schemas for the HTTP service.

The submit payload mirrors what ``kg run`` accepts: ``definition`` plus a free
``options`` map that is validated by the very same argparse contract via
:func:`kernelgen.framework.run_options.resolve_run_options`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SubmitRunRequest(BaseModel):
    definition: str = Field(min_length=1)
    # Resolved-run options (mode, target_hardware, eval_server, n_parallel, ...).
    # Validated downstream by resolve_run_options against the argparse contract.
    options: dict[str, Any] = Field(default_factory=dict)
    # Optional explicit workspace; a timestamped default is used when omitted.
    workspace: str | None = None


class SubmitRunResponse(BaseModel):
    workspace: str
    run_id: str
    pid: int | None = None
    status: str


class CancelRequest(BaseModel):
    workspace: str = Field(min_length=1)
    reason: str = "cancelled via HTTP service"


class KgsInstanceCreateRequest(BaseModel):
    machine: str = Field(min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=200)
    install_flaggems: bool = True


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
