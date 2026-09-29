"""One public Eval timeout with internally derived runtime safety budgets."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_EVAL_TIMEOUT_SECONDS = 1500
_TRANSPORT_GRACE_SECONDS = 300
_IDLE_GRACE_SECONDS = 300
_CODER_COMPLETION_GRACE_SECONDS = 1500


@dataclass(frozen=True)
class TimeoutPolicy:
    """Derive every optimization timeout from one public Eval limit."""

    eval_timeout_seconds: int = DEFAULT_EVAL_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.eval_timeout_seconds <= 0:
            raise ValueError("eval_timeout_seconds must be positive")

    @property
    def eval_transport_timeout_seconds(self) -> int:
        return self.eval_timeout_seconds + _TRANSPORT_GRACE_SECONDS

    @property
    def coder_idle_timeout_seconds(self) -> int:
        return self.eval_transport_timeout_seconds + _IDLE_GRACE_SECONDS

    @property
    def coder_hard_timeout_seconds(self) -> int:
        return (
            self.coder_idle_timeout_seconds
            + _CODER_COMPLETION_GRACE_SECONDS
        )
