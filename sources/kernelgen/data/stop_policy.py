"""Stop policy for the agent-driven inner loop (ADR-3 task #2).

Pure functions — no files, no Ledger. ``next_verdict(snapshot, cfg)`` decides
CONTINUE vs STOP from a ledger snapshot dict + config. This is the "brain";
``Ledger.should_stop`` is the one-line wiring that feeds it ``snapshot()``.

V1 policy (per ADR-3, confirmed 2026-07-16):
- HARD TRIGGERS: ``max_round`` — total measured-round budget exhausted.
- TRIGGERS (want to stop): ``no_improvement`` — plateaued for N rounds.
- VETOES  (block stopping): ``min_rounds_floor`` — haven't run enough rounds yet.
- L1/L2 coverage is NOT a hard veto here (demoted to a soft nudge carried in the
  snapshot).

The decision reads Python's own ledger snapshot — never agent-reported metrics —
so an agent cannot fake a stop/continue (ADR-2 red line).

Extending: append a pure function to TRIGGERS or VETOES; next_verdict's
orchestration is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class StopConfig:
    """Knobs for the stop policy. All have paper-friendly defaults."""
    early_stop_rounds: int = 3       # 0 disables the no_improvement trigger
    min_rounds: int = 1              # floor: don't stop before this many rounds
    max_round: int = 15              # hard cap for total measured rounds
    soft_stop_disabled: bool = False  # paper-repro: never soft-stop (outer cap only)

    def __post_init__(self) -> None:
        if self.max_round < 1:
            raise ValueError("max_round must be at least 1")


@dataclass
class Verdict:
    """The stop decision. (snapshot is NOT here — the caller already has it.)"""
    should_continue: bool
    reason: str
    code: str = ""


# A policy is a pure function: (snapshot, cfg) -> reason string or None.
#   TRIGGER returns a reason when it WANTS to stop, else None.
#   VETO    returns a reason when it BLOCKS stopping, else None.
StopPolicy = Callable[[dict, StopConfig], Optional[str]]


def no_improvement(snap: dict, cfg: StopConfig) -> Optional[str]:
    """TRIGGER: stop once the run has plateaued for ``early_stop_rounds`` rounds."""
    n = snap.get("rounds_without_improvement", 0)
    if cfg.early_stop_rounds > 0 and n >= cfg.early_stop_rounds:
        return f"no improvement for {n} rounds"
    return None


def max_round_reached(snap: dict, cfg: StopConfig) -> Optional[str]:
    """HARD TRIGGER: stop once the total measured-round budget is exhausted."""
    n = snap.get("round_count", 0)
    if n >= cfg.max_round:
        return f"maximum round limit reached: {n}/{cfg.max_round}"
    return None


def min_rounds_floor(snap: dict, cfg: StopConfig) -> Optional[str]:
    """VETO: block stopping until at least ``min_rounds`` rounds are recorded."""
    n = snap.get("round_count", 0)
    if n < cfg.min_rounds:
        return f"only {n}/{cfg.min_rounds} rounds run"
    return None


# V1 registered hooks. Hard triggers bypass soft-stop and vetoes.
HARD_TRIGGERS: List[StopPolicy] = [max_round_reached]
TRIGGERS: List[StopPolicy] = [no_improvement]
VETOES: List[StopPolicy] = [min_rounds_floor]


def _first_reason(policies: List[StopPolicy], snap: dict, cfg: StopConfig) -> Optional[str]:
    for p in policies:
        r = p(snap, cfg)
        if r:
            return r
    return None


def next_verdict(snap: dict, cfg: StopConfig) -> Verdict:
    """Decide CONTINUE/STOP from a ledger snapshot + config.

    Order: hard safety trigger → paper-repro override → soft trigger → veto.
    """
    hard_stop_reason = _first_reason(HARD_TRIGGERS, snap, cfg)
    if hard_stop_reason is not None:
        return Verdict(False, hard_stop_reason, "max_round_reached")
    if cfg.soft_stop_disabled:
        return Verdict(True, "soft-stop disabled (paper mode)", "soft_stop_disabled")
    stop_reason = _first_reason(TRIGGERS, snap, cfg)
    if stop_reason is None:
        return Verdict(True, "no stop trigger", "continue")
    veto_reason = _first_reason(VETOES, snap, cfg)
    if veto_reason:
        return Verdict(True, f"stop deferred — {veto_reason}", "stop_deferred")
    return Verdict(False, stop_reason, "performance_plateau")
