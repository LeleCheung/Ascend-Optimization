"""Linux process identity shared by local leases and target-side management."""

from pathlib import Path


def process_start_identity(pid: int) -> str | None:
    """Return start ticks, or None for absent/dead processes; never fall back to PID."""
    if pid <= 0:
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"cannot determine process identity for pid {pid}") from exc
    fields = raw.rpartition(")")[2].split()
    if len(fields) < 20 or not fields[19].isdigit():
        raise RuntimeError(f"invalid process identity for pid {pid}")
    return None if fields[0] in {"Z", "X"} else fields[19]


def process_is_alive(pid: int, expected_start: str) -> bool:
    actual = process_start_identity(pid)
    return actual is not None and actual == expected_start
