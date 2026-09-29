"""Shared cancellation application service; no provider output-stream polling."""

from contextlib import contextmanager
from queue import SimpleQueue
import signal
import sys
import threading
import time


SERVER_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}


def reconcile_server_operations(control, *, operation_ids=None):
    """Remove only operations whose remote terminal state has been observed."""
    from kernelgen_client.http import get_operation

    unresolved = []
    for operation in control.active_server_operations():
        if operation_ids is not None and operation.operation_id not in operation_ids:
            continue
        try:
            result = get_operation(operation.operation_id, operation.server_url, timeout=10)
            if result.get("state") in SERVER_TERMINAL_STATES:
                control.unregister_server_operation(operation.operation_id)
                continue
        except Exception as exc:
            control.record_event(
                "SERVER_OPERATION_RECONCILE_FAILED", level="WARNING", visibility="DEBUG",
                message=str(exc), data={"operation_id": operation.operation_id},
            )
        unresolved.append(operation)
    return unresolved


def _cancel_registered_operation(control, operation):
    from kernelgen_client.http import ServerError, cancel_operation

    # Registration precedes POST. A concurrent cancel may reach KGS first;
    # retry only DELETE, never the execution request. Missing is not terminal.
    deadline = time.monotonic() + 10
    while True:
        try:
            return cancel_operation(operation.operation_id, operation.server_url, timeout=10)
        except ServerError as exc:
            if exc.status_code != 404 or time.monotonic() >= deadline:
                raise
            if not any(item.operation_id == operation.operation_id for item in control.active_server_operations()):
                return {"state": "NO_LONGER_ACTIVE"}
            time.sleep(0.25)


def request_run_cancellation(control, reason):
    """Persist intent, then best-effort forward every registered operation."""
    state = control.request_cancel(reason)
    requested, failed = forward_server_cancellation(control)
    return state, requested, failed


def forward_server_cancellation(control):
    """Also usable after a local owner dies, without rewriting its lifecycle."""
    requested = failed = 0
    for operation in control.active_server_operations():
        try:
            result = _cancel_registered_operation(control, operation)
            if result.get("state") in SERVER_TERMINAL_STATES:
                control.unregister_server_operation(operation.operation_id)
            requested += 1
            control.record_event(
                "SERVER_OPERATION_CANCEL_REQUESTED", stage="CANCELLING",
                data={"operation_id": operation.operation_id, "operation": operation.kind,
                      "server_state": result.get("state")},
            )
        except Exception as exc:
            failed += 1
            control.record_event(
                "SERVER_OPERATION_CANCEL_FAILED", stage="CANCELLING",
                level="WARNING", visibility="DEBUG", message=str(exc),
                data={"operation_id": operation.operation_id, "operation": operation.kind},
            )
    return requested, failed


@contextmanager
def cooperative_sigint(control):
    """Translate SIGINT to cancellation without unwinding the provider pump."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    notices = SimpleQueue()

    def consume():
        while notices.get() is not None:
            print("kg: cancellation requested; waiting for complete model output", file=sys.stderr)
            try:
                request_run_cancellation(control, "interrupted from the foreground terminal")
            except Exception as exc:
                print(f"kg: cancellation request failed: {exc}", file=sys.stderr)

    worker = threading.Thread(target=consume, name="kg-cancellation", daemon=False)
    previous = signal.signal(signal.SIGINT, lambda *_: notices.put(True))
    worker.start()
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        notices.put(None)
        worker.join()
