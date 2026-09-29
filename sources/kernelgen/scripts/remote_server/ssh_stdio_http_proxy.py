#!/usr/bin/env python3
"""Expose a remote loopback HTTP service through one SSH session per client.

JumpServer disables SSH TCP forwarding in the multi-device environment.  This
proxy listens only on local loopback and opens one SSH stdio bridge per local
TCP connection.  The remote side uses Python's standard library to connect to
the KernelGen Server inside the target container.

This compatibility implementation is not intended for high-concurrency runs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import socket
import socketserver
import subprocess
import sys
import threading
import time


REMOTE_READY_MARKER = "__KERNELGEN_SSH_HTTP_READY__"
REMOTE_CONNECTOR = r"""
import os
import socket
import sys
import threading

sock = socket.create_connection((sys.argv[1], int(sys.argv[2])))
sys.stdout.write("__KERNELGEN_SSH_HTTP_READY__\n")
sys.stdout.flush()

def stdin_to_socket():
    try:
        while True:
            # Avoid BufferedReader's interpreter-level lock here.  This is a
            # daemon thread, and a normal connection close can otherwise race
            # interpreter shutdown and abort the remote bridge process.
            chunk = os.read(sys.stdin.fileno(), 65536)
            if not chunk:
                break
            sock.sendall(chunk)
    finally:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

threading.Thread(target=stdin_to_socket, daemon=True).start()
while True:
    chunk = sock.recv(65536)
    if not chunk:
        break
    sys.stdout.buffer.write(chunk)
    sys.stdout.buffer.flush()
""".strip()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--mode", choices=("jump", "direct"), required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--container", default="-")
    parser.add_argument("--bastion", default="bastion.aiops.baai.ac.cn")
    parser.add_argument("--jump-login")
    parser.add_argument(
        "--max-ssh-sessions",
        type=int,
        default=6,
        help="maximum concurrent SSH stdio bridges; excess clients wait locally",
    )
    parser.add_argument(
        "--gateway-retry-attempts",
        type=int,
        default=6,
        help="SSH startup attempts when JumpServer reports no available gateway",
    )
    parser.add_argument(
        "--gateway-retry-base-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--gateway-retry-max-seconds",
        type=float,
        default=16.0,
    )
    parser.add_argument(
        "--ssh-ready-timeout",
        type=float,
        default=25.0,
        help="seconds allowed for the remote connector to reach the HTTP Server",
    )
    args = parser.parse_args(argv)
    if args.max_ssh_sessions <= 0:
        parser.error("--max-ssh-sessions must be positive")
    if args.gateway_retry_attempts <= 0:
        parser.error("--gateway-retry-attempts must be positive")
    if args.gateway_retry_base_seconds < 0:
        parser.error("--gateway-retry-base-seconds must be non-negative")
    if args.gateway_retry_max_seconds < args.gateway_retry_base_seconds:
        parser.error(
            "--gateway-retry-max-seconds must be at least the retry base"
        )
    if args.ssh_ready_timeout <= 0:
        parser.error("--ssh-ready-timeout must be positive")
    return args


def _ssh_command(args: argparse.Namespace) -> list[str]:
    connector = [
        "python3",
        "-u",
        "-c",
        REMOTE_CONNECTOR,
        "127.0.0.1",
        str(args.remote_port),
    ]
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-p",
        str(args.ssh_port),
        "-i",
        args.identity,
    ]
    if args.mode == "jump":
        if not args.jump_login:
            raise ValueError("--jump-login is required in jump mode")
        if args.container == "-":
            remote = connector
        else:
            remote = ["sudo", "docker", "exec", "-i", args.container, *connector]
        command.extend(
            [
                "-l",
                args.jump_login,
                args.bastion,
                shlex.join(remote),
            ]
        )
    else:
        command.extend([f"root@{args.address}", shlex.join(connector)])
    return command


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _start_ssh_bridge(
    args: argparse.Namespace,
) -> tuple[subprocess.Popen[bytes] | None, str]:
    """Start SSH and return only after the remote connector is ready.

    No client request bytes are forwarded before the readiness marker. This
    makes retries safe: a failed startup cannot have reached the HTTP Server.
    """

    process = subprocess.Popen(
        _ssh_command(args),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    ready = threading.Event()
    startup_failure = threading.Event()
    stderr_lines: list[bytes] = []
    stdout_lines: list[bytes] = []

    def drain_stderr() -> None:
        for raw_line in iter(process.stderr.readline, b""):
            stderr_lines.append(raw_line)
            if _is_gateway_unavailable(
                raw_line.decode("utf-8", errors="replace")
            ):
                startup_failure.set()
            sys.stderr.buffer.write(b"[ssh-http-proxy] " + raw_line)
            sys.stderr.buffer.flush()

    def read_startup_stdout() -> None:
        for raw_line in iter(process.stdout.readline, b""):
            if raw_line.rstrip(b"\r\n").decode(
                "utf-8", errors="replace"
            ) == REMOTE_READY_MARKER:
                ready.set()
                return
            stdout_lines.append(raw_line)
            if _is_gateway_unavailable(
                raw_line.decode("utf-8", errors="replace")
            ):
                startup_failure.set()
                return

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stdout_thread = threading.Thread(target=read_startup_stdout, daemon=True)
    stderr_thread.start()
    stdout_thread.start()
    deadline = time.monotonic() + args.ssh_ready_timeout
    while not ready.wait(timeout=0.05):
        if startup_failure.is_set():
            _terminate_process(process)
            break
        if process.poll() is not None:
            break
        if time.monotonic() >= deadline:
            stderr_lines.append(b"SSH remote connector readiness timeout\n")
            _terminate_process(process)
            break

    if ready.is_set():
        stdout_thread.join(timeout=1)
        return process, ""

    _terminate_process(process)
    stderr_thread.join(timeout=1)
    stdout_thread.join(timeout=1)
    stdout = b"".join(stdout_lines) + process.stdout.read()
    error = (b"".join(stderr_lines) + stdout).decode(
        "utf-8", errors="replace"
    ).strip()
    return None, error or "SSH bridge exited before becoming ready"


def _is_gateway_unavailable(error: str) -> bool:
    normalized = error.lower()
    return (
        "no available gateway" in normalized
        or "gateway is unavailable" in normalized
    )


def _gateway_retry_delay(
    args: argparse.Namespace,
    failure_number: int,
) -> float:
    base = min(
        args.gateway_retry_max_seconds,
        args.gateway_retry_base_seconds * (2 ** (failure_number - 1)),
    )
    return base * random.uniform(0.8, 1.2)


def _open_ssh_bridge(
    args: argparse.Namespace,
) -> tuple[subprocess.Popen[bytes] | None, bool]:
    """Open one bridge, retrying only explicit pre-request gateway failures."""

    for attempt in range(1, args.gateway_retry_attempts + 1):
        process, error = _start_ssh_bridge(args)
        if process is not None:
            return process, False
        gateway_unavailable = _is_gateway_unavailable(error)
        if not gateway_unavailable:
            print(
                f"[ssh-http-proxy] SSH bridge startup failed: {error}",
                file=sys.stderr,
                flush=True,
            )
            return None, False
        if attempt == args.gateway_retry_attempts:
            break
        delay = _gateway_retry_delay(args, attempt)
        print(
            "[ssh-http-proxy] JumpServer has no available gateway; "
            f"retrying attempt {attempt + 1}/"
            f"{args.gateway_retry_attempts} in {delay:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)
    return None, True


def _send_proxy_error(request: socket.socket, *, gateway_unavailable: bool) -> None:
    if gateway_unavailable:
        status = "503 Service Unavailable"
        code = "SSH_GATEWAY_UNAVAILABLE"
        message = (
            "JumpServer has no available gateway after bounded startup retries."
        )
    else:
        status = "502 Bad Gateway"
        code = "SSH_BRIDGE_STARTUP_FAILED"
        message = "The SSH bridge could not reach the remote KernelGen Server."
    body = json.dumps(
        {"status": "error", "error": code, "message": message},
        separators=(",", ":"),
    ).encode("utf-8")
    response = (
        f"HTTP/1.1 {status}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        + (
            "X-KernelGen-SSH-Gateway: unavailable\r\n"
            if gateway_unavailable
            else ""
        )
        + ("Retry-After: 5\r\n" if gateway_unavailable else "")
        + "\r\n"
    ).encode("ascii") + body
    try:
        request.sendall(response)
    except OSError:
        pass


class _ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        with self.server.ssh_slots:  # type: ignore[attr-defined]
            self._handle_with_slot()

    def _handle_with_slot(self) -> None:
        process, gateway_unavailable = _open_ssh_bridge(
            self.server.args  # type: ignore[attr-defined]
        )
        if process is None:
            _send_proxy_error(
                self.request,
                gateway_unavailable=gateway_unavailable,
            )
            return
        assert process.stdin is not None
        assert process.stdout is not None

        def client_to_ssh() -> None:
            try:
                while True:
                    chunk = self.request.recv(65536)
                    if not chunk:
                        break
                    process.stdin.write(chunk)
                    process.stdin.flush()
            except (BrokenPipeError, ConnectionError, OSError):
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        threading.Thread(target=client_to_ssh, daemon=True).start()
        try:
            first_line = process.stdout.readline()
            if first_line.rstrip(b"\r\n") != b"Welcome to JumpServer SSH Server":
                self.request.sendall(first_line)
            while True:
                chunk = process.stdout.read(65536)
                if not chunk:
                    break
                self.request.sendall(chunk)
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            _terminate_process(process)


class _ThreadingProxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    args = _parse_args()
    with _ThreadingProxy(("127.0.0.1", args.listen_port), _ProxyHandler) as server:
        server.args = args
        server.ssh_slots = threading.BoundedSemaphore(args.max_ssh_sessions)
        print(
            f"READY http://127.0.0.1:{args.listen_port} "
            f"-> {args.address}:{args.remote_port} "
            f"max_ssh_sessions={args.max_ssh_sessions} "
            f"gateway_retry_attempts={args.gateway_retry_attempts}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
