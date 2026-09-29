#!/usr/bin/env python3
"""Multiplex local HTTP connections over one persistent SSH stdio session.

JumpServer disables ordinary SSH TCP forwarding in the multi-device
environment. This proxy keeps one SSH process per target and frames multiple
logical TCP streams over that process's stdin and stdout. The remote helper
opens one loopback TCP connection to KernelGen Server per logical stream.

Use the adjacent ``run_persistent_remote_http_proxy.sh`` operational entrypoint.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import shlex
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field


REMOTE_READY_MARKER = b"__KERNELGEN_SSH_HTTP_MUX_READY_V1__"
FRAME_OPEN = 1
FRAME_OPENED = 2
FRAME_DATA = 3
FRAME_EOF = 4
FRAME_CLOSE = 5
FRAME_ERROR = 6
FRAME_HEADER = struct.Struct("!BII")
MAX_FRAME_PAYLOAD = 1024 * 1024
IO_CHUNK_SIZE = 64 * 1024


REMOTE_MULTIPLEXER = r"""
import os
import socket
import struct
import sys
import threading

READY = b"__KERNELGEN_SSH_HTTP_MUX_READY_V1__\n"
OPEN = 1
OPENED = 2
DATA = 3
EOF_FRAME = 4
CLOSE = 5
ERROR = 6
HEADER = struct.Struct("!BII")
MAX_PAYLOAD = 1024 * 1024
CHUNK_SIZE = 64 * 1024

target_host = sys.argv[1]
target_port = int(sys.argv[2])
stdin_fd = sys.stdin.fileno()
stdout_fd = sys.stdout.fileno()
stdout_lock = threading.Lock()
streams_lock = threading.Lock()
streams = {}

def read_exact(size):
    chunks = []
    remaining = size
    while remaining:
        chunk = os.read(stdin_fd, remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def write_all(data):
    view = memoryview(data)
    while view:
        written = os.write(stdout_fd, view)
        view = view[written:]

def send_frame(kind, stream_id, payload=b""):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("frame payload is too large")
    packet = HEADER.pack(kind, stream_id, len(payload)) + payload
    with stdout_lock:
        write_all(packet)

def remove_stream(stream_id, expected=None):
    with streams_lock:
        sock = streams.get(stream_id)
        if sock is None or (expected is not None and sock is not expected):
            return None
        streams.pop(stream_id, None)
        return sock

def close_socket(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass

def fail_stream(stream_id, message, expected=None):
    sock = remove_stream(stream_id, expected)
    if sock is not None:
        close_socket(sock)
    try:
        send_frame(ERROR, stream_id, message.encode("utf-8", errors="replace")[:4096])
    except OSError:
        pass

def socket_to_stdout(stream_id, sock):
    try:
        while True:
            chunk = sock.recv(CHUNK_SIZE)
            if not chunk:
                break
            send_frame(DATA, stream_id, chunk)
        if remove_stream(stream_id, sock) is not None:
            close_socket(sock)
            send_frame(CLOSE, stream_id)
    except (ConnectionError, OSError) as exc:
        fail_stream(stream_id, f"remote Server stream failed: {exc}", sock)

def open_stream(stream_id):
    with streams_lock:
        duplicate = stream_id in streams
    if duplicate:
        send_frame(ERROR, stream_id, b"duplicate stream id")
        return
    try:
        sock = socket.create_connection((target_host, target_port), timeout=10)
        sock.settimeout(None)
    except OSError as exc:
        send_frame(
            ERROR,
            stream_id,
            f"cannot connect to remote KernelGen Server: {exc}".encode(
                "utf-8", errors="replace"
            )[:4096],
        )
        return
    with streams_lock:
        streams[stream_id] = sock
    send_frame(OPENED, stream_id)
    threading.Thread(
        target=socket_to_stdout,
        args=(stream_id, sock),
        daemon=True,
    ).start()

def handle_frame(kind, stream_id, payload):
    if kind == OPEN:
        open_stream(stream_id)
        return
    with streams_lock:
        sock = streams.get(stream_id)
    if sock is None:
        return
    if kind == DATA:
        try:
            sock.sendall(payload)
        except OSError as exc:
            fail_stream(stream_id, f"cannot write to remote Server: {exc}", sock)
    elif kind == EOF_FRAME:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
    elif kind == CLOSE:
        if remove_stream(stream_id, sock) is not None:
            close_socket(sock)
    else:
        fail_stream(stream_id, f"unexpected frame type: {kind}", sock)

write_all(READY)
try:
    while True:
        header = read_exact(HEADER.size)
        kind, stream_id, payload_size = HEADER.unpack(header)
        if payload_size > MAX_PAYLOAD:
            raise ValueError("frame payload is too large")
        payload = read_exact(payload_size) if payload_size else b""
        handle_frame(kind, stream_id, payload)
except (EOFError, BrokenPipeError):
    pass
finally:
    with streams_lock:
        remaining = list(streams.values())
        streams.clear()
    for sock in remaining:
        close_socket(sock)
""".strip()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--ready-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("jump", "direct"))
    parser.add_argument("--address")
    parser.add_argument("--ssh-port", type=int)
    parser.add_argument("--identity")
    parser.add_argument(
        "--ssh-command",
        help="base SSH connection command; must not include a remote command or TTY option",
    )
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--container", default="-")
    parser.add_argument("--bastion", default="bastion.aiops.baai.ac.cn")
    parser.add_argument("--jump-login")
    parser.add_argument(
        "--max-streams",
        type=int,
        default=16,
        help="maximum concurrent logical HTTP connections over the one SSH session",
    )
    parser.add_argument(
        "--gateway-retry-attempts",
        type=int,
        default=6,
        help="SSH startup attempts when JumpServer reports no available gateway",
    )
    parser.add_argument("--gateway-retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--gateway-retry-max-seconds", type=float, default=16.0)
    parser.add_argument("--ssh-ready-timeout", type=float, default=25.0)
    parser.add_argument("--remote-open-timeout", type=float, default=15.0)
    parser.add_argument("--server-alive-interval", type=int, default=6)
    parser.add_argument("--server-alive-count-max", type=int, default=30)
    args = parser.parse_args(argv)
    if args.max_streams <= 0:
        parser.error("--max-streams must be positive")
    if args.gateway_retry_attempts <= 0:
        parser.error("--gateway-retry-attempts must be positive")
    if args.gateway_retry_base_seconds < 0:
        parser.error("--gateway-retry-base-seconds must be non-negative")
    if args.gateway_retry_max_seconds < args.gateway_retry_base_seconds:
        parser.error("--gateway-retry-max-seconds must be at least the retry base")
    if args.ssh_ready_timeout <= 0:
        parser.error("--ssh-ready-timeout must be positive")
    if args.remote_open_timeout <= 0:
        parser.error("--remote-open-timeout must be positive")
    if args.server_alive_interval <= 0:
        parser.error("--server-alive-interval must be positive")
    if args.server_alive_count_max <= 0:
        parser.error("--server-alive-count-max must be positive")
    if args.container != "-" and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.container):
        parser.error("--container must be '-' or a Docker container name")
    if args.ssh_command:
        if any(character in args.ssh_command for character in ("\n", "\r", "\0")):
            parser.error("--ssh-command must be a single command line")
    else:
        missing = [
            name
            for name in ("mode", "address", "ssh_port", "identity")
            if getattr(args, name) in (None, "")
        ]
        if missing:
            parser.error(
                "--mode, --address, --ssh-port and --identity are required "
                "unless --ssh-command is provided"
            )
    return args


def _ssh_command(args: argparse.Namespace) -> list[str]:
    connector = [
        getattr(args, "remote_python", "python3"),
        "-u",
        "-c",
        REMOTE_MULTIPLEXER,
        "127.0.0.1",
        str(args.remote_port),
    ]
    if args.container == "-":
        remote = connector
    else:
        remote = ["sudo", "docker", "exec", "-i", args.container, *connector]
    ssh_command = getattr(args, "ssh_command", None)
    if ssh_command:
        command = shlex.split(ssh_command)
        if not command or os.path.basename(command[0]) != "ssh":
            raise ValueError("--ssh-command must invoke ssh directly")
        if any(option == "-t" or option.startswith("-tt") for option in command[1:]):
            raise ValueError("--ssh-command must not allocate a TTY")
        managed_options = [
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            f"ServerAliveInterval={args.server_alive_interval}",
            "-o",
            f"ServerAliveCountMax={args.server_alive_count_max}",
            "-o",
            "TCPKeepAlive=yes",
        ]
        return [command[0], *managed_options, *command[1:], shlex.join(remote)]
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        f"ServerAliveInterval={args.server_alive_interval}",
        "-o",
        f"ServerAliveCountMax={args.server_alive_count_max}",
        "-o",
        "TCPKeepAlive=yes",
        "-p",
        str(args.ssh_port),
        "-i",
        args.identity,
    ]
    if args.mode == "jump":
        if not args.jump_login:
            raise ValueError("--jump-login is required in jump mode")
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


def _is_gateway_unavailable(error: str) -> bool:
    normalized = error.lower()
    return "no available gateway" in normalized or "gateway is unavailable" in normalized


def _gateway_retry_delay(args: argparse.Namespace, failure_number: int) -> float:
    base = min(
        args.gateway_retry_max_seconds,
        args.gateway_retry_base_seconds * (2 ** (failure_number - 1)),
    )
    return base * random.uniform(0.8, 1.2)


class BridgeUnavailable(RuntimeError):
    def __init__(self, message: str, *, gateway_unavailable: bool = False):
        super().__init__(message)
        self.gateway_unavailable = gateway_unavailable


def _start_ssh_bridge(
    args: argparse.Namespace,
) -> tuple[subprocess.Popen[bytes] | None, str]:
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
            if _is_gateway_unavailable(raw_line.decode("utf-8", errors="replace")):
                startup_failure.set()
            sys.stderr.buffer.write(b"[ssh-http-mux-proxy] " + raw_line)
            sys.stderr.buffer.flush()

    def read_startup_stdout() -> None:
        for raw_line in iter(process.stdout.readline, b""):
            if raw_line.rstrip(b"\r\n") == REMOTE_READY_MARKER:
                ready.set()
                return
            stdout_lines.append(raw_line)
            if _is_gateway_unavailable(raw_line.decode("utf-8", errors="replace")):
                startup_failure.set()
                return

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stdout_thread = threading.Thread(target=read_startup_stdout, daemon=True)
    stderr_thread.start()
    stdout_thread.start()
    deadline = time.monotonic() + args.ssh_ready_timeout
    while not ready.wait(timeout=0.05):
        if startup_failure.is_set() or process.poll() is not None:
            break
        if time.monotonic() >= deadline:
            stderr_lines.append(b"SSH remote multiplexer readiness timeout\n")
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
    return None, error or "SSH multiplexer exited before becoming ready"


def _open_ssh_bridge(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    for attempt in range(1, args.gateway_retry_attempts + 1):
        process, error = _start_ssh_bridge(args)
        if process is not None:
            return process
        gateway_unavailable = _is_gateway_unavailable(error)
        if not gateway_unavailable:
            raise BridgeUnavailable(error)
        if attempt == args.gateway_retry_attempts:
            raise BridgeUnavailable(error, gateway_unavailable=True)
        delay = _gateway_retry_delay(args, attempt)
        print(
            "[ssh-http-mux-proxy] JumpServer has no available gateway; "
            f"retrying attempt {attempt + 1}/{args.gateway_retry_attempts} "
            f"in {delay:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)
    raise AssertionError("unreachable")


def _read_exact(file_descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(file_descriptor, remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(file_descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(file_descriptor, view)
        view = view[written:]


_CLOSE_SENTINEL = object()


@dataclass
class _StreamState:
    stream_id: int
    generation: int
    client: socket.socket
    manager: "PersistentMultiplexBridge"
    opened: threading.Event = field(default_factory=threading.Event)
    closed: threading.Event = field(default_factory=threading.Event)
    outgoing: queue.Queue[bytes | object] = field(default_factory=queue.Queue)
    lock: threading.Lock = field(default_factory=threading.Lock)
    open_succeeded: bool = False
    error: str = ""
    received_data: bool = False
    finished: bool = False

    def start_writer(self) -> None:
        threading.Thread(target=self._write_to_client, daemon=True).start()

    def mark_opened(self) -> None:
        with self.lock:
            if self.finished:
                return
            self.open_succeeded = True
            self.opened.set()

    def deliver(self, payload: bytes) -> None:
        with self.lock:
            if self.finished:
                return
            self.received_data = True
        self.outgoing.put(payload)

    def finish(self, error: str = "", *, synthesize_http_error: bool = False) -> None:
        with self.lock:
            if self.finished:
                return
            self.finished = True
            self.error = error
            should_send_error = synthesize_http_error and not self.received_data
            self.opened.set()
        if should_send_error:
            self.outgoing.put(_proxy_error_response(False, error))
        self.outgoing.put(_CLOSE_SENTINEL)

    def _write_to_client(self) -> None:
        write_failed = False
        try:
            while True:
                item = self.outgoing.get()
                if item is _CLOSE_SENTINEL:
                    break
                assert isinstance(item, bytes)
                self.client.sendall(item)
        except (BrokenPipeError, ConnectionError, OSError):
            write_failed = True
        finally:
            try:
                self.client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.client.close()
            except OSError:
                pass
            self.closed.set()
            if write_failed:
                self.manager.close_stream(self, notify_remote=True)


class PersistentMultiplexBridge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._connect_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stream_slots = threading.BoundedSemaphore(args.max_streams)
        self._streams: dict[int, _StreamState] = {}
        self._process: subprocess.Popen[bytes] | None = None
        self._generation = 0
        self._next_stream_id = 1
        self._shutting_down = False

    def start(self) -> None:
        self._ensure_connected()

    def close(self) -> None:
        with self._state_lock:
            self._shutting_down = True
            process = self._process
            generation = self._generation
        self._fail_bridge(generation, "persistent SSH proxy stopped", terminate=False)
        if process is not None:
            _terminate_process(process)

    def open_stream(self, client: socket.socket) -> _StreamState:
        self._stream_slots.acquire()
        state: _StreamState | None = None
        try:
            generation = self._ensure_connected()
            with self._state_lock:
                if self._shutting_down:
                    raise BridgeUnavailable("persistent SSH proxy is stopping")
                stream_id = self._next_stream_id
                self._next_stream_id = (self._next_stream_id + 1) & 0xFFFFFFFF
                if self._next_stream_id == 0:
                    self._next_stream_id = 1
                state = _StreamState(stream_id, generation, client, self)
                self._streams[stream_id] = state
            self._send_frame(generation, FRAME_OPEN, stream_id)
            if not state.opened.wait(timeout=self.args.remote_open_timeout):
                self.close_stream(state, notify_remote=True)
                raise BridgeUnavailable("remote KernelGen Server stream open timed out")
            with state.lock:
                open_succeeded = state.open_succeeded
                stream_finished = state.finished
                stream_error = state.error
            if not open_succeeded or stream_finished:
                error = stream_error or "remote KernelGen Server stream open failed"
                self.close_stream(state, notify_remote=False)
                raise BridgeUnavailable(error)
            state.start_writer()
            return state
        except Exception:
            if state is None:
                self._stream_slots.release()
            else:
                self.close_stream(state, notify_remote=False)
            raise

    def send_data(self, state: _StreamState, payload: bytes) -> None:
        for offset in range(0, len(payload), MAX_FRAME_PAYLOAD):
            self._send_frame(
                state.generation,
                FRAME_DATA,
                state.stream_id,
                payload[offset : offset + MAX_FRAME_PAYLOAD],
            )

    def send_eof(self, state: _StreamState) -> None:
        self._send_frame(state.generation, FRAME_EOF, state.stream_id)

    def close_stream(self, state: _StreamState, *, notify_remote: bool) -> None:
        removed = self._remove_stream(state)
        if not removed:
            return
        if notify_remote:
            try:
                self._send_frame(state.generation, FRAME_CLOSE, state.stream_id)
            except BridgeUnavailable:
                pass
        state.finish()

    def _ensure_connected(self) -> int:
        with self._connect_lock:
            with self._state_lock:
                if self._shutting_down:
                    raise BridgeUnavailable("persistent SSH proxy is stopping")
                process = self._process
                generation = self._generation
            if process is not None and process.poll() is None:
                return generation
            if process is not None:
                self._fail_bridge(generation, "persistent SSH session exited")
            process = _open_ssh_bridge(self.args)
            with self._state_lock:
                if self._shutting_down:
                    _terminate_process(process)
                    raise BridgeUnavailable("persistent SSH proxy is stopping")
                self._generation += 1
                generation = self._generation
                self._process = process
            threading.Thread(
                target=self._read_bridge,
                args=(process, generation),
                daemon=True,
            ).start()
            return generation

    def _send_frame(
        self,
        generation: int,
        kind: int,
        stream_id: int,
        payload: bytes = b"",
    ) -> None:
        if len(payload) > MAX_FRAME_PAYLOAD:
            raise ValueError("frame payload is too large")
        packet = FRAME_HEADER.pack(kind, stream_id, len(payload)) + payload
        failure: OSError | None = None
        with self._write_lock:
            with self._state_lock:
                process = self._process
                current_generation = self._generation
            if process is None or generation != current_generation or process.poll() is not None:
                raise BridgeUnavailable("persistent SSH session is unavailable")
            assert process.stdin is not None
            try:
                _write_all(process.stdin.fileno(), packet)
            except OSError as exc:
                failure = exc
        if failure is not None:
            self._fail_bridge(generation, f"persistent SSH session write failed: {failure}")
            raise BridgeUnavailable(str(failure))

    def _read_bridge(
        self,
        process: subprocess.Popen[bytes],
        generation: int,
    ) -> None:
        assert process.stdout is not None
        try:
            while True:
                header = _read_exact(process.stdout.fileno(), FRAME_HEADER.size)
                kind, stream_id, payload_size = FRAME_HEADER.unpack(header)
                if payload_size > MAX_FRAME_PAYLOAD:
                    raise ValueError("remote frame payload is too large")
                payload = (
                    _read_exact(process.stdout.fileno(), payload_size)
                    if payload_size
                    else b""
                )
                self._dispatch_frame(generation, kind, stream_id, payload)
        except (EOFError, OSError, ValueError) as exc:
            self._fail_bridge(generation, f"persistent SSH session ended: {exc}")

    def _dispatch_frame(
        self,
        generation: int,
        kind: int,
        stream_id: int,
        payload: bytes,
    ) -> None:
        with self._state_lock:
            state = self._streams.get(stream_id)
        if state is None or state.generation != generation:
            return
        if kind == FRAME_OPENED:
            state.mark_opened()
        elif kind == FRAME_DATA:
            state.deliver(payload)
        elif kind == FRAME_CLOSE:
            if self._remove_stream(state):
                state.finish()
        elif kind == FRAME_ERROR:
            error = payload.decode("utf-8", errors="replace")
            if self._remove_stream(state):
                state.finish(
                    error,
                    synthesize_http_error=state.open_succeeded,
                )
        else:
            self._fail_bridge(
                generation,
                f"unexpected remote frame type {kind}",
            )

    def _remove_stream(self, state: _StreamState) -> bool:
        with self._state_lock:
            if self._streams.get(state.stream_id) is not state:
                return False
            self._streams.pop(state.stream_id, None)
        self._stream_slots.release()
        return True

    def _fail_bridge(
        self,
        generation: int,
        error: str,
        *,
        terminate: bool = True,
    ) -> None:
        with self._state_lock:
            if generation != self._generation:
                return
            process = self._process
            self._process = None
            states = [
                state
                for state in self._streams.values()
                if state.generation == generation
            ]
            for state in states:
                self._streams.pop(state.stream_id, None)
        for _ in states:
            self._stream_slots.release()
        for state in states:
            state.finish(
                error,
                synthesize_http_error=state.open_succeeded,
            )
        if terminate and process is not None:
            _terminate_process(process)


def _proxy_error_response(gateway_unavailable: bool, detail: str = "") -> bytes:
    if gateway_unavailable:
        status = "503 Service Unavailable"
        code = "SSH_GATEWAY_UNAVAILABLE"
        message = "JumpServer has no available gateway after bounded startup retries."
    else:
        status = "502 Bad Gateway"
        code = "SSH_MULTIPLEX_BRIDGE_FAILED"
        message = "The persistent SSH bridge could not reach the remote KernelGen Server."
    if detail:
        print(f"[ssh-http-mux-proxy] {code}: {detail}", file=sys.stderr, flush=True)
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
    return response


class _ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        state: _StreamState | None = None
        bridge: PersistentMultiplexBridge = self.server.bridge  # type: ignore[attr-defined]
        try:
            state = bridge.open_stream(self.request)
            while not state.closed.is_set():
                chunk = self.request.recv(IO_CHUNK_SIZE)
                if not chunk:
                    bridge.send_eof(state)
                    state.closed.wait()
                    return
                bridge.send_data(state, chunk)
        except BridgeUnavailable as exc:
            if state is None:
                try:
                    self.request.sendall(
                        _proxy_error_response(exc.gateway_unavailable, str(exc))
                    )
                except OSError:
                    pass
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            if state is not None:
                bridge.close_stream(state, notify_remote=True)


class _ThreadingProxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    args = _parse_args()
    # Reserve the listener before starting SSH; never borrow another proxy's health.
    with _ThreadingProxy(("127.0.0.1", args.listen_port), _ProxyHandler) as server:
        bridge = PersistentMultiplexBridge(args)
        try:
            try:
                bridge.start()
            except BridgeUnavailable as exc:
                print(f"[ssh-http-mux-proxy] initial SSH bridge failed: {exc}", file=sys.stderr, flush=True)
                return 3 if exc.gateway_unavailable else 2
            server.bridge = bridge
            if args.ready_fd is not None:
                os.write(args.ready_fd, b"READY\n")
                os.close(args.ready_fd)
            print(
                f"READY http://127.0.0.1:{args.listen_port} "
                f"-> {args.address}:{args.remote_port} persistent_ssh=1 "
                f"max_streams={args.max_streams} "
                f"server_alive={args.server_alive_interval}/"
                f"{args.server_alive_count_max}",
                flush=True,
            )
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        finally:
            bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
