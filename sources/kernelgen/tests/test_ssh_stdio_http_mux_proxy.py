from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import threading
import time
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _REPO_ROOT / "scripts" / "remote_server"
_MODULE_PATH = _SCRIPT_DIR / "ssh_stdio_http_mux_proxy.py"
_SPEC = importlib.util.spec_from_file_location(
    "kernelgen_ssh_stdio_http_mux_proxy",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
proxy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = proxy
_SPEC.loader.exec_module(proxy)


def _args(remote_port: int, **updates):
    values = {
        "listen_port": 0,
        "ready_fd": None,
        "mode": "direct",
        "address": "unused",
        "ssh_port": 22,
        "identity": "unused",
        "ssh_command": None,
        "remote_port": remote_port,
        "remote_python": "python3",
        "container": "-",
        "bastion": "unused",
        "jump_login": None,
        "max_streams": 8,
        "gateway_retry_attempts": 2,
        "gateway_retry_base_seconds": 0,
        "gateway_retry_max_seconds": 0,
        "ssh_ready_timeout": 2,
        "remote_open_timeout": 2,
        "server_alive_interval": 6,
        "server_alive_count_max": 30,
    }
    values.update(updates)
    return Namespace(**values)


def test_main_notifies_parent_only_after_bridge_setup(monkeypatch):
    read_fd, write_fd = os.pipe()
    closed = []
    class Bridge:
        def __init__(self, args):
            pass
        def start(self):
            # No notification before SSH setup finishes.
            import select
            assert not select.select([read_fd], [], [], 0)[0]
        def close(self):
            closed.append(True)
    monkeypatch.setattr(proxy, "_parse_args", lambda: _args(8000, ready_fd=write_fd))
    monkeypatch.setattr(proxy, "PersistentMultiplexBridge", Bridge)
    monkeypatch.setattr(proxy._ThreadingProxy, "serve_forever", lambda self: None)
    try:
        assert proxy.main() == 0
        assert os.read(read_fd, 6) == b"READY\n"
        assert os.read(read_fd, 1) == b""
        assert closed == [True]
    finally:
        os.close(read_fd)


def test_main_rejects_occupied_port_before_starting_ssh(monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        monkeypatch.setattr(proxy, "_parse_args", lambda: _args(8000, listen_port=listener.getsockname()[1]))
        monkeypatch.setattr(proxy, "PersistentMultiplexBridge", lambda args: pytest.fail("SSH must not start"))
        with pytest.raises(OSError):
            proxy.main()


class _UpstreamState:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.requests = 0


def _start_upstream(delay: float = 0.0):
    state = _UpstreamState()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            with state.lock:
                state.active += 1
                state.requests += 1
                state.max_active = max(state.max_active, state.active)
            try:
                if delay:
                    time.sleep(delay)
                body = f'{{"path":"{self.path}"}}'.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass
            finally:
                with state.lock:
                    state.active -= 1

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, state


def _local_multiplexer_command(remote_port: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-c",
        proxy.REMOTE_MULTIPLEXER,
        "127.0.0.1",
        str(remote_port),
    ]


def _start_proxy(args):
    bridge = proxy.PersistentMultiplexBridge(args)
    bridge.start()
    server = proxy._ThreadingProxy(("127.0.0.1", 0), proxy._ProxyHandler)
    server.bridge = bridge
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, bridge


def _stop_proxy(server, thread, bridge):
    server.shutdown()
    server.server_close()
    bridge.close()
    thread.join(timeout=5)


def _stop_upstream(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _http_get(port: int, path: str = "/status") -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        client.settimeout(5)
        client.sendall(
            f"GET {path} HTTP/1.1\r\n".encode()
            + b"Host: 127.0.0.1\r\n"
            + b"Connection: close\r\n\r\n"
        )
        chunks = []
        while chunk := client.recv(65536):
            chunks.append(chunk)
    return b"".join(chunks)


def test_parallel_http_connections_share_one_persistent_bridge(monkeypatch):
    upstream, upstream_thread, state = _start_upstream(delay=0.2)
    command_calls = 0
    command_lock = threading.Lock()

    def command(_args):
        nonlocal command_calls
        with command_lock:
            command_calls += 1
        return _local_multiplexer_command(upstream.server_address[1])

    monkeypatch.setattr(proxy, "_ssh_command", command)
    server, server_thread, bridge = _start_proxy(
        _args(upstream.server_address[1], max_streams=6)
    )
    responses: list[bytes] = []

    def request(index: int) -> None:
        responses.append(_http_get(server.server_address[1], f"/request-{index}"))

    clients = [threading.Thread(target=request, args=(index,)) for index in range(6)]
    try:
        for client in clients:
            client.start()
        for client in clients:
            client.join(timeout=5)
    finally:
        _stop_proxy(server, server_thread, bridge)
        _stop_upstream(upstream, upstream_thread)

    assert len(responses) == 6
    assert all(response.startswith(b"HTTP/1.1 200 OK\r\n") for response in responses)
    assert command_calls == 1
    assert state.requests == 6
    assert state.max_active >= 2


def test_bridge_reconnects_only_after_persistent_process_exits(monkeypatch):
    upstream, upstream_thread, _state = _start_upstream()
    command_calls = 0

    def command(_args):
        nonlocal command_calls
        command_calls += 1
        return _local_multiplexer_command(upstream.server_address[1])

    monkeypatch.setattr(proxy, "_ssh_command", command)
    server, server_thread, bridge = _start_proxy(_args(upstream.server_address[1]))
    try:
        first = _http_get(server.server_address[1], "/first")
        with bridge._state_lock:
            process = bridge._process
        assert process is not None
        process.terminate()
        process.wait(timeout=5)
        time.sleep(0.1)
        assert command_calls == 1
        second = _http_get(server.server_address[1], "/second")
    finally:
        _stop_proxy(server, server_thread, bridge)
        _stop_upstream(upstream, upstream_thread)

    assert first.startswith(b"HTTP/1.1 200 OK\r\n")
    assert second.startswith(b"HTTP/1.1 200 OK\r\n")
    assert command_calls == 2


def test_max_streams_queues_without_opening_another_ssh(monkeypatch):
    upstream, upstream_thread, state = _start_upstream(delay=0.2)
    command_calls = 0

    def command(_args):
        nonlocal command_calls
        command_calls += 1
        return _local_multiplexer_command(upstream.server_address[1])

    monkeypatch.setattr(proxy, "_ssh_command", command)
    server, server_thread, bridge = _start_proxy(
        _args(upstream.server_address[1], max_streams=1)
    )
    responses: list[bytes] = []
    clients = [
        threading.Thread(
            target=lambda path=path: responses.append(
                _http_get(server.server_address[1], path)
            )
        )
        for path in ("/one", "/two")
    ]
    try:
        for client in clients:
            client.start()
        for client in clients:
            client.join(timeout=5)
    finally:
        _stop_proxy(server, server_thread, bridge)
        _stop_upstream(upstream, upstream_thread)

    assert len(responses) == 2
    assert all(response.startswith(b"HTTP/1.1 200 OK\r\n") for response in responses)
    assert command_calls == 1
    assert state.max_active == 1


def test_in_flight_request_is_not_replayed_after_ssh_failure(monkeypatch):
    upstream, upstream_thread, state = _start_upstream(delay=0.8)
    command_calls = 0

    def command(_args):
        nonlocal command_calls
        command_calls += 1
        return _local_multiplexer_command(upstream.server_address[1])

    monkeypatch.setattr(proxy, "_ssh_command", command)
    server, server_thread, bridge = _start_proxy(_args(upstream.server_address[1]))
    responses: list[bytes] = []
    client = threading.Thread(
        target=lambda: responses.append(_http_get(server.server_address[1], "/slow"))
    )
    try:
        client.start()
        deadline = time.monotonic() + 3
        while state.requests == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert state.requests == 1
        with bridge._state_lock:
            process = bridge._process
        assert process is not None
        process.terminate()
        process.wait(timeout=5)
        client.join(timeout=5)
        time.sleep(0.9)
        assert command_calls == 1
        assert state.requests == 1
    finally:
        _stop_proxy(server, server_thread, bridge)
        _stop_upstream(upstream, upstream_thread)

    assert len(responses) == 1
    assert responses[0].startswith(b"HTTP/1.1 502 Bad Gateway\r\n")


def test_remote_server_connect_failure_returns_structured_502(monkeypatch):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    unavailable_port = probe.getsockname()[1]
    probe.close()
    monkeypatch.setattr(
        proxy,
        "_ssh_command",
        lambda _args: _local_multiplexer_command(unavailable_port),
    )
    server, server_thread, bridge = _start_proxy(_args(unavailable_port))
    try:
        response = _http_get(server.server_address[1])
    finally:
        _stop_proxy(server, server_thread, bridge)

    assert response.startswith(b"HTTP/1.1 502 Bad Gateway\r\n")
    assert b'"error":"SSH_MULTIPLEX_BRIDGE_FAILED"' in response


def test_jump_command_uses_operations_keepalive_settings():
    args = _args(
        18306,
        mode="jump",
        address="10.0.0.9",
        ssh_port=2224,
        identity="/tmp/id_ed25519",
        container="kernelgen-ascend",
        bastion="bastion.example.com",
        jump_login="user#root#asset-id@bastion.example.com",
    )

    command = proxy._ssh_command(args)

    assert "ServerAliveInterval=6" in command
    assert "ServerAliveCountMax=30" in command
    assert "TCPKeepAlive=yes" in command
    assert command.count("ssh") == 1
    assert "bastion.example.com" in command
    assert "sudo docker exec -i kernelgen-ascend" in command[-1]
    assert "127.0.0.1 18306" in command[-1]


def test_custom_ssh_command_uses_container_and_remote_python():
    args = _args(
        18306,
        ssh_command="ssh -p 2224 user@example.com",
        container="kernelgen-ascend",
        remote_python="/opt/runtime/bin/python3",
    )

    command = proxy._ssh_command(args)

    assert command[0] == "ssh"
    assert "-T" in command
    assert "BatchMode=yes" in command
    assert "ServerAliveInterval=6" in command
    assert "user@example.com" in command
    assert "sudo docker exec -i kernelgen-ascend" in command[-1]
    assert "/opt/runtime/bin/python3 -u -c" in command[-1]


def test_custom_ssh_command_does_not_require_inventory_connection_fields():
    args = proxy._parse_args(
        [
            "--listen-port",
            "19080",
            "--remote-port",
            "18080",
            "--ssh-command",
            "ssh user@example.com",
            "--container",
            "kgs-container",
        ]
    )

    assert args.mode is None
    assert args.identity is None
    assert args.ssh_command == "ssh user@example.com"


def test_gateway_failure_retries_before_persistent_bridge_is_returned(monkeypatch):
    attempts = 0
    sentinel = object()

    def start(_args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return None, "no available gateway"
        return sentinel, ""

    monkeypatch.setattr(proxy, "_start_ssh_bridge", start)
    process = proxy._open_ssh_bridge(_args(18306))

    assert process is sentinel
    assert attempts == 2


@pytest.mark.parametrize(
    ("header", "record"),
    [
        (
            "# name|mode|address|ssh_port|container|deploy_base|server_port|jump_login\n",
            "ascend|jump|10.0.0.9|2224|kernelgen-ascend|/workspace/deploy|18306|ssh user#root#asset-id@bastion.example.com -p 2224\n",
        ),
        (
            "# name|mode|address|ssh_port|container|deploy_base|server_port|cards|jump_login\n",
            "ascend|jump|10.0.0.9|2224|kernelgen-ascend|/workspace/deploy|18306|3,4|ssh user#root#asset-id@bastion.example.com -p 2224\n",
        ),
    ],
)
@pytest.mark.parametrize("markdown", [False, True])
def test_ascend_wrapper_uses_inventory_and_one_persistent_proxy(
    tmp_path, header, record, markdown
):
    inventory = tmp_path / ("hosts.md" if markdown else "hosts.conf")
    text = header + record
    if markdown:
        text = "# Hosts\n\n```kg-hosts\n" + text + "```\n\nDeployment notes.\n"
    inventory.write_text(text, encoding="utf-8")
    identity = tmp_path / "id_ed25519"
    identity.write_text("test-only-placeholder", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "python-args"
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" >\"$PROXY_ARG_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    script = _SCRIPT_DIR / "run_persistent_remote_http_proxy.sh"
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["PROXY_ARG_CAPTURE"] = str(capture)
    environment["MULTI_DEVICE_BASTION"] = "bastion.example.com"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--device",
            "ascend",
            "--listen-port",
            "19606",
            "--max-streams",
            "4",
            "--inventory",
            str(inventory),
            "--identity",
            str(identity),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[0].endswith("ssh_stdio_http_mux_proxy.py")
    assert arguments[arguments.index("--remote-port") + 1] == "18306"
    assert arguments[arguments.index("--container") + 1] == "kernelgen-ascend"
    assert arguments[arguments.index("--max-streams") + 1] == "4"
    assert arguments[arguments.index("--jump-login") + 1] == "user#root#asset-id"
    assert arguments[arguments.index("--server-alive-interval") + 1] == "6"
    assert arguments[arguments.index("--server-alive-count-max") + 1] == "30"
