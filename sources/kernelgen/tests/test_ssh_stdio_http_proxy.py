from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import threading
from argparse import Namespace
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = (
    _REPO_ROOT / "scripts" / "remote_server" / "ssh_stdio_http_proxy.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "kernelgen_ssh_stdio_http_proxy",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
proxy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(proxy)


def _args(**updates):
    values = {
        "listen_port": 0,
        "mode": "direct",
        "address": "unused",
        "ssh_port": 22,
        "identity": "unused",
        "remote_port": 8000,
        "container": "-",
        "bastion": "unused",
        "jump_login": None,
        "max_ssh_sessions": 2,
        "gateway_retry_attempts": 2,
        "gateway_retry_base_seconds": 0,
        "gateway_retry_max_seconds": 0,
        "ssh_ready_timeout": 2,
    }
    values.update(updates)
    return Namespace(**values)


def _serve_once(args):
    server = proxy._ThreadingProxy(
        ("127.0.0.1", 0),
        proxy._ProxyHandler,
    )
    server.args = args
    server.ssh_slots = threading.BoundedSemaphore(args.max_ssh_sessions)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _http_get(port: int) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        client.sendall(
            b"GET /status HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Connection: close\r\n\r\n"
        )
        chunks = []
        while chunk := client.recv(65536):
            chunks.append(chunk)
    return b"".join(chunks)


def test_compat_wrapper_uses_current_inventory_format(tmp_path):
    inventory = tmp_path / "hosts.conf"
    inventory.write_text(
        "# name|mode|address|ssh_port|container|deploy_base|server_port|jump_login\n"
        "ascend|jump|10.0.0.9|2224|kernelgen-ascend|/workspace/deploy|18306|ssh user#root#asset-id@bastion.example.com -p 2224\n",
        encoding="utf-8",
    )
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
    script = (
        _REPO_ROOT
        / "scripts"
        / "remote_server"
        / "run_remote_http_proxy.sh"
    )
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
    assert arguments[0].endswith("ssh_stdio_http_proxy.py")
    assert arguments[arguments.index("--remote-port") + 1] == "18306"
    assert arguments[arguments.index("--container") + 1] == "kernelgen-ascend"
    assert arguments[arguments.index("--bastion") + 1] == "bastion.example.com"
    assert arguments[arguments.index("--jump-login") + 1] == "user#root#asset-id"


def test_gateway_failure_retries_before_forwarding_request(
    tmp_path,
    monkeypatch,
):
    attempts = tmp_path / "attempts"
    bridge = tmp_path / "fake_bridge.py"
    bridge.write_text(
        """
import pathlib
import sys
import time

counter = pathlib.Path(sys.argv[1])
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
if attempt == 1:
    sys.stderr.write("no available gateway\\n")
    sys.stderr.flush()
    time.sleep(5)
    raise SystemExit(1)
sys.stdout.write("__KERNELGEN_SSH_HTTP_READY__\\n")
sys.stdout.flush()
request = b""
while b"\\r\\n\\r\\n" not in request:
    chunk = sys.stdin.buffer.read1(65536)
    if not chunk:
        raise SystemExit(2)
    request += chunk
body = b'{"status":"ok"}'
sys.stdout.buffer.write(
    b"HTTP/1.1 200 OK\\r\\n"
    + b"Content-Type: application/json\\r\\n"
    + b"Content-Length: " + str(len(body)).encode() + b"\\r\\n"
    + b"Connection: close\\r\\n\\r\\n"
    + body
)
sys.stdout.buffer.flush()
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        proxy,
        "_ssh_command",
        lambda _args: [sys.executable, str(bridge), str(attempts)],
    )
    server, thread = _serve_once(_args())
    try:
        response = _http_get(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert attempts.read_text(encoding="utf-8") == "2"
    assert response.startswith(b"HTTP/1.1 200 OK\r\n")
    assert response.endswith(b'{"status":"ok"}')


def test_exhausted_gateway_retries_return_structured_503(
    tmp_path,
    monkeypatch,
):
    bridge = tmp_path / "unavailable.py"
    bridge.write_text(
        'import sys; sys.stderr.write("no available gateway\\\\n"); '
        "raise SystemExit(1)",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        proxy,
        "_ssh_command",
        lambda _args: [sys.executable, str(bridge)],
    )
    server, thread = _serve_once(_args())
    try:
        response = _http_get(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.startswith(b"HTTP/1.1 503 Service Unavailable\r\n")
    assert b"X-KernelGen-SSH-Gateway: unavailable" in response
    assert b'"error":"SSH_GATEWAY_UNAVAILABLE"' in response


def test_max_ssh_sessions_queues_excess_clients(
    tmp_path,
    monkeypatch,
):
    starts = tmp_path / "starts"
    bridge = tmp_path / "slow_bridge.py"
    bridge.write_text(
        """
import pathlib
import sys
import time

with pathlib.Path(sys.argv[1]).open("a") as handle:
    handle.write(f"{time.monotonic()}\\n")
sys.stdout.write("__KERNELGEN_SSH_HTTP_READY__\\n")
sys.stdout.flush()
request = b""
while b"\\r\\n\\r\\n" not in request:
    chunk = sys.stdin.buffer.read1(65536)
    if not chunk:
        raise SystemExit(2)
    request += chunk
time.sleep(0.4)
body = b'{"status":"ok"}'
sys.stdout.buffer.write(
    b"HTTP/1.1 200 OK\\r\\n"
    + b"Content-Length: " + str(len(body)).encode() + b"\\r\\n"
    + b"Connection: close\\r\\n\\r\\n"
    + body
)
sys.stdout.buffer.flush()
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        proxy,
        "_ssh_command",
        lambda _args: [sys.executable, str(bridge), str(starts)],
    )
    server, server_thread = _serve_once(_args(max_ssh_sessions=1))
    responses = []

    def request():
        responses.append(_http_get(server.server_address[1]))

    clients = [threading.Thread(target=request) for _ in range(2)]
    try:
        for client_thread in clients:
            client_thread.start()
        for client_thread in clients:
            client_thread.join(timeout=5)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    started_at = [
        float(value)
        for value in starts.read_text(encoding="utf-8").splitlines()
    ]
    assert len(responses) == 2
    assert all(response.startswith(b"HTTP/1.1 200 OK\r\n") for response in responses)
    assert len(started_at) == 2
    assert started_at[1] - started_at[0] >= 0.35


def test_remote_connector_uses_unbuffered_stdin_read():
    assert "import os" in proxy.REMOTE_CONNECTOR
    assert "os.read(sys.stdin.fileno(), 65536)" in proxy.REMOTE_CONNECTOR
    assert "sys.stdin.buffer.read1" not in proxy.REMOTE_CONNECTOR
