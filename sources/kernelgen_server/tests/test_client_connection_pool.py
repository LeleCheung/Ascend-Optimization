from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kernelgen_server.protocol import client


class _KeepAliveServer(ThreadingHTTPServer):
    connection_count = 0

    def get_request(self):
        request, address = super().get_request()
        self.connection_count += 1
        return request, address


class _StatusHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        assert self.path == "/status"
        payload = json.dumps({"status": "ok"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def test_status_reuses_one_keep_alive_connection(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    session = client._build_session()
    monkeypatch.setattr(client, "_SESSION", session)
    server = _KeepAliveServer(("127.0.0.1", 0), _StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        assert client.status(server_url) == {"status": "ok"}
        assert client.status(server_url) == {"status": "ok"}
        assert server.connection_count == 1
        assert session.get_adapter(server_url).max_retries.total == 0
    finally:
        session.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
