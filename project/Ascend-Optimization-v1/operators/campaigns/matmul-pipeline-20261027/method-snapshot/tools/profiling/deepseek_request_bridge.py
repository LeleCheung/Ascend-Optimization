#!/usr/bin/env python3
"""给CC的DeepSeek请求显式设置thinking disabled，凭据只在内存转发。"""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class DeepSeekRequestBridge:
    def __init__(self, ledger: Path, *, upstream='https://api.deepseek.com/anthropic', timeout=1200):
        self.ledger = ledger
        self.upstream = upstream.rstrip('/')
        self.timeout = timeout
        self.requests = []
        self.lock = threading.Lock()

    def record(self):
        with self.lock:
            self.ledger.write_text(json.dumps({'upstream': self.upstream, 'requests': self.requests},
                                             ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def __enter__(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def do_POST(self):
                started = time.monotonic()
                route = urllib.parse.urlsplit(self.path)
                if route.scheme or route.netloc or not route.path.startswith('/'):
                    self.send_error(400)
                    return
                body = self.rfile.read(int(self.headers['Content-Length']))
                value = json.loads(body)
                entry = {'route': self.path, 'model': value.get('model'),
                         'max_tokens': value.get('max_tokens'),
                         'previous_thinking_type': (value.get('thinking') or {}).get('type')}
                # Beta SDK附带?beta=true；必须按URL的path识别，不能直接比较原字符串。
                if route.path.rstrip('/') == '/v1/messages':
                    value['thinking'] = {'type': 'disabled'}
                    entry['forced_thinking_type'] = 'disabled'
                outgoing = json.dumps(value).encode()
                entry['body_sha256'] = hashlib.sha256(outgoing).hexdigest()
                bridge.requests.append(entry)
                bridge.record()
                headers = {key: val for key, val in self.headers.items()
                           if key.lower() not in {'host', 'content-length', 'connection', 'transfer-encoding'}}
                request = urllib.request.Request(bridge.upstream + self.path, data=outgoing, headers=headers)
                try:
                    try:
                        response = urllib.request.urlopen(request, timeout=bridge.timeout)
                    except urllib.error.HTTPError as error:
                        response = error
                    with response:
                        entry['http_status'] = response.status
                        self.send_response(response.status)
                        for key, val in response.headers.items():
                            if key.lower() not in {'connection', 'transfer-encoding', 'content-length'}:
                                self.send_header(key, val)
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        self.close_connection = True
                        while chunk := response.read1(16384):
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except Exception as error:
                    entry['error_type'] = type(error).__name__
                    self.close_connection = True
                finally:
                    entry['seconds'] = time.monotonic() - started
                    bridge.record()

        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.base_url = f'http://127.0.0.1:{self.server.server_port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.record()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
