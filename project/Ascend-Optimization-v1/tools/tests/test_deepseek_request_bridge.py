"""验证真实HTTP转发、Beta路径、SSE字节及凭据不进入审计文件。"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.request

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tools/profiling'))
from deepseek_request_bridge import DeepSeekRequestBridge


class RequestBridgeTests(unittest.TestCase):
    def test_beta_messages_force_disabled_and_preserve_other_fields_and_stream(self):
        observed = []
        payload = {'model': 'deepseek-flash', 'max_tokens': 16384,
                   'messages': [{'role': 'user', 'content': 'generate source'}], 'stream': True}
        stream = b'data: {"type":"message_stop"}\n\n'

        class Origin(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                observed.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length']))),
                                 self.headers.get('x-api-key')))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(stream)))
                self.end_headers()
                self.wfile.write(stream)

        origin = ThreadingHTTPServer(('127.0.0.1', 0), Origin)
        thread = threading.Thread(target=origin.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                ledger = Path(directory) / 'ledger.json'
                with DeepSeekRequestBridge(ledger, upstream=f'http://127.0.0.1:{origin.server_port}') as bridge:
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    for route in ('/v1/messages?beta=true', '/v1/messages/count_tokens?beta=true'):
                        req = urllib.request.Request(bridge.base_url + route, data=json.dumps(payload).encode(),
                                                     headers={'Content-Type': 'application/json', 'x-api-key': 'unit-test-only'})
                        with opener.open(req, timeout=10) as response:
                            self.assertEqual(response.read(), stream)
                self.assertEqual(observed[0][0], '/v1/messages?beta=true')
                self.assertEqual(observed[0][1], {**payload, 'thinking': {'type': 'disabled'}})
                self.assertEqual(observed[1][1], payload)
                self.assertTrue(all(row[2] == 'unit-test-only' for row in observed))
                text = ledger.read_text(encoding='utf-8')
                self.assertNotIn('unit-test-only', text)
                self.assertNotIn('generate source', text)
                rows = json.loads(text)['requests']
                self.assertEqual(rows[0]['forced_thinking_type'], 'disabled')
                self.assertEqual(rows[0]['http_status'], 200)
                self.assertNotIn('forced_thinking_type', rows[1])
        finally:
            origin.shutdown()
            origin.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
