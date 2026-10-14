"""用真实模型探针响应验证源码来源，并拒绝篡改及供应商继续思考。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tools/profiling'))
spec = importlib.util.spec_from_file_location('generation_verifier', PROJECT / 'tools/profiling/verify-agent-campaign.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
PROBE = PROJECT / 'operators/logaddexp2/reports/model-interface-probes-20261014-four/payload-disabled-source-v2'


class GenerationEvidenceTests(unittest.TestCase):
    def prepare(self, root):
        raw = (PROBE / 'response.jsonl').read_text(encoding='utf-8')
        summary = json.loads((PROBE / 'summary.json').read_text(encoding='utf-8'))
        (root / 'claude-response.json').write_text(raw, encoding='utf-8')
        requests = [{**row, 'forced_thinking_type': row['forced_thinking']['type']}
                    for row in summary['requests']]
        (root / 'provider-request-1.json').write_text(json.dumps({'requests': requests}), encoding='utf-8')
        return raw, summary['source_sha256']

    def test_actual_complete_response_binds_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sha = self.prepare(root)
            self.assertEqual(verifier.verify_generation(root, sha), 1)
            with self.assertRaisesRegex(ValueError, '实际CC生成'):
                verifier.verify_generation(root, '0' * 64)

    def test_disabled_mode_with_thinking_or_missing_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, sha = self.prepare(root)
            event = {'type': 'assistant', 'message': {'content': [{'type': 'thinking', 'thinking': 'test only'}]}}
            (root / 'claude-response.json').write_text(raw + '\n' + json.dumps(event) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '仍输出思考'):
                verifier.verify_generation(root, sha)
            (root / 'claude-response.json').write_text(raw, encoding='utf-8')
            (root / 'provider-request-1.json').write_text('{"requests":[]}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '成功请求'):
                verifier.verify_generation(root, sha)


if __name__ == '__main__':
    unittest.main()
