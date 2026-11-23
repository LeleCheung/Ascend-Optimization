"""Reject unexpected binary changes in fixed-source IR campaigns."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from development.tools.research.ir_binary_compare import compare


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf8')


class IRBinaryComparisonTest(unittest.TestCase):
    def fixture(self, run):
        save(run / 'config.json', dict(source_sha256='source'))
        for variant, latency in [('unchanged-ir', 2), ('candidate', 1)]:
            folder = run / 'evaluations' / variant
            rows = []
            for ast in ['edited', 'untouched']:
                data = (ast + (variant if ast == 'edited' else '')).encode()
                path = folder / (ast + '.npubin')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                rows.append(dict(source_sha256='source', ast_sha256=ast, artifacts=[
                    dict(kind='npubin', path=str(path), sha256=hashlib.sha256(data).hexdigest())]))
            receipt = folder / 'compiler-receipts/one.jsonl'
            receipt.parent.mkdir()
            receipt.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf8')
            save(folder / 'result.json', dict(status='PASSED', num_passed=1, num_workloads=1,
                per_workload=[dict(uuid='target', phase='timing', status='PASSED',
                    latency_ms=latency, reference_latency_ms=3)]))
        save(run / 'evaluations/candidate/implementation/ir-manifest.json', dict(overrides=[
            dict(ast_sha256='edited', original_sha256='old-ir', sha256='new-ir')]))

    def test_exact_untouched_identity_and_latency_ratio(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.fixture(run)
            result = compare(run, 'candidate')
            self.assertEqual((result['changed_ast_count'], result['unchanged_ast_count']), (1, 1))
            self.assertEqual(result['full_contract_candidate_latency_geomean_ratio'], 2)

    def test_tampered_binary_and_unregistered_edit_are_rejected(self):
        for tamper in [True, False]:
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                self.fixture(run)
                binary = run / 'evaluations/candidate/untouched.npubin'
                binary.write_bytes(b'unexpected change')
                if not tamper:
                    receipt = run / 'evaluations/candidate/compiler-receipts/one.jsonl'
                    rows = [json.loads(line) for line in receipt.read_text().splitlines()]
                    rows[1]['artifacts'][0]['sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
                    receipt.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf8')
                with self.assertRaisesRegex(AssertionError, 'binary bytes changed' if tamper else 'unedited specialization'):
                    compare(run, 'candidate')


if __name__ == '__main__':
    unittest.main()
