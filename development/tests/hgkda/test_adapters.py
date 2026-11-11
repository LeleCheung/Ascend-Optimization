"""验证冻结合同、候选审查和后端请求边界。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hgkda.adapters.bundle import pack
from hgkda.adapters.kgs import evaluate
from hgkda.gate import admit, inspect_text
from hgkda.cli import dispatch, parser

ROOT=Path(__file__).resolve().parents[3]


class AdapterTests(unittest.TestCase):
    def test_catalog_matches_original_bundle(self):
        example=ROOT/'hgkda/examples/kgs-rad2deg'
        raw,digest=pack(example/'catalog')
        request=json.loads((example/'request.json').read_bytes())
        self.assertEqual(request['binding']['bundle_id'],'sha256:'+digest)
        self.assertEqual(raw,(example/'operator.tar').read_bytes())

    def test_alias_torch_compute_rejected(self):
        result=inspect_text('import torch as t\ndef run(x):\n return t.exp(x)\n')
        self.assertFalse(result['static_passed'])
        self.assertEqual(result['admission'],'review_required')

    def test_metadata_and_storage_views_are_not_device_computation(self):
        source = ('import torch as t\ndef run(x):\n'
                  ' shape = t.broadcast_shapes(x.shape, x.shape)\n'
                  ' props = t.npu.get_device_properties(x.device)\n'
                  ' count = t.npu.device_count()\n'
                  ' alias = t._C._is_alias_of(x, x)\n'
                  ' return t.view_as_real(x)\n')
        self.assertTrue(inspect_text(source)['static_passed'])
        self.assertFalse(inspect_text(source.replace('t.view_as_real(x)', 't.real(x).clone()'))['static_passed'])
        self.assertFalse(inspect_text(source.replace('t.npu.device_count()', 't.npu.synchronize()'))['static_passed'])

    def test_profiler_state_is_metadata_but_control_is_not_allowed(self):
        source = 'import torch as t\ndef run(x):\n return t.autograd._profiler_enabled()\n'
        self.assertTrue(inspect_text(source)['static_passed'])
        for name in ['_disable_profiler', '_enable_profiler', 'profiler.profile']:
            self.assertFalse(inspect_text(source.replace('_profiler_enabled', name))['static_passed'])

    def test_gate_preserves_crlf_source_identity(self):
        source = b'def run(x):\r\n    return x\r\n'
        review = dict(source_sha256=hashlib.sha256(source).hexdigest(),
                      status='reviewed_strict_triton', all_branches_reviewed=True,
                      dependencies_reviewed=True)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'candidate.py'
            path.write_bytes(source)
            record = Path(temp) / 'review.json'
            record.write_text(json.dumps(review), encoding='utf8')
            result, _ = dispatch(parser().parse_args(['gate', '--source', str(path), '--review', str(record)]))
            self.assertEqual(result['source_sha256'], review['source_sha256'])
            with self.assertRaises(ValueError):
                admit(source.decode().replace('\r\n', '\n'), review)

    def test_frozen_review_and_request_identity(self):
        example=ROOT/'hgkda/examples/kgs-rad2deg'
        request=json.loads((example/'request.json').read_bytes())
        review=json.loads((example/'review.json').read_bytes())
        source=request['implementation']['sources'][0]['content']
        result=admit(source,review)
        self.assertEqual(result['admission'],'review_bound')
        with self.assertRaises(ValueError):admit(source+'\n',review)

    def test_no_retry_after_transport_failure(self):
        example=ROOT/'hgkda/examples/kgs-rad2deg'
        with tempfile.TemporaryDirectory() as temp, patch('hgkda.adapters.kgs.call') as call:
            call.side_effect=[{'status':'ok','api_version':'v6.2'},TimeoutError('uncertain')]
            output=Path(temp)/'new'
            with self.assertRaises(TimeoutError):
                evaluate('http://localhost:1',example/'request.json',example/'review.json',output)
            self.assertEqual(call.call_count,2)
            self.assertTrue((output/'request.json').is_file())
            self.assertTrue((output/'operation.json').is_file())
            self.assertEqual(json.loads((output/'transport-error.json').read_bytes())['state'],'unknown')


if __name__=='__main__':unittest.main()
