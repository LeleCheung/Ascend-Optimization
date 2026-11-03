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

ROOT=Path(__file__).resolve().parents[1]


class AdapterTests(unittest.TestCase):
    def test_catalog_matches_original_bundle(self):
        example=ROOT/'examples/kgs-rad2deg'
        raw,digest=pack(example/'catalog')
        request=json.loads((example/'request.json').read_bytes())
        self.assertEqual(request['binding']['bundle_id'],'sha256:'+digest)
        self.assertEqual(raw,(example/'operator.tar').read_bytes())

    def test_alias_torch_compute_rejected(self):
        result=inspect_text('import torch as t\ndef run(x):\n return t.exp(x)\n')
        self.assertFalse(result['static_passed'])
        self.assertEqual(result['admission'],'review_required')

    def test_frozen_review_and_request_identity(self):
        example=ROOT/'examples/kgs-rad2deg'
        request=json.loads((example/'request.json').read_bytes())
        review=json.loads((example/'review.json').read_bytes())
        source=request['implementation']['sources'][0]['content']
        result=admit(source,review)
        self.assertEqual(result['admission'],'review_bound')
        with self.assertRaises(ValueError):admit(source+'\n',review)

    def test_no_retry_after_transport_failure(self):
        example=ROOT/'examples/kgs-rad2deg'
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
