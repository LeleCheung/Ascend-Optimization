"""模型上下文不能把历史宿主诊断当成新种子的实测。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from context import host_profile_context


class HostContextTests(unittest.TestCase):
    def test_source_identity_and_seed_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'main.py';source.write_text('def run(x): return x')
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            path=root/'host-profile-test.json'
            path.write_text(json.dumps(dict(status='completed',source=str(source),source_sha256=sha,calls=100,
                top_self_time=[dict(function='factory',calls=300,self_seconds=.001)],scope='diagnostic only')))
            self.assertTrue(host_profile_context(root,sha)[0]['matches_seed'])
            row=host_profile_context(root,'other')[0]
            self.assertFalse(row['matches_seed'])
            self.assertEqual(row['top_self_time'][0]['microseconds_per_profiled_call'],10.)
            source.write_text('changed')
            self.assertEqual(host_profile_context(root,sha),[])


if __name__=='__main__':unittest.main()
