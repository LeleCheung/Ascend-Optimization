"""Public runtime works in a distribution with no development directory."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tarfile
import unittest

ROOT=Path(__file__).resolve().parents[3]

class PublicDistributionTests(unittest.TestCase):
    def test_prepare_context_and_actions_without_internal_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ['src','skills']:
                shutil.copytree(ROOT/'hgkda'/name,root/'hgkda'/name,ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(ROOT/'operators/log10_',root/'operators/log10_')
            env=dict(os.environ,PYTHONPATH=str(root/'hgkda/src'),PYTHONDONTWRITEBYTECODE='1')
            config=root/'environment.json'
            config.write_text(json.dumps(dict(container='explicit',device=7,device_python='/python',sdk='/sdk',
                                              lock=str(root/'device.lock'),kgs_server='http://127.0.0.1:1')))
            def invoke(*args):
                p=subprocess.run([sys.executable,'-X','utf8',*args],cwd=root,env=env,capture_output=True,text=True,encoding='utf8')
                self.assertEqual(p.returncode,0,p.stderr);return p.stdout
            invoke('-m','hgkda','doctor')
            for action in ['prepare','context','generate','preflight','pair','simulate','evaluate']:
                invoke('-m','hgkda.runtime',action,'--help')
            run=root/'run'
            invoke('-m','hgkda.runtime','prepare','--operator-directory',str(root/'operators/log10_'),
                   '--package',str(root/'hgkda'),'--environment',str(config),'--run-directory',str(run))
            invoke('-m','hgkda.runtime','context','--config',str(run/'config.json'))
            with tarfile.open(run/'operator.tar') as archive:
                self.assertNotIn('relocation.json',archive.getnames())
                self.assertIn('definition.json',archive.getnames())
            brief=json.loads((run/'workspace/context-brief.json').read_bytes())
            self.assertEqual(brief['operator'],'log10_')
            self.assertEqual(brief['contract']['formal_cases'],36)
            self.assertIsNone(brief['paired_evidence'])
            source=(run/'workspace/candidate.py').read_bytes()
            self.assertEqual(hashlib.sha256(source).hexdigest(),brief['source_sha256'])
            self.assertFalse((root/'development').exists())
            self.assertEqual(json.loads((run/'config.json').read_bytes())['special_checks']['edges']['previous_total'],15)

    def test_selected_versions_keep_exact_candidate_bytes(self):
        index=json.loads((ROOT/'operators/index.json').read_bytes())
        self.assertEqual(sum(r['accepted'] for r in index['rows']),26)
        for row in index['rows']:
            m=json.loads((ROOT/'operators'/row['manifest']).read_bytes())
            self.assertEqual(set(m['versions']),{'historical_noncompliant','historical_strict','independent_strict'})
            if m['selected']:
                self.assertEqual(hashlib.sha256((ROOT/m['selected']['path']).read_bytes()).hexdigest(),m['selected']['sha256'])
            if row['accepted']:
                for spec in m['checks'].values():
                    self.assertTrue((ROOT/'operators'/row['operator']/spec['checker']).is_file())
                    self.assertTrue((ROOT/'operators'/row['operator']/spec['reference']/'oracle.py').is_file())
