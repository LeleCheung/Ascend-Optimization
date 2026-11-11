"""上游验收必须包含实际用例、有效计时和正确源码身份。"""
import hashlib
import sys
import types
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from hgkda.adapters.flaggems import summarize
from hgkda.runtime.upstream_test import snapshot, save, read
from hgkda.runtime import upstream_probe


def fixture():
    return {'result':{'log10_':{
        'accuracy':{'status':'Passed','exit_code':0,'total':3,'passed':2,'skipped':1,'failed':0},
        'performance':{'status':'Passed','exit_code':0,'data':{'float32':{'details':{
            '(1024,)':{'base':2.,'gems':1.,'speedup':2.}}}}}}}}


class FlagGemsAdapterTests(unittest.TestCase):
    def test_import_probe_detects_conflicting_kgs_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'op.py';source.write_text('# fixture')
            def function():pass
            function.__module__='fixture_op'
            module=types.ModuleType('fixture_op');module.log10_=function
            gems=types.ModuleType('flag_gems');gems.__file__=str(root/'__init__.py')
            gems.log10_=function;gems.vendor_name='ascend';gems.device='npu'
            for shared in [False,True]:
                module.run=function if shared else lambda:None
                output=root/(str(shared)+'.json')
                with patch.dict(sys.modules,{'flag_gems':gems,'fixture_op':module}), \
                     patch.object(sys,'argv',['probe','--repository',str(root),'--operator','log10_','--output',str(output)]), \
                     patch.object(upstream_probe.inspect,'getsourcefile',return_value=str(source)):
                    upstream_probe.main()
                self.assertEqual(read(output)['kgs_same_entrypoint'],shared)

    def test_records_skips_and_does_not_claim_pr_ready(self):
        r=summarize(fixture(),'log10_',0)
        self.assertEqual(r['status'],'passed');self.assertEqual(r['skipped_accuracy_cases'],1)
        self.assertEqual(r['geometric_speedup'],2.)
        self.assertFalse(r['pr_ready'])

    def test_zero_exit_and_missing_tests_is_not_success(self):
        self.assertEqual(summarize({},'log10_',0)['status'],'failed')
        r=fixture();r['result']['log10_']['accuracy']['passed']=0
        self.assertEqual(summarize(r,'log10_',0)['status'],'failed')

    def test_collection_error_or_empty_benchmark_fails(self):
        for kind in ['accuracy','performance']:
            r=fixture();r['result']['log10_'][kind]['exit_code']=1
            self.assertEqual(summarize(r,'log10_',0)['status'],'failed')
        r=fixture();r['result']['log10_']['performance']['data']={}
        self.assertEqual(summarize(r,'log10_',0)['status'],'failed')

    def test_invalid_measurement_is_not_a_speedup(self):
        for value in [float('nan'),float('inf'),0.,-1.]:
            r=fixture();r['result']['log10_']['performance']['data']['float32']['details']['(1024,)']['gems']=value
            self.assertEqual(summarize(r,'log10_',0)['status'],'failed')

    def test_repository_and_test_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def git(*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL).decode().strip()
            git('init');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
            for name in ['tools/run_tests.py','src/flag_gems/op.py','tests/test_op.py']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
            git('add','.');git('commit','-m','fixture');head=git('rev-parse','HEAD')
            source=root/'src/flag_gems/op.py';source.write_text('# integration fixture\n')
            allowed={'src/flag_gems/op.py':hashlib.sha256(source.read_bytes()).hexdigest()}
            self.assertEqual(snapshot(root,head,allowed)['changed'],list(allowed))
            (root/'tests/test_op.py').write_text('# weakened test fixture\n')
            with self.assertRaises(AssertionError):snapshot(root,head,allowed)

    def test_json_is_complete_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'result.json';save(path,{'passed':True})
            self.assertEqual(read(path),{'passed':True})
            with self.assertRaises(FileExistsError):save(path,{})
