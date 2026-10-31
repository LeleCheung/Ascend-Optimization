"""恢复时不重发未知请求，不把历史候选标为FlagGems。"""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('fcntl'), '控制器运行于Linux')
class RecoveryTests(unittest.TestCase):
    def test_unchecked_duplicate_cannot_displace_qualified_lower_score(self):
        import run
        gate=self.r.batch/'qualified.json'
        gate.write_text(json.dumps(dict(operator='x',status='passed',source_sha256='verified')))
        base=dict(status='PASSED',num_passed=1,num_workloads=1,latency_ms={'case':1.})
        untested=dict(base,geo_mean=1.1,source_sha256='unchecked')
        verified=dict(base,geo_mean=.7,source_sha256='verified',extra_checks=str(gate))
        row=dict(best=untested)
        run.Runner.consider_best(row,verified)
        self.assertEqual(row['best'],verified)
        run.Runner.consider_best(row,untested)
        self.assertEqual(row['best'],verified)

    def test_stale_faster_supplement_cannot_replace_current_qualified_best(self):
        import run
        old_gate=self.r.batch/'old-extra.json';new_gate=self.r.batch/'new-extra.json'
        old_gate.write_text(json.dumps(dict(operator='log10_',status='passed',source_sha256='old',cases=[])))
        new_gate.write_text(json.dumps(dict(operator='log10_',status='passed',source_sha256='new',cases=[
            dict(case='additional-torch.'+dtype+'-alignment-sequence',passed=True,offset_sequence=[0,1,3,0],elements=1024)
            for dtype in ['float16','bfloat16','float32']])))
        base=dict(status='PASSED',num_passed=1,num_workloads=1,latency_ms={'case':1.})
        stale=dict(base,geo_mean=2.,source_sha256='old',extra_checks=str(old_gate))
        fresh=dict(base,geo_mean=.9,source_sha256='new',extra_checks=str(new_gate))
        row=dict(best=stale)
        run.Runner.consider_best(row,fresh)
        self.assertEqual(row['best'],fresh)
        run.Runner.consider_best(row,stale)
        self.assertEqual(row['best'],fresh)
        self.assertEqual(len(row['retained_unqualified']),1)

    def test_catalog_retry_requires_finished_local_rejection_and_no_device_stage(self):
        workspace=self.r.q.runtime/'kg-runs'/(str('x-')+self.r.batch.name+'-attempt-1')
        metadata=workspace/'.kernelgen';metadata.mkdir(parents=True)
        (metadata/'run-process.json').write_text(json.dumps(dict(state='EXITED',finished_at='now')))
        progress=metadata/'run-progress.json'
        progress.write_text(json.dumps(dict(state='INFRASTRUCTURE_ERROR',message="unexpected files in ops/x: ['__pycache__']")))
        self.assertTrue(self.r.local_catalog_cache_failure('x'))
        (metadata/'active-server-operations.json').write_text(json.dumps(dict(operations={'op':{'kind':'evaluate'}})))
        self.assertFalse(self.r.local_catalog_cache_failure('x'))
        (metadata/'active-server-operations.json').write_text(json.dumps(dict(operations={})))
        (workspace/'stages/optimize').mkdir(parents=True)
        self.assertFalse(self.r.local_catalog_cache_failure('x'))

    def test_selection_matches_relative_speedup_acceptance(self):
        import run
        parent=dict(status='PASSED',num_passed=1,num_workloads=1,geo_mean=.81,latency_ms={'case':1.})
        candidate=dict(parent,geo_mean=.75,latency_ms={'case':.95})
        row=dict(best=parent)
        run.Runner.consider_best(row,candidate)
        self.assertIs(row['best'],parent)

    def test_evaluation_grace_is_bounded_and_owned(self):
        from native_search import evaluation_grace
        server='http://127.0.0.1:19659'
        own=dict(kind='evaluate',server_url=server,operation_id='own')
        operations=dict(own=own,other=dict(own,server_url='http://other'),
            profile=dict(own,kind='profile'),missing_id=dict(kind='evaluate',server_url=server))
        self.assertEqual(evaluation_grace(operations,server,1199,1200),[])
        self.assertEqual(evaluation_grace(operations,server,1210,1200),[own])
        self.assertEqual(evaluation_grace(operations,server,1380,1200),[])

    def test_supplementary_keeps_checker_failure_and_source_identity(self):
        source=self.r.batch/'candidate.py';source.write_text('def run(x): return x')
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        gates=self.r.batch/'extra-checks.json'
        self.r.q.records['x']['catalog_leaf']='catalog/ops/x'
        def failing_check(*args):
            gates.write_text(json.dumps(dict(operator='x',source_sha256=sha,status='failed')))
            raise RuntimeError('checker exited nonzero')
        self.r.q.run=failing_check
        self.assertEqual(self.r.supplementary('x',source,gates,'extra-x'),'failed')
        self.assertEqual(json.loads(gates.read_text())['source_sha256'],sha)

    def test_supplementary_does_not_mask_missing_output(self):
        source=self.r.batch/'candidate.py';source.write_text('def run(x): return x')
        self.r.q.records['x']['catalog_leaf']='catalog/ops/x'
        def unavailable(*args):raise RuntimeError('container unavailable')
        self.r.q.run=unavailable
        with self.assertRaisesRegex(RuntimeError,'container unavailable'):
            self.r.supplementary('x',source,self.r.batch/'extra-checks.json','extra-x')

    def test_supplementary_rejects_output_for_different_source(self):
        source=self.r.batch/'candidate.py';source.write_text('def run(x): return x')
        gates=self.r.batch/'extra-checks.json'
        self.r.q.records['x']['catalog_leaf']='catalog/ops/x'
        def stale_output(*args):
            gates.write_text(json.dumps(dict(operator='x',source_sha256='different',status='failed')))
            raise RuntimeError('wrong checker output')
        self.r.q.run=stale_output
        with self.assertRaisesRegex(RuntimeError,'wrong checker output'):
            self.r.supplementary('x',source,gates,'extra-x')

    def setUp(self):
        import run
        self.r=run.Runner.__new__(run.Runner)
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.r.batch=root/'batch';self.r.batch.mkdir()
        self.r.q=SimpleNamespace(records={'x':dict(source_sha256='original')},runtime=root/'runtime')
        self.r.module=SimpleNamespace(now=lambda:'now')
        self.r.state=dict(jobs={},operators={})

    def test_correct_bootstrap_attribution_without_changing_result(self):
        raw=dict(source_sha256='generated',source='generated.py',result='FlagGems-1.result.json',status='PASSED')
        self.r.state['jobs']['baseline:x']=dict(status='completed',result=raw)
        self.r.state['operators']['x']=dict(baseline=raw)
        self.r.recover_local_preparation()
        self.assertNotIn('baseline:x',self.r.state['jobs'])
        corrected=self.r.state['jobs']['historical-start:x']['result']
        self.assertTrue(corrected['legacy_label_corrected'])
        self.assertEqual(corrected['result'],raw['result'])
        self.assertNotIn('legacy_label_corrected',raw)

    def test_retry_only_local_failure_without_spec_or_worker(self):
        self.r.state['jobs']['search:x']=dict(status='failed',error='FileNotFoundError: missing spec.tmp')
        self.r.recover_local_preparation()
        self.assertNotIn('search:x',self.r.state['jobs'])
        self.assertEqual(len(self.r.state['recovery_events']),1)

    def test_keep_existing_search_and_submitted_identity(self):
        path=self.r.batch/'analysis/x/spec.json';path.parent.mkdir(parents=True);path.write_text('{}')
        self.r.state['jobs']['search:x']=dict(status='failed',error='FileNotFoundError: later file')
        self.r.recover_local_preparation()
        self.assertIn('search:x',self.r.state['jobs'])
        self.r.state['jobs']['baseline:x']=dict(status='failed',error='RuntimeError: request result unknown')
        self.r.recover_local_preparation()
        self.assertIn('baseline:x',self.r.state['jobs'])

    def test_debug_requires_formal_feedback_after_initial_probes(self):
        from debug_budget import admit
        workspace=Path(self.tmp.name)/'work'
        first=admit(workspace,900)
        self.assertEqual(first['timeout_seconds'],120)
        self.assertEqual(admit(workspace,60)['status'],'ADMITTED')
        self.assertEqual(admit(workspace,60)['reason'],'HGKDA_DEBUG_BUDGET')
        result=workspace/'.kernelgen/evals/round-001/result.json';result.parent.mkdir(parents=True);result.write_text('{}')
        self.assertEqual(admit(workspace,60)['status'],'ADMITTED')

    def test_retry_uses_verified_best_source(self):
        source=Path(self.tmp.name)/'best.py';source.write_text('def run(x): return x')
        self.r.state['operators']['x']=dict(best=dict(status='PASSED',source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        self.assertEqual(self.r.search_source('x',2),source)
        source.write_text('changed')
        self.r.module.Attention=RuntimeError
        with self.assertRaises(RuntimeError):self.r.search_source('x',2)

    def test_collect_unknown_baselines_before_long_searches(self):
        import time
        self.r.q.runtime.mkdir()
        self.r.q.plan=dict(priority=['a','b'],duration_hours=14)
        self.r.q.records={}
        for op in ['a','b']:
            source=self.r.batch/(op+'.py');source.write_text('def run(x): return x')
            self.r.q.records[op]=dict(seed=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        (self.r.batch/'baseline-verified.json').write_text('{"status":"verified"}')
        self.r.state['active_started_epoch']=time.time()
        self.r.save=lambda:None
        calls=[]
        self.r.evaluate=lambda *args:dict(status='PARTIAL_PASS')
        def job(key,fn):
            calls.append(key);result=fn()
            self.r.state['jobs'][key]=dict(status='completed',result=result)
            if key=='baseline:b':(self.r.q.runtime/'STOP-AFTER-JOB').touch()
            return result
        self.r.job=job
        self.r.execute()
        self.assertEqual(calls,['baseline:a','baseline:b'])
        self.assertEqual(self.r.state['status'],'budget_boundary')

    def test_first_search_uses_qualified_proposal_but_not_stale_supplement(self):
        source=Path(self.tmp.name)/'proposal.py';source.write_text('def run(x): return x')
        sha=hashlib.sha256(source.read_bytes()).hexdigest();gates=Path(self.tmp.name)/'checks.json'
        gates.write_text(json.dumps(dict(status='passed',source_sha256=sha)))
        self.r.state['operators']['x']=dict(best=dict(status='PASSED',source=str(source),source_sha256=sha,extra_checks=str(gates)))
        self.r.starting_source=lambda op:Path('original.py')
        self.assertEqual(self.r.search_source('x',1),source)
        gates.write_text(json.dumps(dict(status='failed',source_sha256=sha)))
        self.assertEqual(self.r.search_source('x',1),Path('original.py'))

    def test_queued_proposal_identity_checked_before_submission(self):
        from proposals import process
        self.r.q.plan=dict(duration_hours=14)
        folder=self.r.batch/'proposals/x/check';folder.mkdir(parents=True)
        (folder/'main.py').write_text('def run(x): return x')
        (folder/'proposal.json').write_text(json.dumps(dict(operator='x',id='check',source='main.py',source_sha256='wrong')))
        self.r.module.Attention=RuntimeError
        self.r.job=lambda *args:self.fail('源码身份不匹配时不得提交评测')
        with self.assertRaises(RuntimeError):process(self.r)


if __name__=='__main__':unittest.main()
