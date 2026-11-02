"""已发现控制流错误的高分候选不能阻挡修复版进入交付。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measurements
import proposals

spec = importlib.util.spec_from_file_location('night_selection_controller', HERE/'run.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class SelectionTests(unittest.TestCase):
    def test_rejected_manual_proposal_does_not_submit_device_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'proposals/op/known-bad';out.mkdir(parents=True)
            source=out/'main.py';source.write_text('def run(x):\n    return x\n')
            controller.save(out/'spec.json',dict(operator='op',source_sha256=controller.sha(source)))
            runner=SimpleNamespace(batch=root,q=SimpleNamespace(records={'op':{}}),
                state=dict(jobs={},operators={}),persist=mock.Mock(),
                manual_rejection=mock.Mock(return_value='verified-review-rejection.json'),
                wait=mock.Mock(side_effect=AssertionError('不得使用设备')),
                cmd=mock.Mock(side_effect=AssertionError('不得提交请求')))
            with mock.patch('probe_foreach_reference.process'):
                proposals.process(runner)
            result=runner.state['jobs']['proposal:op:known-bad']['result']
            self.assertEqual(result['status'],'manual_review_rejected_no_measurement')
            runner.wait.assert_not_called();runner.cmd.assert_not_called()

    def test_rejected_fast_manual_proposal_cannot_hide_correct_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);entries=[]
            for name,points in [('bad',.95),('repair',.84)]:
                source=root/(name+'.py');source.write_text('def run(x):\n    return x+'+str(points)+'\n')
                result=root/(name+'.json');controller.save(result,dict(points=points))
                entries.append(dict(source=str(source),source_sha256=controller.sha(source),
                    search_result=str(result),qualification='passed',compiled_status='verified'))
            rejected=root/'review-rejections/op'/(entries[0]['source_sha256']+'.json')
            controller.save(rejected,dict(operator='op',source_sha256=entries[0]['source_sha256'],
                status='manual_review_rejected',reason='缓存布局不完整'))
            runner=controller.Runner.__new__(controller.Runner)
            runner.state=dict(operators={'op':dict(attempts=[],proposals=entries,best=entries[0])})
            runner.previous={'op':dict(pytorch=dict(status='PASSED',score=.7),supplementary=dict(status='passed'))}
            with mock.patch.object(controller,'BATCH',root),mock.patch.object(measurements,'score',side_effect=lambda value:value['points']):
                runner.refresh_existing_scores()
            self.assertIs(runner.state['operators']['op']['best'],entries[1])
            self.assertEqual(entries[0]['qualification'],'manual_review_rejected')
            self.assertEqual(entries[0]['compiled_status'],'not_qualified')

    def test_new_verified_proposal_is_retained_without_starting_duplicate_search(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = controller.Runner.__new__(controller.Runner)
            runner.q = SimpleNamespace(plan={'passes': [{}]}, runtime=Path(folder), records={})
            runner.module = SimpleNamespace(now=lambda: 'now')
            runner.state = {'operators': {}, 'jobs': {}}
            runner.persist = mock.Mock()
            ready = dict(qualification='passed', reference='qualified_CANN', search_score=.84,
                         compiled_status='verified', review_status='reviewed_strict_triton',
                         source='reviewed.py', source_sha256='verified-source')
            def finish_proposal():
                runner.state['operators']['op']['best'] = ready
            runner.process_proposals = finish_proposal
            runner.prepare_paired = mock.Mock(side_effect=AssertionError('不应重复启动'))
            runner.search('op', 1)
            self.assertEqual(runner.state['jobs']['op:search-1']['result']['status'],
                             'verified_target_retained_no_remeasure')
            runner.prepare_paired.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_rejected_fast_source_does_not_block_correct_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = root/'workspace'
            controller.save(workspace/'.kernelgen/run-process.json', {'finished_at': 'terminal'})
            contract = workspace/'stages/optimize/work/.kernelgen/evaluation-contract.json'
            controller.save(contract, {'definition': {'reference': 'frozen reference'}})
            leaf = root/'catalog/ops/op'
            leaf.mkdir(parents=True)
            (leaf/'oracle.py').write_text('frozen reference')
            source = root/'candidate/main.py'
            source.parent.mkdir()
            source.write_text('def run(x):\n    return x\n')
            entry = dict(source=str(source), search_result=str(contract.parent/'evals/round-0001/result.json'),
                         search_score=.95, qualification='passed')
            rejection = root/'review-rejections/op'/(controller.sha(source)+'.json')
            controller.save(rejection, dict(operator='op', source_sha256=controller.sha(source),
                                           status='manual_review_rejected', reason='重复原地提交'))
            corrected = root/'corrected.py'
            corrected.write_text('def run(x):\n    return x+1\n')
            result = root/'corrected-result.json'
            controller.save(result, {})
            repair = dict(source=str(corrected), source_sha256=controller.sha(corrected),
                          search_result=str(result), search_score=.84, qualification='passed')
            runner = controller.Runner.__new__(controller.Runner)
            runner.state = {'operators': {'op': {'attempts': [{'candidates': [entry]}],
                                               'proposals': [repair], 'best': entry}}}
            runner.previous = {'op': {'cann': {'score': .7}, 'supplementary': {'status': 'passed'}}}
            with mock.patch.object(controller, 'BATCH', root), \
                    mock.patch.object(measurements, 'ledger', return_value={'status': 'passed', 'score': .95}), \
                    mock.patch.object(measurements, 'score', return_value=.84):
                runner.refresh_existing_scores()
            self.assertEqual(entry['qualification'], 'manual_review_rejected')
            self.assertIs(runner.state['operators']['op']['best'], repair)


if __name__ == '__main__':
    unittest.main()
