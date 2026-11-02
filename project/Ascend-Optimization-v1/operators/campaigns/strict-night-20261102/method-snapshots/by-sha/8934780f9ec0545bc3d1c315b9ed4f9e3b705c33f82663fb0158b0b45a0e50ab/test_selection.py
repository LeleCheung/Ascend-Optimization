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

spec = importlib.util.spec_from_file_location('night_selection_controller', HERE/'run.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class SelectionTests(unittest.TestCase):
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
