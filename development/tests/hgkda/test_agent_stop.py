"""区分验收达标与性能优先搜索的停止条件。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('agent_runner',Path(__file__).resolve().parents[3]/'hgkda/src/hgkda/runtime/agent_runner.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class AgentStopTests(unittest.TestCase):
    def test_default_generation_has_no_workflow_cap(self):
        self.assertEqual(module.generation_limits(), ([], None))
        self.assertEqual(module.generation_limits(12, 60), (['--max-turns','12'], 60))
        with self.assertRaises(ValueError):module.generation_limits(-1)

    def test_acceptance_does_not_end_performance_first_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'performance-feedback.json').write_text(json.dumps(
                dict(source_sha256='current',decision='target_met',accepted=True)))
            self.assertFalse(module.should_finish({},root,'current'))
            self.assertTrue(module.should_finish({'optimization_policy':'target_only'},root,'current'))
            self.assertFalse(module.should_finish({'optimization_policy':'target_only'},root,'other'))

    def test_current_formal_target_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            path=root/'performance-feedback.json'
            for feedback,expected in [
                ({'source_sha256':'current','decision':'target_met','accepted':True},True),
                ({'source_sha256':'old','decision':'target_met','accepted':True},False),
                ({'source_sha256':'current','decision':'complete_supplementary','accepted':False},False),
                ({'source_sha256':'current','decision':'continue_optimization','accepted':False},False),
            ]:
                path.write_text(json.dumps(feedback),encoding='utf8')
                self.assertEqual(module.target_met(root,'current'),expected)

    def test_no_feedback_is_not_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertFalse(module.target_met(Path(temporary),'current'))


if __name__=='__main__':unittest.main()
