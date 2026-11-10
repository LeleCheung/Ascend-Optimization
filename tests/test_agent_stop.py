"""模型读完达标反馈后保持源码不变，仍应识别为正常终止。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('agent_runner',Path(__file__).resolve().parents[1]/'tools/skill_validation/agent_runner.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class AgentStopTests(unittest.TestCase):
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
