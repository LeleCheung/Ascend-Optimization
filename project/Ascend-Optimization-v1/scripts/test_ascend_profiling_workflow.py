"""用 910B 真实归档检验跨来源绑定和证据拒绝条件。"""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPORT = PROJECT / "reports" / "ascend910b" / "narrow-copy-combined-20261006"
SOURCE = PROJECT / "experiments" / "ascend910b" / "narrow_copy" / "launch_plan_compiled.py"
spec = importlib.util.spec_from_file_location("workflow", HERE / "ascend_profiling_workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class EvidenceWorkflowTest(unittest.TestCase):
    def inputs(self, *profiles):
        return ("narrow_copy", SOURCE, REPORT / "eval.json", REPORT / "eval.request.json",
                REPORT / "eval.inspect.json", list(profiles))

    def small(self):
        return (REPORT / "profile-small-metrics.json",
                REPORT / "profile-small-metrics-request.json",
                REPORT / "profile-small-metrics-artifacts")

    def test_real_evidence_and_missing_scope(self):
        report = workflow.analyze(*self.inputs(self.small()))
        self.assertEqual(report["coverage"], {"timing_cases": 15, "profiled_cases": 1})
        self.assertEqual(report["roofline"]["status"], "unavailable")
        small = next(c for c in report["cases"] if "::core::float16::0" in c["case_id"])
        large = next(c for c in report["cases"] if "::core::float16::4" in c["case_id"])
        self.assertEqual(small["diagnosis"]["bound"], "host_or_launch_hypothesis")
        self.assertEqual(large["diagnosis"]["bound"], "inconclusive")

    def test_changed_profile_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            request = workflow.read_json(self.small()[1])
            request["implementation"]["sources"][0]["content"] += "\n# tampered"
            path = Path(folder) / "request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "候选源码不匹配"):
                workflow.analyze(*self.inputs((self.small()[0], path, self.small()[2])))

    def test_changed_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            copied = Path(folder) / "artifacts"
            shutil.copytree(self.small()[2], copied)
            first = next(path for path in copied.iterdir() if path.name.startswith("a001-"))
            first.write_bytes(first.read_bytes()[:-1] + b"x")
            with self.assertRaisesRegex(ValueError, "artifact SHA"):
                workflow.analyze(*self.inputs((self.small()[0], self.small()[1], copied)))


if __name__ == "__main__":
    unittest.main()
