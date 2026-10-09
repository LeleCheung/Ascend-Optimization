"""用归档的 910B 证据验证 case/source 绑定及拒绝条件。"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent.parent / "operators" / "narrow_copy" / "reports" / "narrow-copy-20261005"
spec = importlib.util.spec_from_file_location("ascend_profiling_agent", HERE.parent / "profiling" / "ascend_profiling_agent.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class ProfilingEvidenceTest(unittest.TestCase):
    def inputs(self):
        return (
            EVIDENCE / "eval-round-0001.json",
            EVIDENCE / "profiling-metrics.json",
            EVIDENCE / "profiling-instruction.json",
            EVIDENCE / "narrow_copy.py",
        )

    def test_case_source_and_scope(self):
        requests = (EVIDENCE / "profiling-metrics-request.json", EVIDENCE / "profiling-instruction-request.json")
        result = agent.analyze(*self.inputs(), requests)
        self.assertEqual(result["case_id"], "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::3")
        self.assertEqual(result["roofline"]["status"], "unavailable")
        self.assertEqual(len(result["verified_profile_requests"]), 2)
        self.assertTrue(result["simulator"]["mapped_source_lines"])

    def test_wrong_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.json"
            value = agent.read_json(EVIDENCE / "profiling-metrics-request.json")
            value["case_id"] = "different-case"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "case_id 不匹配"):
                agent.analyze(*self.inputs(), (path,))

    def test_wrong_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.json"
            value = agent.read_json(EVIDENCE / "profiling-metrics-request.json")
            value["implementation"]["sources"][0]["content"] = "def run(): pass"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "候选源码不匹配"):
                agent.analyze(*self.inputs(), (path,))


if __name__ == "__main__":
    unittest.main()
