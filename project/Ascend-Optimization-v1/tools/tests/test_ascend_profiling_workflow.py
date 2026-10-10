"""用 910B 真实归档检验跨来源绑定和证据拒绝条件。"""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
REPORT = PROJECT / "operators" / "narrow_copy" / "reports" / "narrow-copy-combined-20261006"
spec = importlib.util.spec_from_file_location("workflow", HERE.parent / "profiling" / "ascend_profiling_workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class EvidenceWorkflowTest(unittest.TestCase):
    def setUp(self):
        # 历史评测绑定当次源码，实验目录中的候选可能已继续优化。
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name) / "candidate.py"
        request = workflow.read_json(REPORT / "eval.request.json")
        self.source.write_text(workflow.source_in(request), encoding="utf-8")

    def inputs(self, *profiles):
        return ("narrow_copy", self.source, REPORT / "eval.json", REPORT / "eval.request.json",
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

    def test_device_timing_cannot_diagnose_host_gap(self):
        # 同一份真实证据只改变计时范围，设备计时不可推出 host 开销。
        report = workflow.analyze(*self.inputs(self.small()), timing_scope="device_kernel")
        self.assertFalse(any(case["diagnosis"]["bound"] == "host_or_launch_hypothesis"
                             for case in report["cases"]))


class MasterMatmulEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT / "operators" / "matmul_bias_activation" / "reports" / "master-20261010"

    def analyze(self, provenance=None):
        profiles = []
        for name in ("profile-metrics-float16-0", "profile-metrics-float16-2",
                     "profile-instruction-float16-0"):
            directory = self.root / name
            profiles.append((directory / "response.json", directory / "request.json",
                             directory / "artifacts"))
        return workflow.analyze(
            "matmul_bias_activation", self.root / "flaggems-master.py",
            self.root / "flaggems-master-1.result.json",
            self.root / "flaggems-master-1.request.json", self.root / "inspect.json",
            profiles, timing_scope="device_kernel",
            evaluation_provenance=provenance or self.root / "flaggems-master.provenance.json")

    def test_real_master_provenance_and_matmul_diagnosis(self):
        report = self.analyze()
        self.assertEqual(report["evaluation"]["num_passed"], 42)
        self.assertEqual(report["coverage"], {"timing_cases": 15, "profiled_cases": 2})
        bounds = {case["diagnosis"]["bound"] for case in report["cases"]}
        self.assertIn("gemm_control_and_parallelism", bounds)
        self.assertIn("gemm_data_pipeline", bounds)
        self.assertNotIn("host_or_launch_hypothesis", bounds)

    def test_tampered_provenance_is_rejected(self):
        for key, value, error in (("source_sha256", "0" * 64, "SHA 不匹配"),
                                  ("binding", {"definition": "amin"}, "binding 不匹配")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                provenance = workflow.read_json(self.root / "flaggems-master.provenance.json")
                provenance[key] = value
                path = Path(directory) / "provenance.json"
                path.write_text(json.dumps(provenance), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    self.analyze(path)


if __name__ == "__main__":
    unittest.main()
