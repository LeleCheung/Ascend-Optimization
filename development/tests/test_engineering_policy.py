"""Verify language boundaries, historical relocation identity, and legacy-table access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from development.tools.lineage.render import render
from development.tools.maintenance.audit_language import audit
from development.tools.maintenance.resolve_path import relocation_records, resolve


ROOT = Path(__file__).resolve().parents[2]


class EngineeringPolicyTest(unittest.TestCase):
    def test_maintained_text_is_english_and_provenance_is_bound(self):
        result = audit(ROOT)
        self.assertEqual(result["status"], "passed", result["findings"])
        self.assertEqual(result["retained_provenance_identifiers"], 6)

    def test_recorded_moves_preserve_bytes_and_resolve_chained_paths(self):
        for record in relocation_records(ROOT):
            with self.subTest(path=record["old"]):
                found = resolve(ROOT, record["old"])
                self.assertEqual(found["sha256"], record["sha256"])
                self.assertNotEqual(found["requested"], found["path"])
        with self.assertRaisesRegex(ValueError, "without traversal"):
            resolve(ROOT, "../outside")
        with self.assertRaisesRegex(ValueError, "without traversal"):
            resolve(ROOT, "C:/outside")

    def test_language_gate_rejects_new_prose_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_dir = root / "development/manifests"
            policy_dir.mkdir(parents=True)
            (policy_dir / "english-language-scope.json").write_text(json.dumps(dict(
                maintained_roots=[], maintained_files=["README.md"], excluded_roots=[],
                immutable_record_roots=[], provenance_identifiers=[])), encoding="utf-8")
            knowledge = root / "hgkda/src/hgkda/data/knowledge/mechanisms.json"
            knowledge.parent.mkdir(parents=True)
            knowledge.write_text('{"records": []}', encoding="utf-8")
            original = ("# " + chr(0x6d4b) + chr(0x8bd5) + "\n").encode("utf-8")
            (root / "README.md").write_bytes(original)
            result = audit(root)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["findings"][0]["line"], 1)
            self.assertEqual((root / "README.md").read_bytes(), original)

    def test_english_report_reads_original_table_headers_without_losing_values(self):
        base = ROOT / "development/experiments/lineage/20261109"
        index = json.loads((base / "index.json").read_bytes())
        tables = json.loads((base / "inputs/previous-report-tables.json").read_bytes())
        original = copy.deepcopy(tables)
        rendered = render(index, tables, "offline validation")
        self.assertEqual(tables, original)
        self.assertIn("Main operators (1015)", rendered)
        self.assertIn("Early optimization and method validation", rendered)
        # Historical reference values must survive input-header normalization.
        self.assertIn(str(tables[2][1][1]), rendered)


if __name__ == "__main__":
    unittest.main()
