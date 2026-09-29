"""Audit regression tests: prevent omissions and unsupported closure claims."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    path = ROOT / 'scripts' / 'analysis' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    if Path(value.__file__).resolve() != path:
        raise AssertionError('Wrong worktree imported')
    return value


counter = module('analyze_nonascend_retest')
renderer = module('render_nonascend_retest')


class WorkbookTests(unittest.TestCase):
    def test_pass_but_below_threshold_is_failure(self):
        self.assertEqual(counter.failure_reasons({'D': 'PASS', 'E': '0.7988x'}), ['speedup_below_0_8'])
        self.assertEqual(counter.failure_reasons({'D': 'PASS', 'E': '0.800x'}), [])
        self.assertEqual(counter.speed('0.214x*'), .214)
        self.assertIsNone(counter.speed('—'))
        self.assertIsNone(counter.speed('TIMEOUT'))

    def test_ppu_supplementary_failure_is_not_lost(self):
        self.assertIn('explicit_failure', counter.failure_reasons({'D': '', 'E': '0.8258', 'H': '错误'}))
        self.assertIn('explicit_failure', counter.failure_reasons({'D': '', 'E': '1.5022', 'H': '正确性不过'}))
        self.assertEqual(counter.failure_reasons({'D': 'PASS', 'E': '1.0', 'F': '环境说明'}), [])

    def test_timeout_never_turns_into_a_numeric_speedup(self):
        self.assertIn('incomplete_test', counter.failure_reasons({'D': 'TIMEOUT'}))
        self.assertIn('timing_unavailable', counter.failure_reasons({'D': 'PASS', 'E': 'benchmark 超时'}))

    def test_read_actual_xlsx_shared_and_inline_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'workbook.xlsx'
            with ZipFile(path, 'w') as z:
                z.writestr('xl/workbook.xml', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="平头哥测试" sheetId="1" r:id="rId1"/></sheets></workbook>')
                z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="rId1" Target="/xl/worksheets/sheet1.xml"/></Relationships>')
                z.writestr('xl/sharedStrings.xml', '<sst><si><t>linear</t></si></sst>')
                z.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="26"><c r="A26" t="s"><v>0</v></c><c r="E26"><v>0.6325</v></c><c r="H26" t="inlineStr"><is><t>错误</t></is></c></row></sheetData></worksheet>')
            row = counter.read_workbook(path)[0]['rows'][0]
            self.assertEqual(row['id'], 'pingtouge:26')
            self.assertEqual(row['cells']['A'], 'linear')
            self.assertEqual(set(counter.failure_reasons(row['cells'])), {'explicit_failure', 'speedup_below_0_8'})


class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT/'docs/operations/data/nonascend_retest_failure_20260908.json').read_text())

    def test_complete_cohort_and_exclusive_distribution(self):
        result = renderer.stats(self.data)
        self.assertEqual(result['total'], 44)
        self.assertEqual(result['unique_operator_names'], 38)
        self.assertEqual(result['by_vendor'], {'tianshu': 9, 'moer': 13, 'pingtouge': 9, 'muxi': 13})
        self.assertEqual(sum(result['symptoms_exclusive'].values()), 44)
        self.assertEqual(sum(result['priorities_exclusive'].values()), 44)
        self.assertEqual(result['with_confirmed_related_issue'], 31)
        self.assertEqual(result['without_confirmed_related_issue'], 13)
        self.assertEqual(result['exact_original_failure_closed'], 0)

    def test_overlap_is_not_added_as_exclusive_root_count(self):
        result = renderer.stats(self.data)
        sets = result['confirmed_issue_sets_overlapping']
        self.assertEqual(len(sets['candidate']), 17)
        self.assertEqual(len(sets['evaluation_or_comparability']), 17)
        self.assertEqual(len(sets['vendor_compiler_related']), 2)
        self.assertEqual(len(set().union(*map(set, sets.values()))), 31)
        self.assertEqual(result['group_counts']['BF16'], {'confirmed': 2, 'reported_only': 4, 'total': 6})

    def test_no_row_can_disappear_or_duplicate(self):
        data = copy.deepcopy(self.data)
        data['rows'].pop()
        with self.assertRaises(ValueError):
            renderer.stats(data)
        data = copy.deepcopy(self.data)
        data['rows'][0] = data['rows'][1]
        with self.assertRaises(ValueError):
            renderer.stats(data)

    def test_core_pass_cannot_be_used_as_original_failure_closure(self):
        data = copy.deepcopy(self.data)
        data['rows'][0]['original_failure_closed'] = True
        with self.assertRaises(ValueError):
            renderer.stats(data)

    def test_generated_report_and_summary_are_current(self):
        self.assertEqual(renderer.render(self.data), (ROOT/'docs/operations/troubleshooting/nonascend_retest_failure_analysis.md').read_text())
        expected = json.loads((ROOT/'docs/operations/data/nonascend_retest_failure_summary_20260908.json').read_text())
        self.assertEqual(renderer.stats(self.data), expected)


class MainlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = module('render_nonascend_mainline')
        cls.data = json.loads((ROOT/'docs/operations/data/nonascend_retest_mainline_20260908.json').read_text())
        cls.source = json.loads((ROOT/'docs/operations/data/nonascend_retest_failure_20260908.json').read_text())

    def test_original_cohort_and_overlapping_counts_preserved(self):
        result = self.renderer.summarize(self.data, self.source)
        self.assertEqual(result['total'], 44)
        self.assertEqual(sum(result['primary_categories_exclusive'].values()), 44)
        self.assertEqual(result['queue_counts_overlapping']['PERFORMANCE'], 26)
        self.assertEqual(result['queue_counts_overlapping']['SHARED_MEMORY'], 4)
        self.assertEqual(result['primary_categories_exclusive']['resource'], 3)
        self.assertEqual(result['evidence_status_exclusive'], {'pending': 30, 'related': 12, 'located': 2})

    def test_independent_defects_do_not_reclassify_original_failures(self):
        rows = {r['id']: r for r in self.data['rows']}
        self.assertEqual(rows['muxi:17']['primary_category'], 'compiler')
        self.assertEqual(rows['muxi:51']['primary_category'], 'performance')
        self.assertEqual(rows['tianshu:40']['primary_category'], 'resource')
        self.assertEqual(rows['muxi:34']['primary_category'], 'timeout')
        self.assertEqual(rows['muxi:48']['primary_category'], 'test_scope')
        result = self.renderer.summarize(self.data, self.source)
        self.assertEqual(result['primary_categories_exclusive']['compiler'], 7)
        self.assertEqual(result['queue_counts_overlapping']['ILLEGAL_ACCESS'], 1)

    def test_missing_row_or_altered_original_measurement_is_rejected(self):
        data = copy.deepcopy(self.data)
        data['rows'].pop()
        with self.assertRaises(ValueError):
            self.renderer.summarize(data, self.source)
        data = copy.deepcopy(self.data)
        data['rows'][0]['reported_speedup'] = 10.
        with self.assertRaises(ValueError):
            self.renderer.summarize(data, self.source)

    def test_rendered_mainline_is_current(self):
        self.assertEqual(self.renderer.render(self.data, self.source),
                         (ROOT/'docs/validation/nonascend_retest_mainline.md').read_text())
        self.assertEqual(self.renderer.summarize(self.data, self.source),
                         json.loads((ROOT/'docs/operations/data/nonascend_retest_mainline_summary_20260908.json').read_text()))


class PerformanceFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = module('analyze_nonascend_performance')
        cls.data = json.loads((ROOT/'docs/operations/data/nonascend_performance_followup_20260908.json').read_text())

    def test_scope_keeps_starred_speedup_and_excludes_dispatch_once(self):
        original = json.loads((ROOT/'docs/operations/data/nonascend_retest_mainline_20260908.json').read_text())
        expected = {r['id'] for r in original['rows'] if isinstance(r['reported_speedup'], (int, float)) and 0 < r['reported_speedup'] < .8}
        rows = self.data['rows']
        ids = {r['id'] for r in rows}
        self.assertEqual(len(rows), len(ids))
        self.assertEqual(ids | {self.data['dispatch_record']['id']}, expected)
        self.assertNotIn('pingtouge:2', ids)
        self.assertEqual(next(r for r in rows if r['id'] == 'tianshu:17')['reported_speedup'], .214)

    def test_baseline_pollution_never_becomes_usable_latency_comparison(self):
        row = next(r for r in self.data['rows'] if r['id'] == 'pingtouge:8')
        self.assertTrue(row['paired'])
        self.assertFalse(row['usable_for_recorded_latency_comparison'])
        self.assertFalse(any(r['exact_original_failure_closed'] for r in self.data['rows']))

    def test_recovered_pairs_recompute_both_sides_and_reject_duplicate_keys(self):
        code = 'candidate'
        expected = self.analyzer.sha(code)
        row = {'id': 'synthetic', 'candidate_sha256': expected}
        workload = {'uuid': 'one', 'axes': {'dtype': 'float32', 'shape_detail': [[4]]}, 'latency_ms': 2., 'reference_latency_ms': 4.}
        ledger = {'best_round': 1, 'rounds': [{'round_num': 1, 'solution': {'code': code}, 'evaluation': {'geo_mean': 2., 'workloads': [workload]}}]}
        case = {'dtype': 'torch.float32', 'shape_detail': [[4]], 'latency': 1., 'latency_base': 1.}
        native = {'candidate_sha256': expected, 'timing': {'cases': [case]}}
        self.analyzer.recover(row, ledger, native)
        self.assertEqual(row['reference_ratio_geo'], .25)
        self.assertEqual(row['candidate_ratio_geo'], .5)
        native['timing']['cases'].append(case)
        self.analyzer.recover(row, ledger, native)
        self.assertEqual(row['matched_cases'], 0)
        self.assertFalse(row['full_timing_signature_match'])
        self.assertNotIn('reference_ratio_geo', row)
        native['candidate_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            self.analyzer.recover(row, ledger, native)


if __name__ == '__main__':
    unittest.main()
