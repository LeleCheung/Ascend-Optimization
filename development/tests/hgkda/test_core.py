"""验证统计范围、身份约束与时间重叠处理。"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hgkda.artifacts import emit, envelope
from hgkda.compare import compare
from hgkda.context import build_context
from hgkda.diagnostics import diagnose_case
from hgkda.regions import analyze, length, subtract, union

ROOT = Path(__file__).resolve().parents[3]


class EvidenceTests(unittest.TestCase):
    def test_real_pair_and_identity(self):
        directory = ROOT / 'hgkda/examples/paired-expm1'
        result = compare(directory / 'paired')
        self.assertTrue(result['differences'])
        self.assertTrue(result['derived_differences'])
        with tempfile.TemporaryDirectory() as temp:
            comparison = Path(temp) / 'comparison.json'
            emit(envelope('compare', result), comparison)
            context = build_context('expm1_', directory / 'candidate.py', comparison)
            self.assertEqual(context['source']['sha256'], result['spec']['source_sha256'])
            self.assertEqual(context['reference_kind'], 'original_PyTorch_only')
            with self.assertRaises(ValueError):
                build_context('selu_', directory / 'candidate.py', comparison)
            with self.assertRaises(FileExistsError):
                emit({}, comparison)

    def test_invalid_metrics_are_not_evidence(self):
        result = diagnose_case({'operator': 'amin', 'metrics': {'aiv_mte2_ratio': float('nan')}})
        self.assertEqual(result['stage'], 'collect_metrics')
        self.assertEqual(result['hypotheses'], [])
        result = diagnose_case({'operator': 'amin', 'metrics': {'aiv_mte2_ratio': .8}})
        self.assertEqual(result['hypotheses'][0]['category'], 'data_movement')
        self.assertIn('尚未证明', result['hypotheses'][0]['interpretation'])

    def test_zero_denominator_and_missing_counter(self):
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'paired'
            shutil.copytree(ROOT / 'hgkda/examples/paired-expm1/paired', directory)
            import csv
            for file in (directory / 'reference/profile').glob('Memory.csv'):
                with file.open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    if 'aiv_mte2_instructions' in row:
                        row['aiv_mte2_instructions'] = '0'
                with file.open('w', encoding='utf8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            result = compare(directory)
            values = [r for r in result['differences'] if r['metric'] == 'aiv_mte2_instructions']
            self.assertTrue(values)
            self.assertTrue(all(r['candidate_over_reference'] is None for r in values))
            meta = directory / 'candidate/collection.json'
            data = json.loads(meta.read_bytes()); data['spec_sha256'] = '0' * 64
            meta.write_text(json.dumps(data), encoding='utf8')
            with self.assertRaises(ValueError):
                compare(directory)

    def test_overlapping_pipeline_intervals(self):
        self.assertEqual(union([(0, 5), (3, 7), (8, 9)]), [[0., 7.], [8., 9.]])
        self.assertEqual(length(subtract([(0, 7)], [(2, 4)])), 5)
        listing = {'kernels': [{'name': 'k', 'cores': [{'name': 'c', 'instructions': []}]}]}
        events = [{'ph': 'M', 'name': 'process_name', 'pid': 1, 'args': {'name': 'VECTOR'}},
                  {'ph': 'X', 'name': 'WAIT', 'pid': 1, 'ts': 0, 'dur': 5},
                  {'ph': 'X', 'name': 'ADD', 'pid': 1, 'ts': 3, 'dur': 5}]
        row = analyze(listing, events, kernel='k', core='c', capture=(0, 10))['timeline']['pipelines'][0]
        self.assertEqual(row['productive_union'] + row['wait_only_union'] + row['idle_union'], 10)
        with self.assertRaises(ValueError):
            analyze(listing, events)

    def test_bound_timeline_rejects_wrong_case_core_and_unit(self):
        listing=dict(source_sha256='source',case=dict(name='case-a'),kernels=[
            dict(name='k',cores=[dict(name='c',instructions=[])])])
        binding=dict(source_sha256='source',case=dict(name='case-a'),kernel='k',core='c',time_unit='us')
        trace=dict(hgkda_binding=binding,traceEvents=[
            dict(ph='M',name='process_name',pid=1,args=dict(name='MTE2')),
            dict(ph='X',name='LOAD',pid=1,ts=0,dur=2),
            dict(ph='X',name='WAIT',pid=1,ts=3,dur=1)])
        result=analyze(listing,trace,kernel='k',core='c',capture=(0,6))
        row=result['timeline']['pipelines'][0]
        self.assertEqual([row[k] for k in ['productive_union','wait_only_union','idle_union']],[2,1,3])
        for field,value in [('case',dict(name='case-b')),('core','other'),('time_unit','ns'),('source_sha256','other')]:
            invalid=dict(trace,hgkda_binding=dict(binding,**{field:value}))
            with self.assertRaises(ValueError):analyze(listing,invalid,kernel='k',core='c')


if __name__ == '__main__':
    unittest.main()
