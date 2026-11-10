"""验证知识准入与去重，不将缺失证据升级为成功结论。"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hgkda.curation import organize, retrieve
from hgkda.context import knowledge


class KnowledgeTests(unittest.TestCase):
    def test_performance_failure_requires_matching_score_source_and_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source='import triton\n@triton.jit\ndef k():\n    pass\n'
            digest=hashlib.sha256(source.encode()).hexdigest()
            (root/'candidate.py').write_bytes(source.encode())
            values={'result.json':dict(status='PASSED',num_passed=1,num_workloads=1,geo_mean=.75),
                'request.json':dict(implementation=dict(sources=[dict(content=source)])),
                'review.json':dict(source_sha256=digest,status='reviewed_strict_triton',all_branches_reviewed=True,dependencies_reviewed=True),
                'feedback.json':dict(result=dict(source_sha256=digest,geo_mean=.75,threshold=.8,decision='continue_optimization',accepted=False))}
            for name,value in values.items():(root/name).write_text(json.dumps(value))
            sha=lambda name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            record=dict(id='failed',operator_type='pointwise',action_id='prefetch',
                scope=dict(operator='log10_',performance_threshold=.8),summary='正确性通过但性能未过线',status='observed_failure',
                evidence=dict(source='candidate.py',source_sha256=digest,formal_result='result.json',formal_result_sha256=sha('result.json'),
                    formal_request='request.json',backend_review='review.json',backend_review_sha256=sha('review.json'),
                    performance_feedback=dict(path='feedback.json',sha256=sha('feedback.json'))))
            self.assertEqual(organize([record],root=root)['active'][0]['status'],'observed_failure')
            invalid=copy.deepcopy(record);invalid['scope']['performance_threshold']=.9
            self.assertEqual(organize([invalid],root=root)['active'][0]['status'],'draft')
            values['feedback.json']['result']['source_sha256']='wrong'
            (root/'feedback.json').write_text(json.dumps(values['feedback.json']))
            record['evidence']['performance_feedback']['sha256']=sha('feedback.json')
            self.assertEqual(organize([record],root=root)['active'][0]['status'],'draft')

    def test_missing_evidence_downgrades_validation(self):
        record={'id':'sample','operator_type':'pointwise','action_id':'tile',
                'scope':{'operator':'expm1_'},'summary':'待核验分块修改',
                'status':'validated','evidence':{'source':'missing.py','formal_result':'missing.json'}}
        with tempfile.TemporaryDirectory() as temp:
            index=organize([record],root=Path(temp))
            self.assertEqual(index['active'][0]['status'],'draft')
            result=retrieve(index,'pointwise',{'operator':'expm1_'})
            self.assertEqual(result['records'],[])

    def test_dedup_keeps_history(self):
        first={'id':'old','operator_type':'pointwise','action_id':'tile','scope':{'operator':'expm1_'},
               'summary':'历史待验证假设','status':'draft','evidence':{'note':'unverified'},'updated_at':'2026-11-01'}
        latest=copy.deepcopy(first);latest.update(id='new',updated_at='2026-11-03')
        result=organize([first,latest])
        self.assertEqual(result['active'][0]['id'],'new')
        self.assertEqual(result['superseded'][0]['id'],'old')

    def test_operator_knowledge_is_scoped(self):
        result=knowledge('expm1_')
        self.assertEqual(result['operator_type'],'pointwise')
        self.assertEqual([r['id'] for r in result['mechanisms'] if r['match']=='operator'],['expm1-precision-buffer'])
        self.assertTrue(all(r['source_sha256'] for r in result['mechanisms']))

    def test_selu_retrieves_bound_launch_experience(self):
        import hashlib
        root=Path(__file__).resolve().parents[1]
        result=knowledge('selu_')
        self.assertFalse(result['retrieval']['empty'])
        exact=[r for r in result['mechanisms'] if r['match']=='operator']
        self.assertEqual(len(exact),1)
        record=exact[0]
        self.assertEqual(hashlib.sha256((root/record['source']).read_bytes()).hexdigest(),record['source_sha256'])
        self.assertTrue(all((root/path).exists() for path in record['evidence']))
        current=record['current_validation']
        for key in ['source','formal_result','audit','summary']:
            item=current[key]
            self.assertEqual(hashlib.sha256((root/item['path']).read_bytes()).hexdigest(),item['sha256'])
        self.assertTrue(current['accepted'])
        other=knowledge('amin')
        self.assertNotIn(record['id'],[r['id'] for r in other['mechanisms']])

    def test_type_transfer_is_explicit_and_keeps_provenance(self):
        result=knowledge('negative_')
        transferable=[r for r in result['mechanisms'] if r['match']=='explicit_type_transfer']
        self.assertTrue(transferable)
        self.assertTrue(all(r['transferable_by_type'] for r in transferable))
        self.assertTrue(all(r['operators']==['selu_'] for r in transferable))


if __name__=='__main__':unittest.main()
