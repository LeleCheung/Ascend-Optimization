"""验证知识准入与去重，不将缺失证据升级为成功结论。"""
import copy
import tempfile
import unittest
from pathlib import Path

from hgkda.curation import organize, retrieve
from hgkda.context import knowledge


class KnowledgeTests(unittest.TestCase):
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
        self.assertEqual([r['id'] for r in result['mechanisms']],['expm1-precision-buffer'])
        self.assertTrue(all(r['source_sha256'] for r in result['mechanisms']))


if __name__=='__main__':unittest.main()
