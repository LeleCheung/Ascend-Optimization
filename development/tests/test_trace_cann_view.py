"""验证固定参考重算的数学口径和用例完整性，避免凭总分换算。"""
import copy
import math
import unittest

from development.tools.traces.cann_trace import rebase, timing_contract


def result(candidates, references=(12.0, 20.0)):
    return {'per_workload':[dict(uuid='op-timing-'+str(i),phase='timing',status='PASSED',
                                latency_ms=c,reference_latency_ms=r)
                            for i,(c,r) in enumerate(zip(candidates,references))]}


class FixedReferenceTest(unittest.TestCase):
    def test_uses_candidate_latencies_not_reported_aggregate(self):
        anchor=result((6.0,10.0));old=result((3.0,10.0),(900.0,1.0));old['geo_mean']=999
        self.assertAlmostEqual(rebase(anchor,old)['score'],math.sqrt(8))

    def test_anchor_self_reproduces_geometric_mean(self):
        anchor=result((6.0,5.0))
        self.assertAlmostEqual(rebase(anchor,anchor)['score'],math.sqrt(8))

    def test_missing_or_duplicate_case_is_rejected(self):
        anchor=result((6.0,5.0));bad=copy.deepcopy(anchor)
        bad['per_workload'].pop()
        with self.assertRaises(ValueError):rebase(anchor,bad)
        bad=copy.deepcopy(anchor);bad['per_workload'][1]['uuid']=bad['per_workload'][0]['uuid']
        with self.assertRaises(ValueError):rebase(anchor,bad)

    def test_failed_or_nonpositive_case_is_rejected(self):
        anchor=result((6.0,5.0));bad=copy.deepcopy(anchor)
        bad['per_workload'][0]['status']='FAILED'
        with self.assertRaises(ValueError):rebase(anchor,bad)
        for latency in (0,-1,float('inf'),float('nan')):
            bad=result((latency,5.0))
            with self.assertRaises(ValueError):rebase(anchor,bad)

    def test_only_known_operator_name_alias_is_normalized(self):
        old=b'{"name":"conj_physical-timing-0","seed":1729,"inputs":{"shape":[8]}}\n'
        new=old.replace(b'conj_physical',b'_conj_physical')
        self.assertEqual(timing_contract(old),timing_contract(new))
        self.assertNotEqual(timing_contract(old),timing_contract(new.replace(b'[8]',b'[16]')))


if __name__=='__main__':unittest.main()
