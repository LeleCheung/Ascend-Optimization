"""回归：恢复队列时不把更高分或缺少候选延迟的CANN对照当作候选。"""
import unittest
from selection_roles import optimization_jobs


class Tests(unittest.TestCase):
    def test_reference_and_candidate_are_not_interchangeable(self):
        jobs={'verify-current:log10_':{'result':{'geo_mean':.72,'latency_ms':{'one':.01}}},
              'reference-evaluate:log10_':{'result':{'geo_mean':10.0}},
              'reference-cpp:log10_:v1':{'result':{'geo_mean':20.0}},
              'queued-proposal:log10_:ascendc-taskqueue-v2':{'result':{'geo_mean':1.0}}}
        selected=optimization_jobs(jobs)
        self.assertEqual(set(selected),{'verify-current:log10_','queued-proposal:log10_:ascendc-taskqueue-v2'})
        self.assertEqual(len(jobs),4)

    def test_native_candidate_is_retained(self):
        job={'result':{'backend':'AscendC_with_Triton_layout_fallback'}}
        self.assertIs(optimization_jobs({'native-candidate-evaluate:log10_:v1':job})['native-candidate-evaluate:log10_:v1'],job)


if __name__=='__main__':unittest.main()
