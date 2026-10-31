"""CPU回退参考的数值成绩不能进入常规NPU达标计数。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from report import generate,reference_stability


class ReportTests(unittest.TestCase):
    def test_stability_counterexample_requires_input_identity_and_empty_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            batch=Path(directory)/'operators/campaigns/test'
            root=batch/'analysis/cumulative_trapezoid/reference-stability-replay-v1';root.mkdir(parents=True)
            raw=b'\0'*4096;sha=hashlib.sha256(raw).hexdigest()
            (root/'input-fp16-256x8.bin').write_bytes(raw)
            script=b'# frozen reproducer\n';(root/'reproducer.py').write_bytes(script)
            def write(name,value):(root/name).write_text(json.dumps(value),encoding='utf8')
            write('provenance.json',dict(input_sha256=sha,script_sha256=hashlib.sha256(script).hexdigest()))
            base=dict(input_sha256=sha,shape=[256,8],dtype='float16',dim=0,dx=1,repeats=12,
                torch_version='2.9.0+cpu',torch_npu_version='2.9.0.post2',torch_git='fixed',physical_device=7)
            default=dict(base,deterministic=False,empty_tolerance_intersections=3,
                position=[233,1],values=[-.13,-.19]*6)
            write('default.json',default)
            write('deterministic.json',dict(base,deterministic=True,empty_tolerance_intersections=0,over_tolerance_vs_first=[0]*11))
            result=reference_stability(batch,'cumulative_trapezoid')
            self.assertEqual(result['status'],'default_reference_inconsistent')
            self.assertTrue(result['deterministic_consistent_in_tolerance'])
            write('default.json',dict(default,values=[-.13]*12))
            self.assertEqual(reference_stability(batch,'cumulative_trapezoid')['status'],'evidence_requires_review')
            write('default.json',default)
            (root/'input-fp16-256x8.bin').write_bytes(b'changed')
            self.assertEqual(reference_stability(batch,'cumulative_trapezoid')['status'],'evidence_requires_review')

    def test_npu_and_separate_reference_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def write(path,value):path.write_text(json.dumps(value),encoding='utf8')
            def record(name,score,extra=False):
                source=root/(name+'.py');source.write_text('# '+name+'\ndef run(x): return x\n')
                sha=hashlib.sha256(source.read_bytes()).hexdigest()
                result=root/(name+'.result.json')
                write(result,dict(status='PASSED',num_passed=1,num_workloads=1,geo_mean=score))
                write(root/(name+'.request.json'),dict(implementation=dict(sources=[dict(content=source.read_text())])))
                row=dict(source=str(source),source_sha256=sha,result=str(result))
                if extra:
                    gates=root/(name+'-extra.json');write(gates,dict(status='passed',operator='x',source_sha256=sha))
                    row['extra_checks']=str(gates)
                return row
            baseline=record('original',.3);best=record('new',1.1,True)
            write(root/'targets.json',dict(targets=[dict(order=1,operator='x',source_sha256=baseline['source_sha256'],status='prepared')]))
            write(root/'software-lock.json',{})
            for reference,expected in [('reference_requires_separate_review',False),('not_run',False),('npu_paths_observed',True)]:
                write(root/'next-state.json',dict(status='running',operators=dict(x=dict(baseline=baseline,best=best,reference_path_status=reference))))
                row=generate(root,quiet=True)['operators'][0]
                self.assertTrue(row['numerical_threshold_passed'])
                self.assertEqual(row['threshold_passed'],expected)
                self.assertEqual(row['comparison_conditions_require_review'],not expected)


if __name__=='__main__':unittest.main()
