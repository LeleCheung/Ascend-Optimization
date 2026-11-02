"""汇总回归：空设备trace不能达标，参考实现不能进入候选上下文。使用合成证据测试门禁。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from common import save,read,sha,load
report=load('strict_report_tested',Path(__file__).with_name('report.py'))
from verify_compiled import validate_manifest


class DeliveryGate(unittest.TestCase):
    def test_compiled_proof_required_and_references_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);batch=root/'batch';old=root/'old';folder=batch/'candidate';folder.mkdir(parents=True)
            source=folder/'main.py';source.write_bytes(b'def run(x): return x\n');digest=sha(source)
            review=dict(source=str(source),source_sha256=digest,status='reviewed_strict_triton',all_branches_reviewed=True,dependencies_reviewed=True)
            save(folder/'review.json',review);save(batch/'candidate-index.json',dict(example=review))
            reference=old/'reference.py';reference.parent.mkdir();reference.write_bytes(b'# reference fixture')
            save(old/'delivery.json',dict(rows=[dict(order=1,operator='example',cann_reference_qualified=True,
                cann_status='合格',reference_kind='CANN_native_single_api',reference_source=str(reference),reference_source_sha256=sha(reference))]))
            state=dict(status='completed',operators=dict(example=dict(status='本轮已复核')))
            row=state['operators']['example']
            for kind in ['runtime','supplementary','cann_supplementary']:
                path=folder/(kind+'.json');save(path,dict(status='passed',source_sha256=digest))
                row[kind]=dict(status='passed',path=str(path))
            for kind in ['pytorch','cann']:
                path=folder/(kind+'.result.json')
                save(path,dict(status='PASSED',num_passed=1,num_workloads=1,per_workload=[dict(phase='timing',latency_ms=1,reference_latency_ms=1)]))
                save(folder/(kind+'.request.json'),dict(implementation=dict(sources=[dict(content=source.read_text(encoding='utf8'))]),
                    settings=dict(warmup_ms=1000,benchmark_ms=100,num_trials=3,timeout_seconds=1800)))
                row[kind]=dict(status='PASSED',score=1.0,passed=1,total=1,path=str(path))
            trace=folder/'trace.json';summary=folder/'summary.json';save(trace,[]);save(summary,dict(status='no_device_tasks'))
            row['device_path']=dict(status='captured',trace=str(trace),summary=str(summary))
            save(batch/'state.json',state)
            with patch.object(report,'BATCH',batch),patch.object(report,'PREVIOUS',old):
                report.main()
                self.assertEqual(read(batch/'delivery.json')['accepted'],0)
                context=read(batch/'strict-implementation-context.json')
                self.assertEqual(context['implementations'],[])
                self.assertEqual(len(context['references']),1)
                artifacts=[]
                for kind in ['ttir','npubin']:
                    path=folder/('kernel.'+kind);path.write_bytes(('synthetic-'+kind).encode())
                    artifacts.append(dict(kind=kind,path=str(path),sha256=sha(path)))
                manifest=folder/'manifest.json';save(manifest,dict(spec=dict(source_sha256=digest),
                    entries=[dict(symbol='k',metadata='test fixture',artifacts=artifacts)]))
                proof=folder/'compiled-proof.json';save(proof,validate_manifest(manifest,source))
                row['compiled_path']=dict(status='verified',path=str(proof),source_sha256=digest)
                save(batch/'state.json',state);report.main()
                self.assertEqual(read(batch/'delivery.json')['accepted'],1)
                context=read(batch/'strict-implementation-context.json')
                self.assertEqual(len(context['implementations']),1)
                self.assertEqual(context['implementations'][0]['source'],str(source))
                self.assertNotEqual(context['references'][0]['source'],str(source))
                (folder/'kernel.npubin').write_bytes(b'altered')
                with self.assertRaises(AssertionError):report.main()


if __name__=='__main__':unittest.main()
