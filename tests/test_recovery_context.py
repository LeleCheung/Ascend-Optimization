"""启动诊断拒绝其他源码的反馈和轨迹，并排除重复派生时间。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec=importlib.util.spec_from_file_location('recovery_context',Path(__file__).resolve().parents[1]/'tools/skill_validation/recovery_context.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class RecoveryContextTests(unittest.TestCase):
    def fixture(self,root):
        workspace=root/'workspace';workspace.mkdir()
        source=workspace/'starter.py';source.write_text('def run(A): return A\n')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        probe=root/'probe';probe.mkdir()
        def save(path,value):path.write_text(json.dumps(value))
        save(probe/'probe.json',dict(source_sha256=digest,compiled=[],measurements=[]))
        save(workspace/'performance-feedback.json',dict(source_sha256=digest))
        trace=probe/'trace.json'
        save(trace,[dict(ph='X',name='kernel',dur=2,args={'Task Type':'KERNEL_AIVEC'}),
                    dict(ph='X',name='Computing',dur=2,args={})])
        save(probe/'traces.json',[dict(case='small',side='candidate',path=str(trace),
             sha256=hashlib.sha256(trace.read_bytes()).hexdigest())])
        return dict(run_directory=str(root),probe_directory=str(probe),trace_directory=str(probe))

    def test_only_actual_task_time_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.fixture(Path(tmp));brief=module.build(config)
            self.assertEqual(brief['device_tasks'][0]['count'],1)
            self.assertEqual(brief['device_tasks'][0]['median_us'],2)
            self.assertEqual(brief['knowledge_retrieval']['exact'],1)

    def test_stale_feedback_or_trace_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.fixture(root)
            (root/'workspace/performance-feedback.json').write_text(json.dumps(dict(source_sha256='different')))
            with self.assertRaises(AssertionError):module.build(config)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.fixture(root)
            trace=root/'trace-from-other-source';trace.mkdir()
            (trace/'probe.json').write_text(json.dumps(dict(source_sha256='different')))
            config['trace_directory']=str(trace)
            with self.assertRaises(AssertionError):module.build(config)


if __name__=='__main__':unittest.main()
