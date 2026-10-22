"""控制器配置不能被命令参数覆盖，恢复时不得重复启动已有worker或正式评测。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('paired_kg_resume', Path(__file__).resolve().parents[1]/'run-paired-kg.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QueueResumeTest(unittest.TestCase):
    def test_resume_finished_worker_reuses_semantics_and_reaches_dense_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root/'batch'
            batch.mkdir()
            runtime = root/'runtime'
            runtime.mkdir()
            for op in ['log10_', 'selu_']:
                workspace = runtime/'kg-runs'/(op+'-batch-paired-hgkda')
                (workspace/'.kernelgen').mkdir(parents=True)
                (workspace/'.kernelgen/run-process.json').write_text('{"finished_at":"done"}')
                role = workspace/'kernel-profile-analyzer.md'
                role.write_text('hgkda-paired-v1')
                output = batch/'results'/op/'paired-HG-KDA'
                (output/'export').mkdir(parents=True)
                source = output/'export/main.py'
                source.write_text('def run(A): return A\n')
                (output/'export/provenance.json').write_text('{}')
                (output/'native-evidence').mkdir()
                (output/'native-evidence/archive-manifest.json').write_text('{}')
                digest = module.hashlib.sha256(source.read_bytes()).hexdigest()
                (output/'semantics.json').write_text(json.dumps(dict(status='passed', source_sha256=digest)))
                (output/'dense-semantics.json').write_text(json.dumps(dict(status='passed', source_sha256=digest)))
            (batch/'paired-evidence.json').write_text('{}')
            (batch/'kg-paired-state.json').write_text(json.dumps(dict(status='needs_attention', results={}, error='old')))
            records = {op: dict(seed='results/'+op+'/paired-HG-KDA/export/main.py') for op in ['log10_', 'selu_']}
            queue = SimpleNamespace(batch=batch, runtime=runtime, project=root/'project', records=records,
                py='python3', idle=lambda:None, run=lambda *args: self.fail('不应重新启动worker或补测已有门禁'),
                evaluate=lambda *args: dict(status='passed', scores=[.9]))
            fake = SimpleNamespace(Queue=lambda *args:queue,
                save=lambda p,v:p.write_text(json.dumps(v)))
            original_path = module.Path
            class RemotePath:
                def __new__(cls, value):
                    if value == '/data/hanle/ascend-optimization':
                        return root
                    return original_path(value)
            # 覆盖层使用控制器root/runtime/batch/sources；显式创建同一位置。
            target = root/'runtime'/'batch'/'sources'
            target.mkdir(parents=True)
            (target/'hgkda-overlay.json').write_text('{}')
            (root/'runs').mkdir()
            loader = SimpleNamespace(exec_module=lambda m:None)
            with patch.object(module, 'Path', RemotePath), \
                 patch.object(module.importlib.util, 'spec_from_file_location', return_value=SimpleNamespace(loader=loader)), \
                 patch.object(module.importlib.util, 'module_from_spec', return_value=fake), \
                 patch.dict(sys.modules, {'fcntl': SimpleNamespace(flock=lambda *args:None, LOCK_EX=1)}), \
                 patch.object(sys, 'argv', ['run-paired-kg.py', '--batch', 'batch', '--seed-plan', '--dense', '--resume']):
                module.main()
            state = json.loads((batch/'kg-paired-state.json').read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(set(state['results']), {'log10_', 'selu_'})
            self.assertTrue(list(batch.glob('kg-state-before-resume-*.json')))


if __name__ == '__main__':
    unittest.main()
