"""交付索引必须拒绝补测不合格、合同漂移和源码篡改，且支持跨机器路径重定位。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from export_best import REMOTE_PROJECT, build, resolve, verify_index


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


class ExportTests(unittest.TestCase):
    def fixture(self, project, extra_passed=True, reference='npu_paths_observed'):
        batch = project / 'operators/campaigns/test'
        batch.mkdir(parents=True)
        source = batch / 'candidate.py'
        source.write_text('def run(x): return x\n', encoding='utf8')
        result = batch / 'formal.result.json'
        write(result, dict(status='PASSED', num_passed=2, num_workloads=2, geo_mean=1.2))
        write(batch / 'formal.request.json', dict(implementation=dict(sources=[dict(content=source.read_text())])))
        extra = batch / 'extra.json'
        write(extra, dict(operator='x', status='passed' if extra_passed else 'failed',
                         passed=2 if extra_passed else 1, total=2, source_sha256=sha(source)))
        leaf = batch / 'catalog/x'
        write(leaf / 'definition.json', dict(name='x'))
        # 使用910B绝对路径构造输入，在另一根目录重定位并核验。
        def remote(path):
            return REMOTE_PROJECT + '/' + path.relative_to(project).as_posix()
        record = dict(source=remote(source), source_sha256=sha(source), result=remote(result), extra_checks=remote(extra))
        write(batch / 'next-state.json', dict(status='running', operators=dict(x=dict(
            baseline=dict(source_sha256='original'), best=record, reference_path_status=reference))))
        write(batch / 'targets.json', dict(targets=[dict(order=1, operator='x')]))
        write(batch / 'software-lock.json', dict(fixed='test'))
        write(batch / 'plan.json', dict(records=[dict(operator='x', catalog_leaf='catalog/x',
              contract_sha256={'definition.json': sha(leaf / 'definition.json')})]))
        return batch, source

    def test_relocates_and_detects_changed_source(self):
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            batch, source = self.fixture(project)
            index = build(project, batch)
            self.assertEqual(index['counts'], {'new_threshold_passed': 1})
            self.assertEqual(verify_index(project, index)['status'], 'verified')
            self.assertFalse(Path(index['operators'][0]['source']).is_absolute())
            source.write_text('def run(x): return x + 1\n')
            with self.assertRaisesRegex(ValueError, '索引文件已变化'):
                verify_index(project, index)

    def test_supplement_failure_not_exported(self):
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            batch, _ = self.fixture(project, extra_passed=False)
            index = build(project, batch)
            self.assertEqual(index['counts'], {'not_qualified': 1})
            self.assertIsNone(index['operators'][0]['source'])

    def test_reference_with_cpu_fallback_separate(self):
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            batch, _ = self.fixture(project, reference='reference_requires_separate_review')
            index = build(project, batch)
            self.assertEqual(index['counts'], {'reference_requires_separate_review': 1})
            self.assertNotIn('new_threshold_passed', index['counts'])

    def test_contract_change_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            batch, _ = self.fixture(project)
            write(batch / 'catalog/x/definition.json', dict(name='changed'))
            with self.assertRaisesRegex(ValueError, '证据SHA不符'):
                build(project, batch)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            with self.assertRaises(ValueError):
                resolve(project, '../outside.py')
            with self.assertRaises(ValueError):
                resolve(project, REMOTE_PROJECT + '/../outside.py')


if __name__ == '__main__':
    unittest.main()
