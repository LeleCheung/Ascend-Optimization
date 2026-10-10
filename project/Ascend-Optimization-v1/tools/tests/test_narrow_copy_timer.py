"""核验已安装计时器对 N/A 的解析修复，不修改原模块与耗时统计。"""
import ast
import fnmatch
import importlib.util
import os
from pathlib import Path
import tempfile
from typing import Optional
import unittest

PROJECT = Path(__file__).resolve().parents[2]
HELPER = PROJECT / 'operators/narrow_copy/scripts/ascend-copy-timer.py'
spec = importlib.util.spec_from_file_location('copy_timer', HELPER)
timer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timer)
INSTALLED = Path(os.environ.get('ASCEND_TIMER_SOURCE',
    '/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/testing.py'))


class CopyTimerTests(unittest.TestCase):
    def test_unexpected_structure_rejected(self):
        with self.assertRaises(RuntimeError):
            timer.repair_collector('def changed(): pass\n')

    @unittest.skipUnless(INSTALLED.exists(), '完整 CSV 统计测试在 910B 的已安装计时器上运行')
    def test_dma_and_kernel_preserve_original_task_time(self):
        tree = ast.parse(INSTALLED.read_text())
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in ('ProfilerResultMismatchError', '_collect_prof_result')]
        namespace = {'Optional': Optional, 'os': os, 'fnmatch': fnmatch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(INSTALLED), 'exec'), namespace)
        original = namespace['_collect_prof_result']
        import inspect
        fixed_namespace = dict(namespace)
        exec(compile(timer.repair_collector(inspect.getsource(original)), '<fixed>', 'exec'), fixed_namespace)
        fixed = fixed_namespace['_collect_prof_result']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'task_time_test.csv'
            for name, kind in (('N/A', 'MEMCPY_ASYNC'), ('copy_kernel', 'AI_VECTOR_CORE')):
                lines = ['kernel_name,kernel_type,task_time(us)',
                         'N/A,PROFILING_ENABLE,0',
                         f'{name},{kind},99', f'{name},{kind},0.5', f'{name},{kind},0.7',
                         'N/A,PROFILING_DISABLE,0']
                path.write_text('\n'.join(lines))
                repaired = fixed(temporary, [lambda: None], 1, 2)
                self.assertAlmostEqual(repaired, 0.0006)
                if name == 'N/A':
                    with self.assertRaises(AttributeError):
                        original(temporary, [lambda: None], 1, 2)
                else:
                    self.assertEqual(original(temporary, [lambda: None], 1, 2), repaired)
        self.assertIs(namespace['_collect_prof_result'], original)


if __name__ == '__main__':
    unittest.main()
