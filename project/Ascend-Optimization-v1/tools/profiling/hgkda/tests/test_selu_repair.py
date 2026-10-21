"""SELU扩展语义未完整同身份通过前，不能把额外候选列为交付。"""
import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location('selu_repair', Path(__file__).resolve().parents[1]/'run-selu-repair.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SeluRepairTests(unittest.TestCase):
    def data(self):
        return {'operator':'selu_', 'status':'passed', 'source_sha256':'source',
            'checker_sha256':'checker', 'physical_device':7, 'logical_device':'npu:0',
            'passed':36, 'total':36, 'cases':[
                {'dtype':dtype, 'case':i, 'status':'passed', 'repetitions':2}
                for dtype in ['torch.float16','torch.bfloat16','torch.float32'] for i in range(12)]}

    def test_all_cases_with_repeated_alias_checks_pass(self):
        self.assertTrue(m.valid_semantics(self.data(), 'source', 'checker'))

    def test_partial_duplicate_or_unrepeated_cannot_hide_under_passing_total(self):
        for mutation in [lambda d:d['cases'].pop(),
                         lambda d:d['cases'].__setitem__(-1, d['cases'][0]),
                         lambda d:d['cases'][0].update(repetitions=1),
                         lambda d:d['cases'][0].update(status='failed')]:
            data=self.data();mutation(data)
            self.assertFalse(m.valid_semantics(data,'source','checker'))

    def test_foreign_source_checker_or_card_cannot_reuse(self):
        for field in ['source_sha256','checker_sha256','physical_device','logical_device']:
            data=self.data();data[field]='foreign'
            self.assertFalse(m.valid_semantics(data,'source','checker'))


class LaunchRepairTests(unittest.TestCase):
    def setup_launch(self, *, hit=True, hooks=()):
        adapter = Path(__file__).resolve().parents[1]/'adapters/selu_launch_repair.py'
        tree = ast.parse(adapter.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_selu_launch_repaired')
        namespace = {
            '_driver':SimpleNamespace(active=SimpleNamespace(
                get_current_device=lambda:0, get_current_stream=lambda device:123)),
            '_repair_knobs':SimpleNamespace(runtime=SimpleNamespace(
                debug=False, launch_enter_hook='enter', launch_exit_hook='exit'))}
        def options_hook(options):
            options['compile_marker'] = False
        def binder(*args, **options):
            return {'X':args[0], 'N':options['N']}, [('ptr',16)], options
        key = lambda cache,specialization,options:tuple(sorted(options.items()))
        namespace['_repair_jit'] = SimpleNamespace(
            _flagprism=SimpleNamespace(apply_compile_options=options_hook), compute_cache_key=key)
        compiled = SimpleNamespace(function='function', packed_metadata='packed',
            launch_metadata=Mock(return_value='metadata'), run=Mock())
        expected_key = (('N',1024),('compile_marker',False),('debug',False))
        stock = Mock()
        class Kernel:
            pre_run_hooks=hooks
            debug=None
            device_caches={0:({expected_key:compiled} if hit else {}, {}, None, None, binder)}
            def __getitem__(self, grid):return stock
        code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(adapter), 'exec')
        exec(code, namespace)
        return namespace['_selu_launch_repaired'], Kernel(), stock, compiled

    def test_same_stock_options_hit_cache_and_preserve_hooks_and_actual_arguments(self):
        launch,kernel,stock,compiled=self.setup_launch()
        tensor=object()
        launch(kernel,(tensor,),{'N':1024},(1,1,1))
        stock.assert_not_called()
        compiled.run.assert_called_once_with(1,1,1,123,'function','packed','metadata','enter','exit',tensor,1024)

    def test_cache_miss_uses_one_stock_launch(self):
        launch,kernel,stock,compiled=self.setup_launch(hit=False)
        tensor=object()
        launch(kernel,(tensor,),{'N':1024},(1,1,1))
        stock.assert_called_once_with(tensor,N=1024)
        compiled.run.assert_not_called()

    def test_error_after_possible_submission_cannot_repeat_inplace_computation(self):
        launch,kernel,stock,compiled=self.setup_launch()
        compiled.run.side_effect=RuntimeError('launch result unknown')
        with self.assertRaisesRegex(RuntimeError,'unknown'):
            launch(kernel,(object(),),{'N':1024},(1,1,1))
        compiled.run.assert_called_once()
        stock.assert_not_called()

    def test_prerun_hooks_keep_stock_path_without_retrying_launch_error(self):
        launch,kernel,stock,compiled=self.setup_launch(hooks=('hook',))
        stock.side_effect=TypeError('may already have submitted')
        with self.assertRaisesRegex(TypeError,'submitted'):
            launch(kernel,(object(),),{'N':1024},(1,1,1))
        stock.assert_called_once()
        compiled.run.assert_not_called()

if __name__ == '__main__':unittest.main()
