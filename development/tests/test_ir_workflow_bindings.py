"""Guard against attributing an IR experiment to changed source or mutable evidence."""
import hashlib
import json
import os
from collections import namedtuple
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from development.tools.research.ir_model import bridge, checked, files
from development.tools.research.ir_lowering_review import comparison_ir
from development.tools.research.ir_binary_compare import formal_receipts
from development.tools.research.ir_override_runtime import checked_compiler_binding, install, normalized, option_identity, verify_receipts


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


class FrozenIRWorkflowTest(unittest.TestCase):
    def test_formal_receipt_scope_excludes_later_supplementary_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / 'compiler-receipts/formal.jsonl'
            save(receipt, dict(ast_sha256='formal'))
            save(root / 'compiler-receipts/supplement.jsonl', dict(ast_sha256='extra'))
            save(root / 'result.json', dict(status='PASSED'))
            save(root / 'compiler-verification.json', dict(receipts=1))
            os.utime(receipt, ns=(1_000_000_000, 1_000_000_000))
            os.utime(root / 'result.json', ns=(2_000_000_000, 2_000_000_000))
            os.utime(root / 'compiler-receipts/supplement.jsonl', ns=(3_000_000_000, 3_000_000_000))
            selected, scope = formal_receipts(root)
            self.assertEqual(selected, [receipt])
            self.assertEqual(scope['excluded_later_files'], ['compiler-receipts/supplement.jsonl'])
            save(root / 'formal-compiler-receipts.json', dict(files=scope['files']))
            selected, scope = formal_receipts(root)
            self.assertEqual(selected, [receipt.resolve()])
            self.assertEqual(scope['method'], 'recorded_before_supplements')
            receipt.write_bytes(b'changed compiler receipt')
            with self.assertRaisesRegex(AssertionError, 'receipt bytes changed'):
                formal_receipts(root)

    def test_compile_repair_compares_against_frozen_original_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captured = root / 'captured.ttir'
            captured.write_bytes(b'valid frozen original')
            call = root / 'repair'
            (call / 'input-ir/ir').mkdir(parents=True)
            (call / 'input-ir/ir/kernel.ttir').write_bytes(b'invalid exploration')
            entry = dict(path='ir/kernel.ttir', original=str(captured),
                         original_sha256=hashlib.sha256(captured.read_bytes()).hexdigest())
            self.assertEqual(comparison_ir(entry, call, 'captured_original'), captured)
            self.assertEqual(comparison_ir(entry, call, 'invocation_input'),
                             call / 'input-ir/ir/kernel.ttir')
            captured.write_bytes(b'mutated original')
            with self.assertRaisesRegex(AssertionError, 'identity changed'):
                comparison_ir(entry, call, 'captured_original')

    def prepare_decision(self, run):
        workspace = run / 'workspace'
        save(workspace / 'ir-inventory.json', dict(entries=[dict(path='ir/kernel.ttir')]))
        (workspace / 'ir').mkdir()
        (workspace / 'ir/kernel.ttir').write_text('original', encoding='utf8')
        (workspace / 'frozen.py').write_text('frozen', encoding='utf8')
        save(workspace / 'profiling.json', {})
        save(workspace / 'context.json', dict(evidence_mode='full'))
        digest = hashlib.sha256(b'frozen').hexdigest()
        save(workspace / 'diagnosis.json', dict(source_sha256=digest, selected_layer='single_operator_ir',
            hypothesis='keep original', change='no change', expected='same behavior', validation='identity',
            evidence=[dict(path='profiling.json', category=category) for category in
                ['paired_profiling', 'simulator', 'source_ir']]))
        save(run / 'config.json', dict(source_sha256=digest))
        save(run / 'agent-calls/one/input-manifest.json', files(workspace))
        return workspace

    def test_no_change_handoff_preserves_the_registered_judgment(self):
        with tempfile.TemporaryDirectory() as temporary, patch('builtins.print'):
            run = Path(temporary)
            workspace = self.prepare_decision(run)
            bridge(run, 'one', 'decision')
            bridge(run, 'one', 'no-change')
            self.assertTrue((run / 'agent-calls/one/no-change.json').exists())
            self.assertEqual((workspace / 'ir/kernel.ttir').read_text(), 'original')
            (workspace / 'ir/kernel.ttir').write_text('new IR', encoding='utf8')
            with self.assertRaisesRegex(AssertionError, 'requires the original IR'):
                bridge(run, 'one', 'no-change')

    def test_decision_cannot_be_rewritten_after_editing_ir(self):
        with tempfile.TemporaryDirectory() as temporary, patch('builtins.print'):
            run = Path(temporary)
            workspace = self.prepare_decision(run)
            bridge(run, 'one', 'decision')
            (workspace / 'ir/kernel.ttir').write_text('different mechanism', encoding='utf8')
            diagnosis = json.loads((workspace / 'diagnosis.json').read_text())
            diagnosis['change'] = 'a newly invented mechanism'
            save(workspace / 'diagnosis.json', diagnosis)
            with self.assertRaisesRegex(AssertionError, 'Diagnosis changed after decision'):
                bridge(run, 'one', 'submit')
            self.assertFalse((run / 'agent-calls/one/submission.json').exists())

    def test_simulator_metadata_alias_requires_identical_binary_and_other_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / 'kernel.npubin'
            binary.write_bytes(b'real-device-binary')
            target_type = namedtuple('GPUTarget', ['backend', 'arch', 'warp_size'])
            original = target_type('npu', 'Ascend910B4-1', 0)
            simulator = target_type('npu', 'Ascend910B4', 0)
            item = dict(ast_sha256='ast', target=normalized(original),
                compiler_options=dict(arch=original.arch, num_warps=2))
            metadata = dict(arch=simulator.arch, num_warps=2, target=simulator)
            group = {'kernel.npubin': str(binary)}
            with self.assertRaisesRegex(AssertionError, 'differ from captured'):
                checked_compiler_binding(item, metadata, group)
            alias = dict(expected_arch=original.arch, observed_arch=simulator.arch,
                binary_sha256={'ast': hashlib.sha256(binary.read_bytes()).hexdigest()})
            _, qualification = checked_compiler_binding(item, metadata, group, alias)
            self.assertEqual(qualification['identical_binary_sha256'], alias['binary_sha256']['ast'])
            with self.assertRaisesRegex(AssertionError, 'additional compiler options'):
                checked_compiler_binding(item, dict(metadata, num_warps=4), group, alias)
            binary.write_bytes(b'different-simulator-binary')
            with self.assertRaisesRegex(AssertionError, 'binary differs'):
                checked_compiler_binding(item, metadata, group, alias)

    def test_cached_option_arrays_keep_same_values_and_distinct_launch_settings(self):
        captured = normalized(dict(cluster_dims=(1, 1, 1), precisions=('ieee', 'hf32'), num_warps=2))
        cached = dict(cluster_dims=(1, 1, 1), precisions=['ieee', 'hf32'], num_warps=2)
        self.assertEqual(option_identity(captured), option_identity(cached))
        self.assertNotEqual(option_identity(captured), option_identity(dict(cached, num_warps=4)))

    def test_passing_execution_requires_each_advertised_ir_to_be_consumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = dict(source_sha256='frozen', ast_sha256='target', override_applied=False)
            path = root / 'receipts.jsonl'
            path.write_text(json.dumps(row) + '\n', encoding='utf8')
            manifest = dict(source_sha256='frozen', overrides=[dict(ast_sha256='target', sha256='ir')])
            with self.assertRaisesRegex(AssertionError, 'never consumed'):
                verify_receipts(root, manifest)
            row.update(override_applied=True, ir_sha256='ir', parsed_ir_sha256='parsed', cached_ir_sha256='parsed')
            path.write_text(json.dumps(row) + '\n', encoding='utf8')
            self.assertEqual(verify_receipts(root, manifest)['observed_overrides'], 1)

    def test_parser_round_trip_binds_raw_input_and_cached_stage_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'frozen.py').write_bytes(b'frozen')
            (root / 'kernel.ttir').write_bytes(b'named SSA input')
            source_sha = hashlib.sha256(b'frozen').hexdigest()
            raw_sha = hashlib.sha256(b'named SSA input').hexdigest()
            manifest = dict(source='frozen.py', source_sha256=source_sha, compiler_identity={},
                cache_root=str(root / 'cache'), receipt_directory=str(root / 'receipts'),
                overrides=[dict(filename='kernel.ttir', path='kernel.ttir', ast_sha256='0' * 64,
                    sha256=raw_sha, original_sha256=raw_sha, compiler_options={}, target='fake')])
            save(root / 'manifest.json', manifest)
            triton = ModuleType('triton')
            package = ModuleType('triton.compiler')
            compiler = ModuleType('triton.compiler.compiler')
            compiler.parse = lambda *args: 'canonical SSA input'
            package.compiler = compiler
            triton.compiler = package
            triton.knobs = SimpleNamespace(cache=SimpleNamespace(), compilation=SimpleNamespace(listener=None))
            modules = {'triton': triton, 'triton.compiler': package, 'triton.compiler.compiler': compiler}
            with patch.dict('sys.modules', modules), patch.dict(os.environ, {}, clear=True), patch(
                    'development.tools.research.ir_override_runtime.compiler_identity', return_value={}), patch(
                    'development.tools.research.ir_override_runtime.raw_pointer_profile_adapter', return_value='test-adapter'):
                install(root / 'manifest.json')
                override = next(Path(triton.knobs.cache.override_dir).rglob('kernel.ttir'))
                parsed = compiler.parse(str(override), 'ttir', object())
                cached = root / 'cached.ttir'
                cached.write_text(parsed, encoding='utf8')
                src = SimpleNamespace(hash=lambda: '0' * 64, name='kernel')
                triton.knobs.compilation.listener(src, dict(hash='cache-key', target='fake'),
                    {'kernel.ttir': str(cached)}, None, False)
                receipt = json.loads(next((root / 'receipts').glob('*.jsonl')).read_text())
                self.assertEqual(receipt['ir_sha256'], raw_sha)
                self.assertEqual(receipt['cached_ir_sha256'], receipt['parsed_ir_sha256'])
                self.assertNotEqual(receipt['cached_ir_sha256'], receipt['ir_sha256'])
                cached.write_text('unexpected compiler stage', encoding='utf8')
                with self.assertRaisesRegex(AssertionError, 'differs from the cached'):
                    triton.knobs.compilation.listener(src, dict(hash='cache-key', target='fake'),
                        {'kernel.ttir': str(cached)}, None, True)

    def test_compiler_tuple_keys_survive_json_and_keep_distinct_identity(self):
        constants = {(0,): 'fp32', (1, 2): 128, '(0,)': 'literal-key'}
        first = normalized(constants)
        self.assertEqual(first, json.loads(json.dumps(first)))
        self.assertEqual(first, normalized(dict(reversed(list(constants.items())))))
        self.assertEqual(len(first['__mapping__']), 3)
        self.assertNotEqual(first, normalized({'(0,)': 'fp32', (1, 2): 128}))

    def test_changed_source_is_rejected_before_importing_device_compiler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'frozen.py').write_text('changed source', encoding='utf8')
            save(root / 'manifest.json', dict(source='frozen.py', source_sha256='0' * 64))
            with self.assertRaisesRegex(AssertionError, 'Frozen source changed'):
                install(root / 'manifest.json')

    def test_ir_edits_allowed_while_evidence_and_inventory_are_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            workspace = run / 'workspace'
            (workspace / 'ir').mkdir(parents=True)
            (workspace / 'frozen.py').write_text('frozen', encoding='utf8')
            (workspace / 'ir/kernel.ttir').write_text('original IR', encoding='utf8')
            (workspace / 'profiling.json').write_text('{}', encoding='utf8')
            save(workspace / 'diagnosis.json', {})
            save(workspace / 'ir-inventory.json', dict(entries=[dict(path='ir/kernel.ttir')]))
            save(run / 'config.json', dict(source_sha256=hashlib.sha256(b'frozen').hexdigest()))
            save(run / 'agent-calls/one/input-manifest.json', files(workspace))
            (workspace / 'ir/kernel.ttir').write_text('model authored IR', encoding='utf8')
            checked(run, 'one')
            (workspace / 'profiling.json').write_text('{"fabricated":true}', encoding='utf8')
            with self.assertRaisesRegex(AssertionError, 'Frozen input changed'):
                checked(run, 'one')


if __name__ == '__main__':
    unittest.main()
