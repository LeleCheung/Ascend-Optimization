"""Protect the distinction between compiler-origin IR and retained-best IR."""
import json
from pathlib import Path
import tempfile
import unittest

from development.tools.research.ir_continue import prepare, restore_retained, sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


class IRContinuationTest(unittest.TestCase):
    def test_next_hypothesis_restores_exact_retained_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = self.parent(root)
            run = root / 'new/operator'
            prepare(parent, 'accepted', run)
            current = run / 'workspace/ir/ast/kernel.ttir'
            retained = current.read_bytes()
            current.write_bytes(b'qualified but slower model candidate')
            previous = run / 'agent-calls/attempt'
            save(previous / 'model-usage.json', {})
            output = previous / 'output-ir/ir/ast/kernel.ttir'
            output.parent.mkdir(parents=True)
            output.write_bytes(current.read_bytes())
            record = restore_retained(run, 'attempt', 'next')
            self.assertEqual(current.read_bytes(), retained)
            self.assertEqual(output.read_bytes(), b'qualified but slower model candidate')
            self.assertEqual(record['status'], 'retained_ir_restored')
            self.assertNotEqual(record['files'][0]['previous_sha256'], record['files'][0]['retained_sha256'])

    def parent(self, root):
        parent = root / 'previous/operator'
        evaluation = parent / 'evaluations/accepted'
        implementation = evaluation / 'implementation'
        implementation.mkdir(parents=True)
        source = implementation / 'frozen.py'
        source.write_bytes(b'qualified unchanged source')
        digest = sha(source)
        retained = implementation / 'ir/ast/kernel.ttir'
        retained.parent.mkdir(parents=True)
        retained.write_bytes(b'qualified retained IR, with earlier optimization')
        save(evaluation / 'result.json', dict(status='PASSED', num_passed=2, num_workloads=2,
             per_workload=[dict(uuid='target', phase='timing', latency_ms=1.0)]))
        save(evaluation / 'supplementary-summary.json', dict(extra=dict(status='passed')))
        save(implementation / 'ir-manifest.json', dict(source_sha256=digest, compiler_identity={},
             overrides=[dict(path='ir/ast/kernel.ttir', filename='kernel.ttir', ast_sha256='ast',
                             sha256=sha(retained), original_sha256='compiler-origin-IR')]))
        save(parent / 'selection.json', dict(status='retained_qualified_ir_improvement',
             invocation='accepted', formal_result_sha256=sha(evaluation / 'result.json')))
        save(parent / 'config.json', dict(operator='operator', source_sha256=digest,
             experiment_goal=dict(target_case='target')))
        save(parent / 'model-settings.json', {})
        observation = parent / 'observations/accepted'
        save(observation / 'simulator/ir-summary.json', dict(source_sha256=digest,
             reports=[dict(core='core0', timeline={}, absolute_unit_validation={})]))
        save(observation / 'paired/comparison.json', dict(bound_to='retained binary'))
        artifact = evaluation / 'compiled/kernel.bcmlir'
        artifact.parent.mkdir()
        artifact.write_bytes(b'lowered retained IR')
        receipt = evaluation / 'compiler-receipts/receipt.jsonl'
        save(receipt, dict(override_applied=True, ast_sha256='ast', symbol='kernel',
             artifacts=[dict(path=str(artifact), kind='bcmlir', sha256=sha(artifact))]))
        return parent

    def test_continuation_uses_retained_bytes_and_preserves_compiler_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = self.parent(root)
            original = parent / 'evaluations/accepted/implementation/ir/ast/kernel.ttir'
            before = original.read_bytes()
            run = root / 'new/operator'
            prepare(parent, 'accepted', run)
            inventory = json.loads((run / 'workspace/ir-inventory.json').read_bytes())
            item = inventory['entries'][0]
            self.assertEqual((run / 'workspace/ir/ast/kernel.ttir').read_bytes(), before)
            self.assertEqual(item['original_sha256'], sha(original))
            self.assertEqual(item['compiler_original_sha256'], 'compiler-origin-IR')
            self.assertEqual(Path(item['original']).read_bytes(), before)
            self.assertEqual(original.read_bytes(), before)

    def test_failed_or_unretained_parent_is_rejected_before_creation(self):
        for kind in ['failed', 'unretained']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                parent = self.parent(root)
                if kind == 'failed':
                    save(parent / 'evaluations/accepted/result.json', dict(status='FAILED', num_passed=1, num_workloads=2))
                else:
                    selection = json.loads((parent / 'selection.json').read_bytes())
                    selection['status'] = 'candidate_only'
                    save(parent / 'selection.json', selection)
                run = root / 'new/operator'
                with self.assertRaises(AssertionError):
                    prepare(parent, 'accepted', run)
                self.assertFalse(run.exists())

    def test_observation_for_different_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = self.parent(root)
            save(parent / 'observations/accepted/simulator/ir-summary.json', dict(source_sha256='other'))
            run = root / 'new/operator'
            with self.assertRaises(AssertionError):
                prepare(parent, 'accepted', run)
            self.assertFalse(run.exists())

    def test_explicit_unobserved_original_setup_never_fabricates_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = self.parent(root)
            source = parent / 'evaluations/accepted'
            source.rename(parent / 'evaluations/unchanged-ir')
            # References in the fake receipt follow the renamed fixture.
            receipt = parent / 'evaluations/unchanged-ir/compiler-receipts/receipt.jsonl'
            d = json.loads(receipt.read_bytes())
            d['artifacts'][0]['path'] = str(parent / 'evaluations/unchanged-ir/compiled/kernel.bcmlir')
            save(receipt, d)
            manifest = parent / 'evaluations/unchanged-ir/implementation/ir-manifest.json'
            d = json.loads(manifest.read_bytes())
            d['overrides'][0]['original_sha256'] = d['overrides'][0]['sha256']
            save(manifest, d)
            run = root / 'new/operator'
            prepare(parent, 'unchanged-ir', run, allow_unobserved=True)
            self.assertFalse((run / 'workspace/hotspots.json').exists())
            self.assertFalse((run / 'workspace/timeline.json').exists())
            self.assertFalse(json.loads((run / 'workspace/context.json').read_bytes())['evidence_ready'])
            self.assertEqual(json.loads((run / 'status.json').read_bytes())['status'], 'awaiting_baseline_evidence')


if __name__ == '__main__':
    unittest.main()
