"""Check optional evidence context and pre-edit reads on the Linux research host."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/research'))
from ir_campaign import read, save, sha
import gqa_model_start
from gqa_review_bindings import review, evidence_read


class GQAEvidenceModes(unittest.TestCase):
    def test_successful_read_only_shell_commands_are_recognized(self):
        for command in ['cat current-profiling.json', "sed -n '60,140p' actual-ir/kernel.bcmlir",
                        'head -120 actual-ir/kernel.ttir']:
            self.assertTrue(evidence_read(dict(name='Bash', input=dict(command=command)), command.split()[-1]))
        self.assertTrue(evidence_read(dict(name='Bash', input=dict(
            command='sha256sum candidate.py starter.py; echo "---"; cat candidate.py')), 'candidate.py'))
        self.assertTrue(evidence_read(dict(name='Bash', input=dict(
            command='echo "profile" && cat current-profiling.json && ls actual-ir/')), 'current-profiling.json'))
        self.assertTrue(evidence_read(dict(name='Bash', input=dict(
            command='grep -n "tt.dot\\|tt.load" actual-ir/kernel.ttir | head -40')), '.ttir'))
        batch=dict(name='Bash',input=dict(command='cat repair-feedback.json; echo done'))
        self.assertTrue(evidence_read(batch,'repair-feedback.json','exact feedback','exact feedback\ndone'))
        self.assertFalse(evidence_read(batch,'repair-feedback.json','exact feedback','missing file\ndone'))
        for command in ['echo current-profiling.json', 'cat current-profiling.json > output',
                        'cat current-profiling.json; echo done',
                        'rm marker; cat current-profiling.json',
                        "sed -i '1,5p' current-profiling.json",
                        "sed -n 'w output' current-profiling.json",
                        'cat current-profiling.json || echo fallback']:
            self.assertFalse(evidence_read(dict(name='Bash', input=dict(command=command)), 'current-profiling.json'))

    def test_source_context_archives_stale_simulator_without_supplying_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, template, package = (root / name for name in ['run', 'template', 'package'])
            workspace = run / 'workspace'
            for folder in [workspace, template, package / 'skills',
                           run / 'capture-parent/candidate/actual-ir', run / 'paired-parent',
                           run / 'parent-validation']:
                folder.mkdir(parents=True)
            (workspace / 'candidate.py').write_text('def run(x): return x\n')
            digest = sha(workspace / 'candidate.py')
            golden = root / 'golden.py'
            golden.write_text('# fixed reference\n')
            save(run / 'contract.json', dict(golden=str(golden)))
            save(run / 'config.json', dict(required_simulator_evidence=dict(path='stale')))
            save(template / 'config.json', dict(package=str(package), method_manifest='fixed'))
            save(template / 'model-settings.json', {})
            save(run / 'observation-status.json', dict(status='parent_dual_evidence_ready', source_sha256=digest))
            save(run / 'paired-parent/comparison.json', dict(candidate=dict(basic={}, tables={}),
                                                          reference=dict(basic={}, tables={})))
            result = dict(cases=[dict(case_id=4, candidate_timing=dict(full_call_us=472.46))])
            save(run / 'parent-validation/results.json', result)
            save(run / 'parent-results.json', result)
            (run / 'capture-parent/candidate/actual-ir/kernel.ttir').write_text('actual IR')
            archived_only = ['simulator-evidence.json', 'simulator-evidence-full.json',
                             'repair-feedback.json', 'host-repair-analysis.json']
            for name in archived_only:
                save(workspace / name, dict(stale=True))
            args = ['start', '--run', str(run), '--template', str(template), '--invocation', 'next',
                    '--compact-evidence', '--evidence-mode', 'profiling-source-ir']
            with patch.object(sys, 'argv', args), patch.object(gqa_model_start.subprocess, 'Popen') as launch:
                launch.return_value.pid = 1
                gqa_model_start.main()
            self.assertNotIn('required_simulator_evidence', read(run / 'config.json'))
            self.assertIsNone(read(workspace / 'context-brief.json')['current_simulator'])
            self.assertTrue(read(workspace / 'context-brief.json')['prior_host_simulator_knowledge'])
            for name in archived_only:
                self.assertFalse((workspace / name).exists())
                self.assertTrue((run / 'evidence-history/next' / name).exists())
            prompt = (run / 'next-prompt.md').read_text(encoding='utf8')
            self.assertNotIn('simulator-evidence.json', prompt)
            self.assertIn('当前profiling、相关TTIR与BCMLIR', prompt)

    def test_source_mode_requires_ir_reads_and_does_not_require_repair_feedback(self):
        for missing_ir in [False, True]:
            with self.subTest(missing_ir=missing_ir), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                call = run / 'agent-calls/next'
                call.mkdir(parents=True)
                (run / 'workspace').mkdir()
                (call / 'input-source.py').write_text('old')
                for path in [call / 'output-source.py', run / 'workspace/candidate.py']:
                    path.write_text('new')
                save(call / 'model-usage.json', {})
                save(call / 'config.json', dict(evidence_mode='profiling-source-ir'))
                decision = call / 'decision.json'
                save(decision, {})
                events = []
                def event(block):
                    events.append(dict(message=dict(content=[block])))
                reads = ['current-profiling.json', '.ttir', 'candidate.py'] + ([] if missing_ir else ['.bcmlir'])
                for i, token in enumerate(reads):
                    event(dict(type='tool_use', name='Read', id=str(i), input=dict(file_path=token)))
                    event(dict(type='tool_result', tool_use_id=str(i), content='{}'))
                event(dict(type='tool_use', name='Bash', id='failed-decision',
                           input=dict(command='bridge decision --record diagnosis.json')))
                event(dict(type='tool_result', tool_use_id='failed-decision', content='Command failed before recording'))
                event(dict(type='tool_use', name='Bash', id='decision',
                           input=dict(command='bridge decision --record diagnosis.json')))
                event(dict(type='tool_result', tool_use_id='decision', content=json.dumps(dict(
                    status='recorded', decision_id='one', path=str(decision)))))
                event(dict(type='tool_use', name='Edit', id='edit', input=dict(file_path='/workspace/candidate.py')))
                (call / 'agent-events.jsonl').write_text('\n'.join(map(json.dumps, events)))
                if missing_ir:
                    with self.assertRaisesRegex(AssertionError, '.bcmlir'):
                        review(run, 'next')
                else:
                    self.assertEqual(review(run, 'next')['evidence_mode'], 'profiling-source-ir')


if __name__ == '__main__':
    unittest.main()
