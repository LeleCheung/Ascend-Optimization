"""Keep accepted IR history intact when a later campaign is imported."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from development.tools.traces.import_ir_progress import import_case, check_measurement_decision, check_ir_lineage


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IRProgressContinuationTest(unittest.TestCase):
    phase = 'midday-whitebox-20261120'

    def test_conditional_measurement_policy_accepts_explicit_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            save(run / 'validation-policy.json', dict(abba='conditional'))
            check_measurement_decision(dict(stability=dict(status='not_requested_conditional_policy',
                reason='Clear formal gain; no unresolved measurement concern')), run)
            with self.assertRaises(AssertionError):
                check_measurement_decision(dict(stability=dict(status='stability_measured',
                    candidate_below_every_original=False)), run)
            with self.assertRaises(AssertionError):
                check_measurement_decision(dict(stability=dict(status='not_requested_conditional_policy',
                    reason='Small contradictory gain', unresolved_measurement_concern=True)), run)

    def test_unretained_repair_branch_has_a_verified_intermediate_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            chain = []
            for name, parent, child in [('explore', 'retained', 'unretained'), ('repair', 'unretained', 'new')]:
                save(run / (name + '-decision.json'), dict(input_files={'kernel.ttir': parent}))
                save(run / (name + '-submission.json'), dict(output_files={'kernel.ttir': child}))
                chain.append(dict(parent_ir_sha256=parent, ir_sha256=child, path='kernel.ttir',
                    decision=name + '-decision.json', submission=name + '-submission.json'))
            selection = dict(implementation_parent_ir_sha256='unretained', ir_lineage=chain)
            check_ir_lineage(selection, run, 'retained', 'new')
            chain[-1]['parent_ir_sha256'] = 'retained'
            with self.assertRaisesRegex(AssertionError, 'ancestry'):
                check_ir_lineage(selection, run, 'retained', 'new')

    def fixture(self, reports):
        run = reports / self.phase / 'example'
        output = reports / 'trajectory'
        events = []
        for label, source, ir, latency in [('native', 'native-source', None, 10),
                                           ('source', 'best-source', None, 8),
                                           ('prior IR gain', 'best-source', 'retained-ir', 6.2)]:
            evaluation = reports / 'history' / label
            self.result(evaluation, latency)
            event = dict(label=label, qualified=True, source_sha256=source,
                measurement_event_id=label, latency_ms=latency,
                replay=dict(result=(evaluation / 'result.json').relative_to(reports).as_posix()))
            if ir:
                event.update(ir_sha256=ir, selected_layer='single_operator_ir')
            events.append(event)
        save(output / 'mainline-presentation.json', dict(operator='example', scope='history', events=events))
        self.result(run / 'evaluations/unchanged-ir', 6)
        self.result(run / 'evaluations/ir-001', 5)
        save(run / 'evaluations/ir-001/supplementary-summary.json', dict(extra=dict(status='passed')))
        save(run / 'evaluations/ir-001/implementation/ir-manifest.json', dict(overrides=[
            dict(original_sha256='retained-ir', sha256='new-ir')]))
        save(run / 'selection.json', dict(status='retained_qualified_ir_improvement',
            invocation='ir-001', source_sha256='best-source',
            formal_result_sha256=digest(run / 'evaluations/ir-001/result.json'),
            stability=dict(candidate_below_every_original=True),
            simulator_replay=dict(replay='fresh simulator evidence')))
        save(run / 'config.json', dict(experiment_goal=dict(target_case='target')))
        save(run / 'retained-baseline-verification.json', dict(status='verified',
            expected_binaries=dict(ast='retained-binary'), observed_binaries=dict(ast='retained-binary'),
            result_sha256=digest(run / 'evaluations/unchanged-ir/result.json')))
        return run, output

    def result(self, evaluation, latency):
        save(evaluation / 'result.json', dict(status='PASSED', num_passed=1, num_workloads=1,
            geo_mean=10 / latency, per_workload=[dict(uuid='target', phase='timing',
                status='PASSED', latency_ms=latency, reference_latency_ms=10, speedup=10 / latency)]))
        save(evaluation / 'request.json', dict(implementation={}, settings=dict(trials=3)))

    def historical_fixture(self, reports):
        run, output = self.fixture(reports)
        data = json.loads((output / 'mainline-presentation.json').read_bytes())
        history = []
        for index, event in enumerate(data['events']):
            evaluation = reports / 'history' / event['label']
            content = event['source_sha256']
            source = evaluation / 'original.py'
            source.write_text(content, encoding='utf8')
            event['source_sha256'] = digest(source)
            event.pop('replay')
            request = evaluation / 'actual-request.json'
            save(request, dict(implementation=dict(sources=[dict(path='upstream.py', content=content)]),
                               settings=dict(trials=3)))
            history.append(dict(event_index=index, source_sha256=digest(source),
                source=source.relative_to(reports).as_posix(),
                result=(evaluation / 'result.json').relative_to(reports).as_posix(),
                result_sha256=digest(evaluation / 'result.json'),
                actual_request=request.relative_to(reports).as_posix(), request_sha256=digest(request)))
        save(output / 'mainline-presentation.json', data)
        review = run / 'history-review.json'
        save(review, dict(history=history))
        selection = json.loads((run / 'selection.json').read_bytes())
        selection.update(source_sha256=data['events'][-1]['source_sha256'],
            historical_measurements=dict(path=review.name, sha256=digest(review)))
        save(run / 'selection.json', selection)
        return run, output, history

    @patch('development.tools.traces.import_ir_progress.write_report')
    def test_append_preserves_prior_ir_and_is_repeatable(self, writer):
        with tempfile.TemporaryDirectory() as temporary:
            reports = Path(temporary)
            _, output = self.fixture(reports)
            args = (reports, 'example', 'trajectory', 'new method', 'new story', 'evidence')
            first = import_case(*args, phase=self.phase)
            self.assertEqual(len(first['events']), 5)
            self.assertEqual(first['events'][2]['ir_sha256'], 'retained-ir')
            self.assertEqual(first['events'][-1]['parent_ir_sha256'], 'retained-ir')
            self.assertEqual(first['events'][-1]['category'], 'sim')
            self.assertEqual(first['events'][-1]['location'], 'ir')
            self.assertEqual(first['events'][-1]['simulator_replay']['replay'], 'fresh simulator evidence')
            self.assertEqual(first['events'][-1]['score'], 2)
            save(output / 'mainline-presentation.json', first)
            self.assertEqual(import_case(*args, phase=self.phase), first)

    @patch('development.tools.traces.import_ir_progress.write_report')
    def test_different_parent_ir_is_rejected(self, writer):
        with tempfile.TemporaryDirectory() as temporary:
            reports = Path(temporary)
            run, _ = self.fixture(reports)
            save(run / 'evaluations/ir-001/implementation/ir-manifest.json', dict(overrides=[
                dict(original_sha256='another-branch', sha256='new-ir')]))
            with self.assertRaisesRegex(AssertionError, 'displayed retained IR'):
                import_case(reports, 'example', 'trajectory', 'method', 'story', 'evidence', phase=self.phase)
            writer.assert_not_called()

    @patch('development.tools.traces.import_ir_progress.write_report')
    def test_historical_adapter_uses_actual_request_and_keeps_native_source(self, writer):
        with tempfile.TemporaryDirectory() as temporary:
            reports = Path(temporary)
            _, _, history = self.historical_fixture(reports)
            # An unused request beside the result has a different timing contract.
            save(reports / 'history/native/request.json', dict(implementation={}, settings=dict(trials=99)))
            result = import_case(reports, 'example', 'trajectory', 'method', 'story', 'evidence', phase=self.phase)
            self.assertEqual(result['events'][0]['source_sha256'], history[0]['source_sha256'])
            self.assertEqual(result['events'][0]['replay']['request'], history[0]['actual_request'])
            self.assertEqual(result['events'][-1]['score'], 2)

    @patch('development.tools.traces.import_ir_progress.write_report')
    def test_mutated_historical_source_is_rejected_before_rendering(self, writer):
        with tempfile.TemporaryDirectory() as temporary:
            reports = Path(temporary)
            _, _, history = self.historical_fixture(reports)
            (reports / history[0]['source']).write_text('changed historical source', encoding='utf8')
            with self.assertRaisesRegex(AssertionError, 'Historical source bytes changed'):
                import_case(reports, 'example', 'trajectory', 'method', 'story', 'evidence', phase=self.phase)
            writer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
