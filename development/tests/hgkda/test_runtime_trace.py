"""连续运行记录保留失败、无修改和修改前证据，不自动追认归因。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('runtime_trace', Path(__file__).resolve().parents[3] / 'hgkda/src/hgkda/runtime/trace.py')
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class RuntimeTraceTests(unittest.TestCase):
    def fixture(self, root):
        workspace = root / 'workspace'
        workspace.mkdir()
        source = workspace / 'candidate.py'
        source.write_bytes(b'original synthetic candidate\n')
        return dict(run_directory=str(root), operator='example', trace_round_id='iteration-1'), source

    def test_exception_and_unchanged_source_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, source = self.fixture(root)
            original = source.read_bytes()
            with self.assertRaises(RuntimeError):
                with trace.action_trace(config, 'generate', 'iteration-1'):
                    raise RuntimeError('model interrupted')
            record = trace.read(root / 'trace/rounds/iteration-1/round.json')
            self.assertEqual(record['status'], 'raised')
            self.assertEqual(record['error_type'], 'RuntimeError')
            self.assertFalse(record['source_changed'])
            self.assertEqual(record['attribution_status'], 'not_recorded')
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(len(list((root / 'trace/events').glob('*.json'))), 2)

    def test_decision_is_bound_before_edit_and_feedback_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, source = self.fixture(root)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            evidence = root / 'simulator.json'
            trace.immutable(evidence, dict(source_sha256=digest, wait_ratio=.7))
            document = dict(source_sha256=digest, evidence=[dict(path='simulator.json', category='simulator',
                role='disambiguation', finding='explicit wait in selected region')], hypothesis='wait limits overlap',
                change='alter loop organization', expected='less wait', validation='repeat device measurement')
            with trace.action_trace(config, 'generate', 'iteration-1'):
                decision = trace.record_decision(config, document)
                source.write_bytes(b'edited synthetic candidate\n')
                trace.immutable(root / 'workspace/diagnosis.json', dict(observed='not measured yet'))
            record = trace.read(root / 'trace/rounds/iteration-1/round.json')
            self.assertTrue(record['source_changed'])
            self.assertEqual(record['input_source']['sha256'], digest)
            self.assertEqual(record['attribution_status'], 'recorded_claim')
            self.assertEqual(len(record['decisions']), 1)
            saved = trace.read(decision['path'])
            self.assertEqual(saved['attribution_status'], 'author_claim_not_causal_proof')
            evidence.write_bytes(b'overwritten measurement')
            frozen=saved['evidence'][0]['reference']
            self.assertEqual(trace.reference(root/frozen['path'],root)['sha256'],frozen['sha256'])
            self.assertEqual(trace.read(root/frozen['path'])['wait_ratio'],.7)
            self.assertEqual(saved['evidence'][0]['original']['path'],'simulator.json')
            (root / 'workspace/diagnosis.json').write_bytes(b'new mutable context')
            for item in record['artifacts']:
                self.assertEqual(trace.reference(root / item['path'], root)['sha256'], item['sha256'])

    def test_stale_evidence_cannot_be_claimed_for_current_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, source = self.fixture(root)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            trace.immutable(root / 'old.json', dict(result=dict(spec=dict(source_sha256='old-source'))))
            document = dict(source_sha256=digest, evidence=[dict(path='old.json', category='simulator', role='trigger')],
                            hypothesis='h', change='c', expected='e', validation='v')
            with self.assertRaisesRegex(AssertionError, 'another source'):
                trace.record_decision(config, document)
            self.assertFalse(list((root / 'trace/decisions').glob('*.json')))

    def test_reused_source_does_not_merge_operation_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.fixture(root)
            with trace.action_trace(config, 'evaluate'):
                pass
            with trace.action_trace(config, 'evaluate'):
                pass
            self.assertEqual(len(list((root / 'trace/sources').glob('*.py'))), 1)
            self.assertEqual(len(list((root / 'trace/events').glob('*.json'))), 4)

    def test_evaluation_feedback_is_not_paired_profiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,source=self.fixture(root)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            trace.immutable(root/'feedback.json',dict(schema_version='hgkda.result/v1',
                result=dict(source_sha256=digest,geo_mean=1.2,ranked_cases=[])))
            document=dict(source_sha256=digest,evidence=[dict(path='feedback.json',
                category='paired_profiling',role='trigger')],hypothesis='h',change='c',expected='e',validation='v')
            with self.assertRaisesRegex(AssertionError,'evaluation evidence'):
                trace.record_decision(config,document)
            document['evidence'][0]['category']='evaluation'
            saved=trace.read(trace.record_decision(config,document)['path'])
            self.assertEqual(saved['evidence'][0]['category'],'evaluation')

    def test_historical_background_keeps_actual_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,source=self.fixture(root)
            trace.immutable(root/'old.json',dict(source_sha256='old-source',score=.9))
            document=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          evidence=[dict(path='old.json',category='other',role='background')],
                          hypothesis='h',change='c',expected='e',validation='v')
            saved=trace.read(trace.record_decision(config,document)['path'])
            self.assertEqual(saved['evidence'][0]['source_sha256'],'old-source')
            document['evidence'][0]['source_sha256']='incorrect-source'
            with self.assertRaisesRegex(AssertionError,'differs from artifact'):
                trace.record_decision(config,document)

    def test_nested_tool_records_stream_prefix_and_freezes_mutable_call_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,_=self.fixture(root)
            call=root/'agent-calls/iteration-1';call.mkdir(parents=True)
            log=call/'agent-events.jsonl';log.write_bytes(b'{"event":1}\n')
            provenance=call/'source-provenance.json';provenance.write_bytes(b'{"input":"initial"}\n')
            with trace.action_trace(config,'decision'):pass
            event=next(v for p in (root/'trace/events').glob('*') if (v:=trace.read(p))['kind']=='action_finished')
            log.write_bytes(log.read_bytes()+b'{"event":2}\n');provenance.write_bytes(b'{"output":"later"}\n')
            for item in event['artifacts']:
                raw=(root/item['path']).read_bytes()
                if 'prefix_bytes' in item:raw=raw[:item['prefix_bytes']]
                self.assertEqual(hashlib.sha256(raw).hexdigest(),item['sha256'])

    def test_equals_argument_form_is_also_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.fixture(root)
            trace.immutable(root / 'config.json', config)
            value = trace.traced_main(lambda: 'returned-value', 'context', ['--config=' + str(root / 'config.json')])
            self.assertEqual(value, 'returned-value')
            self.assertEqual(len(list((root / 'trace/events').glob('*.json'))), 2)


if __name__ == '__main__':
    unittest.main()
