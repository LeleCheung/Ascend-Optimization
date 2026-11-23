import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hgkda.runtime.launch_replay import freeze, validate, validate_replay_ir


class LaunchReplayTests(unittest.TestCase):
    def test_source_only_and_ir_modes_cannot_be_mixed(self):
        validate_replay_ir({}, {})
        for data,spec in [({'ir_input':{'manifest':'unused'}},{}),
                          ({},{'ir_input':{'manifest':'unused'}})]:
            with self.assertRaisesRegex(AssertionError,'IR mode differs'):
                validate_replay_ir(data,spec)

    def test_target_ir_may_change_while_preceding_stage_ir_stays_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def binding(name,auxiliary,target):
                path=root/name
                path.write_text(json.dumps({'overrides':[
                    dict(symbol='prepare',ast_sha256='prep-ast',sha256=auxiliary),
                    dict(symbol='attention',ast_sha256='target-ast',sha256=target)]}))
                return dict(manifest=str(path),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            original=binding('original.json','fixed','original-target')
            changed=binding('changed.json','fixed','modified-target')
            validate_replay_ir(dict(kernel='attention',ir_input=original),dict(ir_input=changed))
            wrong=binding('wrong.json','changed-auxiliary','modified-target')
            with self.assertRaisesRegex(AssertionError,'Preceding-stage IR differs'):
                validate_replay_ir(dict(kernel='attention',ir_input=original),dict(ir_input=wrong))

    def test_replay_helper_can_load_beside_direct_driver_script(self):
        from hgkda.runtime import paired_driver
        helper=paired_driver.imported('_direct_replay_test',Path(paired_driver.__file__).with_name('launch_replay.py'))
        self.assertTrue(callable(helper.load_call))
        self.assertTrue(callable(helper.validate))

    def test_frozen_snapshot_binding_and_tensor_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            tensor=root/'data.pt';tensor.write_bytes(b'exact diagnostic tensors')
            case={'name':'fixed-case'}
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(source_sha256='source',case=case,kernel='target',
                tensors='data.pt',tensors_sha256=hashlib.sha256(tensor.read_bytes()).hexdigest())))
            binding=freeze(manifest,root/'frozen','source',case,'target')
            self.assertEqual(validate(binding['manifest'],'source',case,'target')[1].read_bytes(),tensor.read_bytes())
            for source,check_case,kernel in [('other',case,'target'),('source',{},'target'),('source',case,'other')]:
                with self.assertRaises(AssertionError):validate(manifest,source,check_case,kernel)
            tensor.write_bytes(b'changed')
            with self.assertRaisesRegex(AssertionError,'bytes changed'):validate(manifest,'source',case,'target')

    def test_external_tensor_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=root/'bundle';bundle.mkdir()
            manifest=bundle/'manifest.json'
            manifest.write_text(json.dumps(dict(source_sha256='source',case={},kernel='target',
                tensors='../outside.pt',tensors_sha256='unused')))
            with self.assertRaisesRegex(AssertionError,'escapes'):validate(manifest,'source',{},'target')
