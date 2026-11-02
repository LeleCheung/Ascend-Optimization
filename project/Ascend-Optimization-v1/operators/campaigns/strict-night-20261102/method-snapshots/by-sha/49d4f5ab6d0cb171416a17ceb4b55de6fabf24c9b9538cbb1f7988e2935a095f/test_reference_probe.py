"""原生参考探测的内容身份与恢复边界；不运行NPU，不把探测计为候选达标。"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import probe_foreach_reference as probe
import fp32_rounding_probe as arithmetic_probe
import compensated_arithmetic


class ReferenceProbeTests(unittest.TestCase):
    def prepare(self,root):
        out=root/'reference-probes/native-foreach-expm1-v1';out.mkdir(parents=True)
        files={}
        for name in ['source','binary','header']:
            p=out/name;p.write_text(name);files[name]=p
        build=dict(status='built',source=str(files['source']),source_sha256=probe.sha(files['source']),
                   binary=str(files['binary']),binary_sha256=probe.sha(files['binary']),
                   dependencies={'vendor':dict(frozen=str(files['header']),sha256=probe.sha(files['header']))})
        path=out/'build.json';path.write_text(json.dumps(build))
        return out,path,files

    def test_changed_binary_or_vendor_header_cannot_reuse_build_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            out,build,files=self.prepare(Path(folder))
            self.assertEqual(probe.verify_build(build)['status'],'built')
            for name in ['binary','header']:
                old=files[name].read_bytes();files[name].write_bytes(b'changed')
                with self.assertRaises(AssertionError):probe.verify_build(build)
                files[name].write_bytes(old)

    def test_complete_receipt_recovers_without_device_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out,build,files=self.prepare(root)
            checker=out/'temporary.py';checker.write_text('frozen checker')
            identity=probe.sha(checker);checker.rename(out/('checker-'+identity+'.py'))
            (out/'numerics.json').write_text(json.dumps(dict(status='numerics_failed',
                build_sha256=probe.sha(build),checker_sha256=identity,qualified_CANN=False,performance_measured=False)))
            runner=SimpleNamespace(state=dict(jobs={}),persist=mock.Mock(),
                cmd=mock.Mock(side_effect=AssertionError('不应重复提交设备')),wait=mock.Mock())
            helper=SimpleNamespace(BATCH=root,HERE=root,PY='unused',
                read=lambda p:json.loads(p.read_text()),save=lambda *args:None)
            with mock.patch.dict('sys.modules',night_common=helper):probe.process(runner)
            self.assertEqual(runner.state['jobs']['reference-probe:native-foreach-expm1-v1']['status'],'completed')
            self.assertNotIn('operators',runner.state)
            runner.cmd.assert_not_called();runner.wait.assert_not_called()

    def test_running_without_receipt_is_not_resubmitted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.prepare(root)
            runner=SimpleNamespace(state=dict(jobs={'reference-probe:native-foreach-expm1-v1':dict(status='running')}),
                cmd=mock.Mock(),wait=mock.Mock(),persist=mock.Mock())
            helper=SimpleNamespace(BATCH=root,HERE=root,PY='unused',read=lambda p:json.loads(p.read_text()),save=lambda *args:None)
            with mock.patch.dict('sys.modules',night_common=helper),self.assertRaises(AssertionError):probe.process(runner)
            runner.cmd.assert_not_called();runner.wait.assert_not_called()


class ArithmeticProbeRecoveryTests(unittest.TestCase):
    probe = arithmetic_probe
    probe_name = 'fp32-rounding-v1'

    def prepare(self,root):
        out=root/'numerical-probes'/self.probe_name;out.mkdir(parents=True)
        spec=out/'spec.json';spec.write_text(json.dumps(dict(checker_sha256=self.probe.sha(self.probe.__file__))))
        return out,spec

    def test_known_arithmetic_receipt_is_reused_without_device_call(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out,spec=self.prepare(root)
            result=dict(status='measured',checker_sha256=self.probe.sha(self.probe.__file__),
                spec_sha256=self.probe.sha(spec),qualified_CANN=False,performance_measured=False)
            (out/'result.json').write_text(json.dumps(result))
            runner=SimpleNamespace(batch=root,state=dict(jobs={}),persist=mock.Mock(),
                cmd=mock.Mock(side_effect=AssertionError('不得重复提交')))
            self.probe.process(runner)
            self.assertEqual(runner.state['jobs']['diagnostic:'+self.probe_name]['status'],'completed')
            runner.cmd.assert_not_called()

    def test_unknown_arithmetic_request_is_not_submitted_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.prepare(root)
            runner=SimpleNamespace(batch=root,state=dict(jobs={'diagnostic:'+self.probe_name:dict(status='running')}),
                persist=mock.Mock(),cmd=mock.Mock())
            with self.assertRaises(AssertionError):self.probe.process(runner)
            runner.cmd.assert_not_called()


class CompensatedProbeRecoveryTests(ArithmeticProbeRecoveryTests):
    probe = compensated_arithmetic
    probe_name = 'fp32-compensated-v2'


if __name__=='__main__':unittest.main()
