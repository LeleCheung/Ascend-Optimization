"""方法迁移上下文绑定当前源码、合同与冻结方法，避免借用旧候选采集。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('transfer_context',ROOT/'tools/skill_validation/transfer_context.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class TransferContextTests(unittest.TestCase):
    def test_compact_regions_preserve_wait_accounting_and_source_identity(self):
        data=dict(source_sha256='current',case=dict(name='case'),total_regions=1,limitations=['overlap'],
            regions=[dict(id='region',pcs=list(range(100)),source_lines=list(range(100)))],
            timeline=dict(status='completed',capture_duration=10,pipelines=[
                dict(productive_union=4,wait_only_union=3,idle_union=3,idle_intervals=[[i,i+1] for i in range(20)])]))
        result=module.compact_regions(data)
        self.assertEqual(result['source_sha256'],'current')
        self.assertEqual(result['regions'][0]['pc_count'],100)
        self.assertEqual(len(result['regions'][0]['pcs']),8)
        row=result['timeline']['pipelines'][0]
        self.assertEqual(row['productive_union']+row['wait_only_union']+row['idle_union'],10)

    def test_host_profile_units_use_operator_iterations(self):
        rows=[dict(function='key',self_seconds=.0002,cumulative_seconds=.0003,total_calls=100)]
        result=module.profile_frames(rows,50)[0]
        self.assertAlmostEqual(result['self_us_per_op'],4.)
        self.assertAlmostEqual(result['inclusive_us_per_op'],6.)
        self.assertEqual(result['self_seconds'],.0002)
        with self.assertRaises(AssertionError):module.profile_frames(rows,0)

    def fixture(self,root):
        workspace=root/'workspace';workspace.mkdir()
        (workspace/'starter.py').write_text('fixed upstream bytes\n')
        (workspace/'contract.json').write_text(json.dumps(dict(formal_cases=36,supplementary_each=66)))
        package=root/'package';package.mkdir();(package/'method.md').write_text('fixed method\n')
        method=root/'method.json';method.write_text(json.dumps(dict(files=[dict(path='method.md',sha256=hashlib.sha256((package/'method.md').read_bytes()).hexdigest())])))
        return dict(run_directory=str(root),package=str(package),method_manifest=str(method),
                    job=dict(operator='log10_'),phase='frozen')

    def test_contract_and_exact_methods_are_available_without_fabricated_profiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.fixture(Path(tmp));result=module.build(config)
            self.assertEqual(result['operator'],'log10_')
            self.assertEqual(result['contract']['supplementary_each'],66)
            self.assertEqual(result['knowledge_retrieval']['exact'],1)
            self.assertIsNone(result['paired_evidence'])
            self.assertEqual(result['actual_ir'],[])

    def test_stale_pair_or_changed_frozen_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.fixture(root)
            (root/'workspace/paired-context-pointer.json').write_text(json.dumps(dict(source_sha256='another')))
            with self.assertRaises(AssertionError):module.build(config)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.fixture(root);(root/'package/method.md').write_text('modified')
            with self.assertRaises(AssertionError):module.build(config)

    def test_retired_evidence_is_preserved_and_current_feedback_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace=Path(tmp)
            old=workspace/'paired-context-pointer.json';old.write_text(json.dumps(dict(source_sha256='old')))
            raw=old.read_bytes()
            current=workspace/'latest-preflight.json';current.write_text(json.dumps(dict(source_sha256='current',status='passed')))
            module.retire_previous(workspace,'current')
            self.assertFalse(old.exists())
            self.assertEqual(next((workspace/'retired-evidence').iterdir()).read_bytes(),raw)
            self.assertEqual(json.loads(current.read_bytes())['source_sha256'],'current')

    def test_host_stack_is_bound_to_current_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.fixture(root)
            path=root/'workspace/host-profile.json'
            path.write_text(json.dumps(dict(source_sha256='stale',records=[])))
            with self.assertRaises(AssertionError):module.build(config)
            digest=hashlib.sha256((root/'workspace/starter.py').read_bytes()).hexdigest()
            path.write_text(json.dumps(dict(source_sha256=digest,records=[],scope='diagnostic only')))
            self.assertEqual(module.build(config)['host_diagnosis']['source_sha256'],digest)


if __name__=='__main__':unittest.main()
