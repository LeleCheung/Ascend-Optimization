"""不同输入不可混比；真实搬运量与指令数变化分别保留。"""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from compare_records import compare,ir_signatures


class CompareRecordsTests(unittest.TestCase):
    def test_reject_different_input_contract_or_multiple_kernels(self):
        context=dict(spec=dict(operator='qkv',case=dict(seed=1),contract_sha256=dict(timing='fixed')))
        metrics=dict(kernels=[dict(kernel='old',metrics=[])])
        for field,value in [('operator','other'),('case',dict(seed=2)),('contract_sha256',dict(timing='different'))]:
            changed=copy.deepcopy(context);changed['spec'][field]=value
            with self.assertRaises(ValueError):compare(context,changed,metrics,metrics)
        with self.assertRaises(ValueError):compare(context,context,metrics,dict(kernels=[]))

    def test_less_instructions_does_not_imply_less_data(self):
        context=dict(spec=dict(operator='qkv',case=dict(seed=1),contract_sha256=dict(timing='fixed')))
        def metrics(instructions):
            return dict(kernels=[dict(kernel=str(instructions),metrics=[
                dict(table='Memory.csv/0',metric='GM_to_UB_datas(KB)',value=64,rows=40,aggregation='sum'),
                dict(table='Memory.csv/0',metric='aiv_mte2_instructions',value=instructions,rows=40,aggregation='sum')])])
        result=compare(context,context,metrics(1024),metrics(64))
        rows={r['metric']:r for r in result['all_metrics']}
        self.assertEqual(rows['GM_to_UB_datas(KB)']['candidate_over_reference'],1)
        self.assertEqual(rows['aiv_mte2_instructions']['candidate_over_reference'],1/16)
        self.assertEqual(result['reference_kind'],'previous_candidate')

    def test_static_ir_signals_require_exact_bound_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'actual.mlir'
            p.write_text('%x = memref.load %a[%i] : memref<1xf32>\nscf.yield {DiscreteMemAccess}\n')
            spec=dict(case='frozen',source_sha256='source')
            context=dict(spec=spec,actual_ir=dict(spec=spec,entries=[dict(symbol='k',artifacts=[
                dict(kind='ttadapter',path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())])]))
            result=ir_signatures(context)
            self.assertEqual(result['artifacts'][0]['static_counts']['scalar_memref_load_ops'],1)
            p.write_text('changed')
            with self.assertRaises(ValueError):ir_signatures(context)


if __name__=='__main__':unittest.main()
