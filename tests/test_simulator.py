"""公开模拟器格式解析及逐核隔离；不启动采集器。"""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hgkda.simulator import parse


class SimulatorTests(unittest.TestCase):
    def test_same_pc_on_different_cores_is_not_merged_and_source_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'candidate.py';source.write_text('def k():\n    pass\n')
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            for index in range(2):
                core=root/'OPPROF_one/simulator'/('core'+str(index));core.mkdir(parents=True)
                trace=dict(traceEvents=[dict(ph='X',name='LOAD',args=dict(pc_addr='0x10',code=str(source)+':2'))])
                (core/'trace.json').write_text(json.dumps(trace))
                with (core/'x_instr_exe.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['addr','instr','pipe','call_count','cycles','running_time(us)'])
                    writer.writeheader();writer.writerow(dict(addr='0x10',instr='LOAD',pipe='MTE2',call_count='1',cycles=str(index+2),**{'running_time(us)':str(index+1)}))
            result=parse(root,'k',source,digest,dict(name='case'))
            cores=result['kernels'][0]['cores']
            self.assertEqual(len(cores),2)
            self.assertEqual([r['instructions'][0]['running_time_us'] for r in cores],[1,2])
            self.assertEqual(cores[0]['instructions'][0]['source']['code'],'pass')
            (root/'OPPROF_two/simulator').mkdir(parents=True)
            with self.assertRaises(ValueError):parse(root,'k',source,digest,dict(name='case'))


if __name__=='__main__':
    unittest.main()
