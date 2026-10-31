import json
from pathlib import Path
import tempfile
import unittest

from trace_summary import summarize


class TraceSummaryTests(unittest.TestCase):
    def test_ascend_tasks_without_category_and_exact_timestamps(self):
        events=[dict(ph='M',name='process_name',pid=10,args=dict(name='Ascend Hardware')),
                dict(ph='X',pid=10,name='log10',ts='1793336420017585.485',dur=.2,args={'Task Type':'KERNEL_AIVEC'}),
                dict(ph='X',pid=10,name='copy',ts='1793336420017585.585',dur=.3,args={'Task Type':'MEMCPY_ASYNC'}),
                dict(ph='X',pid=11,name='Computing',ts='1793336420017585.485',dur=100),
                dict(ph='X',pid=12,name='aten::log10_',cat='cpu_op',ts=1,dur=1000)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.json';path.write_text(json.dumps(events))
            result=summarize(path)
        self.assertEqual(result['kernel_names'],['log10','copy'])
        self.assertEqual(result['device_interval_union_us'],.4)
        self.assertEqual(result['device_elapsed_span_us'],.4)

    def test_missing_device_tasks_is_not_zero_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.json';path.write_text('{"traceEvents":[]}')
            result=summarize(path)
        self.assertEqual(result['status'],'no_device_tasks')
        self.assertIsNone(result['device_elapsed_span_us'])


if __name__=='__main__':unittest.main()
