"""厂商trace的时间戳可能是字符串；重叠任务不能把耗时简单相加。"""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('night_component',Path(__file__).with_name('component_probe.py'))
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class TraceTests(unittest.TestCase):
    def test_vendor_string_timestamps_and_overlapping_tasks(self):
        tasks=[dict(ts='100.5',dur=10.),dict(ts='105.5',dur=10.),dict(ts=130.5,dur='2.5')]
        self.assertEqual(probe.interval_union(tasks),17.5)
        self.assertEqual(probe.interval_union([]),0.)
        self.assertEqual(probe.interval_union([dict(ts='1793626034895473.080',dur=50.981)]),50.981)


if __name__=='__main__':unittest.main()
