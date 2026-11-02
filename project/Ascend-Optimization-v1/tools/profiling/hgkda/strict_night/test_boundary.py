"""控制器恢复不能重启未知/活跃工作区，也不能相信与存活进程矛盾的终态。"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('night_boundary',Path(__file__).with_name('boundary_upgrade.py'))
boundary=importlib.util.module_from_spec(spec);spec.loader.exec_module(boundary)


class BoundaryResumeTests(unittest.TestCase):
    def test_unknown_active_and_known_terminal_are_distinguished(self):
        with tempfile.TemporaryDirectory() as folder:
            w=Path(folder)/'workspace'
            self.assertFalse(boundary.terminal_workspace(w))
            w.mkdir()
            with self.assertRaises(AssertionError):boundary.terminal_workspace(w)
            m=w/'.kernelgen/run-process.json';m.parent.mkdir()
            m.write_text(json.dumps(dict(pid=999999999,process_start='0',finished_at=None)))
            with self.assertRaises(AssertionError):boundary.terminal_workspace(w)
            m.write_text(json.dumps(dict(pid=999999999,process_start='0',finished_at='known terminal')))
            self.assertTrue(boundary.terminal_workspace(w))
    @unittest.skipUnless(Path('/proc/self/stat').exists(),'需要Linux进程身份')
    def test_alive_matching_process_cannot_be_called_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            w=Path(folder);m=w/'.kernelgen/run-process.json';m.parent.mkdir()
            start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
            m.write_text(json.dumps(dict(pid=os.getpid(),process_start=start,finished_at='incorrect terminal')))
            with self.assertRaises(AssertionError):boundary.terminal_workspace(w)


if __name__=='__main__':unittest.main()
