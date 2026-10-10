"""原生 profiling 附件短路径归档保持原始字节和完整映射。"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / 'evaluation/archive-kg-evidence.py'


class ArchiveTest(unittest.TestCase):
    def test_flat_paths_preserve_nested_artifacts_and_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / 'run'
            control = workspace / '.kernelgen'
            control.mkdir(parents=True)
            (control / 'run-progress.json').write_text('{"state":"SUCCEEDED"}', encoding='utf-8')
            work = workspace / 'stages/optimize/work'
            paths = [work / '.kernelgen/profiles/round-0002/workload-hash/profile-id/manifest.json',
                     work / '.kernelgen/profiles/round-0002/workload-other/profile-id/manifest.json',
                     work / '.kernelgen/profile-analysis/round-0002.json']
            contents = [b'{"original":"/absolute/profile/path"}\r\n', b'{}\n', b'{"evidence":true}\n']
            for path, content in zip(paths, contents):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            output = root / 'archive'
            subprocess.run([sys.executable, str(TOOL), str(workspace), str(output)],
                           capture_output=True, check=True)
            manifest = json.loads((output / 'archive-manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['format_version'], 2)
            records = manifest['files']
            self.assertEqual(len(records), 3)
            self.assertEqual(len({r['archived_path'] for r in records}), 3)
            by_source = {r['source_path']: r for r in records}
            for path, content in zip(paths, contents):
                row = by_source[path.relative_to(work).as_posix()]
                relative = Path(row['archived_path'])
                self.assertEqual(relative.parts[0], 'files')
                self.assertEqual(len(relative.parts), 2)
                self.assertEqual((output / relative).read_bytes(), content)
                self.assertEqual(row['sha256'], hashlib.sha256(content).hexdigest())
                self.assertEqual(row['size_bytes'], len(content))
                self.assertIn('original_archive_path', row)
            # 第二次调用不能覆盖既有附件。
            repeated = subprocess.run([sys.executable, str(TOOL), str(workspace), str(output)], capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)

    def test_running_workspace_is_not_archived(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = root / 'run/.kernelgen'
            control.mkdir(parents=True)
            (control / 'run-progress.json').write_text('{"state":"RUNNING"}', encoding='utf-8')
            output = root / 'archive'
            result = subprocess.run([sys.executable, str(TOOL), str(root / 'run'), str(output)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
