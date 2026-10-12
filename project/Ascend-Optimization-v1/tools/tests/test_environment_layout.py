"""迁移不能越过项目边界，未复验不得清理旧目录。"""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'organize', Path(__file__).resolve().parents[1] / 'environment/organize-910b.py')
organize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(organize)


class LayoutTests(unittest.TestCase):
    def test_root_and_outside_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'project'
            root.mkdir()
            with patch.object(organize, 'ROOT', root):
                for target in (root, root.parent / 'other', root / '../other'):
                    with self.assertRaises(ValueError):
                        organize.bound(target)

    def test_move_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'old'
            target = root / 'new'
            source.mkdir(); target.mkdir()
            with patch.object(organize, 'ROOT', root):
                with self.assertRaises(RuntimeError):
                    organize.move(source, target)
            self.assertTrue(source.is_dir())
            self.assertTrue(target.is_dir())

    def test_cleanup_requires_completed_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(organize, 'JOURNAL', Path(directory)):
                with self.assertRaises(FileNotFoundError):
                    organize.compact()

    def test_known_worktree_uses_preserved_git_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / 'runs/migration'
            target = root / 'Ascend-Optimization-current'
            target.mkdir()
            original = root / 'Ascend-Optimization/.git/worktrees/old'
            actual = journal / 'old-Ascend-Optimization/.git/worktrees/old'
            actual.mkdir(parents=True)
            (target / '.git').write_text('gitdir: ' + original.as_posix(), encoding='utf-8')
            with patch.object(organize, 'ROOT', root), patch.object(organize, 'JOURNAL', journal):
                self.assertEqual(organize.git_arguments(target)[1], '--git-dir=' + str(actual))

    def test_foreign_worktree_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'old'
            target.mkdir()
            (target / '.git').write_text('gitdir: /another-project/.git/worktrees/old', encoding='utf-8')
            with patch.object(organize, 'ROOT', root):
                with self.assertRaises(RuntimeError):
                    organize.git_arguments(target)


if __name__ == '__main__':
    unittest.main()
