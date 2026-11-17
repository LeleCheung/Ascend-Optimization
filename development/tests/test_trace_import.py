"""Safety and source identity boundaries for offline literal trace replay."""
import unittest

from development.tools.traces.importer import source_mutation
from development.tools.traces.index import source_occurrence


class TraceReplayTests(unittest.TestCase):
    def test_heredoc_extracts_literal_without_running_following_shell(self):
        tool = dict(name="Bash", input=dict(command="cat > candidate.py <<'PY'\nx = 1\nPY\nrm -rf never-run"))
        self.assertEqual(source_mutation(tool, None), ("x = 1\n", "literal_heredoc"))

    def test_ambiguous_edit_does_not_invent_recovered_source(self):
        tool = dict(name="Edit", input=dict(file_path="/tmp/candidate.py", old_string="x", new_string="y"))
        self.assertEqual(source_mutation(tool, "x + x"), (None, "ambiguous_edit"))

    def test_nonliteral_shell_is_flagged_without_execution(self):
        tool = dict(name="Bash", input=dict(command="compute_source > candidate.py"))
        self.assertEqual(source_mutation(tool, "old"), (None, "unresolved_shell_edit"))

    def test_repeated_source_hash_does_not_merge_distinct_occurrences(self):
        session = dict(calls=[dict(id="first", output_sha256="a", edits=1),
                              dict(id="rollback", output_sha256="a", edits=1)])
        with self.assertRaisesRegex(ValueError, "explicit evaluation anchor"):
            source_occurrence(session, "a")


if __name__ == "__main__":
    unittest.main()
