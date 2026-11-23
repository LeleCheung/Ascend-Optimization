"""Check explicit simulator aliases without accepting computation changes."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/research'))
from gqa_bcmlir_runtime import canonical_input


class GQABCMLIRBinding(unittest.TestCase):
    def test_only_explicit_alias_and_source_location_are_normalized(self):
        manifest = dict(source='/copied/candidate.py', original_source_location='/original/candidate.py',
                        target_aliases={'Ascend910B4': 'Ascend910B4-1'})
        original = '#hacc.target<"Ascend910B4-1"> loc("/original/candidate.py") arith.addf'
        incoming = '#hacc.target<"Ascend910B4"> loc("/copied/candidate.py") arith.addf'
        self.assertEqual(canonical_input(incoming, manifest), original)
        self.assertNotEqual(canonical_input(incoming.replace('arith.addf', 'arith.mulf'), manifest), original)
        self.assertNotEqual(canonical_input(incoming.replace('Ascend910B4', 'Ascend910B3'), manifest), original)
        self.assertNotEqual(canonical_input(incoming, dict(manifest, target_aliases={})), original)


if __name__ == '__main__':
    unittest.main()
