"""舍入迁移必须保留两份合同各自的位置参数、必需性与默认值。"""
import unittest
from signature_gate import validate


class SignatureTests(unittest.TestCase):
    def definition(self):
        return dict(parameters=[dict(name='self', kind='positional_or_keyword', required=True),
                                dict(name='out', kind='positional_or_keyword', required=True),
                                dict(name='decimals', kind='keyword_only', required=False, default=0)])

    def test_exact_required_out_and_alias_pass(self):
        validate('def run(self, out, *, decimals=0):\n    pass\n', self.definition())
        validate('def implementation(self, out, *, decimals=0):\n    pass\nrun=implementation\n', self.definition())

    def test_keyword_only_out_migration_is_rejected(self):
        with self.assertRaises(ValueError):
            validate('def run(self, *, decimals=0, out=None):\n    pass\n', self.definition())

    def test_default_status_and_default_type_must_match(self):
        for signature in ['self, out=None, *, decimals=0', 'self, out, *, decimals=False',
                          'self, out, decimals=0', 'self, out, *, decimals=1']:
            with self.assertRaises(ValueError):
                validate('def run('+signature+'):\n    pass\n', self.definition())


if __name__ == '__main__':
    unittest.main()
