"""随机数算法核验的CPU见证，不导入设备运行时。"""
import importlib.util
from pathlib import Path
import unittest

TOOLS=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('uniform_philox',TOOLS/'check-uniform-philox.py')
checker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PhiloxReferenceTests(unittest.TestCase):
    def test_random123_zero_vector(self):
        self.assertEqual(checker.reference(0,0),(0x6627E8D5,0xE169C58D,0xBC57AC4C,0x9B00DBD8))

    def test_high_seed_and_counter_bits_are_retained(self):
        base=checker.reference(7,0)
        for bit in [31,32,40,62]:
            self.assertNotEqual(checker.reference(7|(1<<bit),0),base)
            self.assertNotEqual(checker.reference(7,1<<bit),base)


if __name__=='__main__':
    unittest.main()
