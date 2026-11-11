"""IR比较只能忽略调试位置，不能抹去类型、运算或常量差异。"""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('compare_compiled',Path(__file__).resolve().parents[3]/'hgkda/src/hgkda/runtime/compare_compiled.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CompiledComparisonTests(unittest.TestCase):
    def test_debug_locations_are_ignored(self):
        first='#loc1 = loc("first.py":1:2)\n%0 = math.log %x : f32 loc(#loc1)\n'
        other='#loc9 = loc("other.py":10:20)\n%0 = math.log %x : f32 loc(#loc9)\n'
        self.assertEqual(module.normalized(first),module.normalized(other))

    def test_types_operations_and_constants_are_preserved(self):
        text='%0 = math.log %x : f32\n%1 = arith.constant 1 : i32\n'
        for changed in [text.replace('f32','f16'),text.replace('math.log','math.exp'),text.replace('1 : i32','2 : i32')]:
            self.assertNotEqual(module.normalized(text),module.normalized(changed))

    def test_inline_and_nested_locations_preserve_quoted_program_data(self):
        first='%0 = op %x loc("arg"(loc("left.py":1:2))) : f32\n'
        other='%0 = op %x loc("right.py":9:8) : f32\n'
        self.assertEqual(module.normalized(first),module.normalized(other))
        data='%0 = arith.constant "loc(program-data)" : !str\n'
        self.assertEqual(module.normalized(data),data)
        with self.assertRaises(AssertionError):module.normalized('%0 = op %x loc(#loc1\n')


if __name__=='__main__':unittest.main()
