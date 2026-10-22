"""地址适配只改变元数据协议，不能缓存本次输入地址或修改数值kernel。"""
import ast
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'prepare-profilemeta.py'
s = importlib.util.spec_from_file_location('profilemeta', path)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


class MetadataTest(unittest.TestCase):
    def test_current_address_and_specialization_shape_are_preserved(self):
        source = '''
def make(n):
    def invoke(X, ptr):
        call(ptr, n)
    return invoke
'''
        updated, changed = m.transform(source)
        self.assertEqual(changed,['invoke'])
        calls=[]
        namespace={'call':lambda *args:calls.append(args)}
        exec(updated,namespace)
        small=namespace['make'](1024)
        big=namespace['make'](65536)
        small(None,1234);small(None,5678);big(None,9012)
        self.assertEqual([int(row[0]) for row in calls],[1234,5678,9012])
        self.assertEqual([row[0].size() for row in calls],[(1024,),(1024,),(65536,)])
        self.assertTrue(all(isinstance(row[0],int) for row in calls))

    def test_device_computation_is_unchanged_and_unknown_launcher_rejected(self):
        source='''
def compute(x):
    return x * 2 + 1
def make(n):
    def launch(X, ptr):
        call(ptr, None)
    return launch
'''
        updated,_=m.transform(source)
        before=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='compute')
        after=next(n for n in ast.parse(updated).body if isinstance(n,ast.FunctionDef) and n.name=='compute')
        self.assertEqual(ast.dump(before),ast.dump(after))
        with self.assertRaises(ValueError):m.transform('def run(x): return x\n')


if __name__ == '__main__':
    unittest.main()
