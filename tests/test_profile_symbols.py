"""CANN任务别名必须绑定唯一实测符号；不凭名称选择第一条记录。"""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('collect_pair',ROOT/'tools/skill_validation/collect_pair.py')
module=importlib.util.module_from_spec(spec)
# 采集仅在服务器执行；离线只加载纯函数时避免导入Linux文件锁。
import sys,types
if sys.platform=='win32':
    previous=sys.modules.get('fcntl');sys.modules['fcntl']=types.ModuleType('fcntl')
try:spec.loader.exec_module(module)
finally:
    if sys.platform=='win32':
        if previous is None:sys.modules.pop('fcntl',None)
        else:sys.modules['fcntl']=previous


class ProfileSymbolTests(unittest.TestCase):
    def test_alias_uses_unique_symbol_actually_reported_by_collector(self):
        log='\n'.join('Kernel '+name+' skipped: not selected via --kernel-name.' for name in
                      ['Abs_aaa','Add_bbb','Log_ccc'])
        self.assertEqual(module.resolve_reported_symbol('aclnnLog10_LogAiCore_Log',log),'Log_ccc')

    def test_missing_and_ambiguous_symbols_are_rejected(self):
        for names in [[],['Add_aaa'],['Log_aaa','Log_bbb']]:
            log='\n'.join('Kernel '+name+' skipped: not selected via --kernel-name.' for name in names)
            with self.assertRaises(ValueError):module.resolve_reported_symbol('aclnnLog10_LogAiCore_Log',log)


if __name__=='__main__':unittest.main()
