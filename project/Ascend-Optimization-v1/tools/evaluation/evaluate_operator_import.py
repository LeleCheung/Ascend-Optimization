"""供同目录工具导入现有带连字符的评测脚本，统一 HTTP 和证据存储逻辑。"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("evaluate_operator", Path(__file__).with_name("evaluate-operator.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
call = module.call
save = module.save
