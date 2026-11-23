import importlib.util
import pathlib
import sys
_root = pathlib.Path(__file__).parent
_helper_spec = importlib.util.spec_from_file_location("_ir_override_runtime", _root / "ir_override_runtime.py")
_helper = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helper)
_source = _helper.install(_root / "ir-manifest.json")
_spec = importlib.util.spec_from_file_location("_ir_frozen_candidate", _source)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
run = _module.run
