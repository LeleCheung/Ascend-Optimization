"""核验并加载本项目的宿主启动包装；不构建、不分配设备张量。"""
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path


def load_host_bridge():
    import torch

    directory = Path(os.environ.get(
        'HGKDA_HOST_BRIDGE_DIR',
        '/data/hanle/ascend-optimization/runtime/next25-host-bridge-build'))
    record = json.loads((directory / 'build-provenance.json').read_text(encoding='utf8'))
    source = Path(__file__).with_name('host_bridge.cpp')
    binary = directory / '_hgkda_host_bridge_v1.so'
    for path, field in [(source, 'source_sha256'), (binary, 'binary_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != record[field]:
            raise RuntimeError('宿主启动包装身份不符：' + str(path))
    if record['torch'] != torch.__version__ or record['architecture'] != platform.machine():
        raise RuntimeError('宿主启动包装与当前Torch或架构不符，请按README重新构建')
    if record['abi'] != torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError('宿主启动包装的C++ ABI不符')
    name = '_hgkda_host_bridge_v1'
    loaded = sys.modules.get(name)
    if loaded is not None:
        if Path(loaded.__file__).resolve() != binary.resolve():
            raise RuntimeError('同名宿主模块已从不同路径加载')
        return loaded
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module
