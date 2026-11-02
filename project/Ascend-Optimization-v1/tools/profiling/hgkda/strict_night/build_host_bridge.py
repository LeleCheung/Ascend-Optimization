"""单CPU构建只负责元数据与Triton启动的宿主桥；不执行设备任务。"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
ROOT = PROJECT.parents[2]
BATCH = PROJECT/'operators/campaigns/strict-night-20261102'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.environ['MAX_JOBS'] = '1'
    import torch
    from torch.utils.cpp_extension import load
    source = HERE/'host_bridge.cpp'
    folder = BATCH/'host-glue/v2'
    folder.mkdir(parents=True, exist_ok=True)
    record = folder/'build-provenance.json'
    if record.exists():
        value = json.loads(record.read_text())
        assert value['source_sha256'] == sha(source)
        assert value['binary_sha256'] == sha(value['binary'])
        print(json.dumps(value, ensure_ascii=False))
        return
    build = ROOT/'runtime/strict-night-host-bridge-v2'
    build.mkdir(parents=True, exist_ok=True)
    module = load(name='_hgkda_host_bridge_v2', sources=[str(source)],
                  build_directory=str(build), extra_cflags=['-O2', '-g0'],
                  with_cuda=False, verbose=True)
    binary = folder/Path(module.__file__).name
    shutil.copyfile(module.__file__, binary)
    snapshot = folder/'host_bridge.cpp'
    shutil.copyfile(source, snapshot)
    value = dict(status='built', source=str(snapshot), source_sha256=sha(snapshot),
                 binary=str(binary), binary_sha256=sha(binary), torch=torch.__version__,
                 abi=torch._C._GLIBCXX_USE_CXX11_ABI, architecture=platform.machine(),
                 scope='仅THPVariable元数据、当前stream查询及PyObject_Vectorcall到Triton生成启动器；无设备运算或搬运')
    record.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
