"""单CPU构建out接口的元数据与Triton启动包装；不执行设备任务。"""
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
    source = HERE/'host_bridge_out.cpp'
    folder = BATCH/'host-glue/out-v1'
    folder.mkdir(parents=True, exist_ok=True)
    record = folder/'build-provenance.json'
    if record.exists():
        value = json.loads(record.read_text())
        assert value['source_sha256'] == sha(source)
        assert value['binary_sha256'] == sha(value['binary'])
        print(json.dumps(value, ensure_ascii=False))
        return
    build = ROOT/'runtime/strict-night-host-bridge-out-v1'
    build.mkdir(parents=True, exist_ok=True)
    module = load(name='_hgkda_host_bridge_out_v1', sources=[str(source)],
                  build_directory=str(build), extra_cflags=['-O2', '-g0'],
                  with_cuda=False, verbose=True)
    binary = folder/Path(module.__file__).name
    shutil.copyfile(module.__file__, binary)
    snapshot = folder/'host_bridge_out.cpp'
    shutil.copyfile(source, snapshot)
    value = dict(status='built', source=str(snapshot), source_sha256=sha(snapshot),
                 binary=str(binary), binary_sha256=sha(binary),
                 build_script_sha256=sha(__file__), torch=torch.__version__,
                 cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI, machine=platform.machine(),
                 max_jobs=1, device_computation='none; metadata and Triton launcher only')
    record.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
