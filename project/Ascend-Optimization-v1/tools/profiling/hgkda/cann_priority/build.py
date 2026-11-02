"""单CPU编译CANN参考桥，不占用NPU、不改共享环境。"""
import os
import platform
from pathlib import Path
from common import BATCH, HERE, ROOT, read, save, sha


def main():
    os.environ['MAX_JOBS'] = '1'
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    record = BATCH / 'cann-build-v1.json'
    if record.exists():
        old = read(record)
        assert sha(old['source']) == old['source_sha256']
        assert sha(old['binary']) == old['binary_sha256']
        print('EXISTS', record)
        return
    npu = Path(torch_npu.__file__).parent
    build = ROOT / 'runtime/cann-priority-20261102-build-v1'
    build.mkdir(parents=True, exist_ok=True)
    source = HERE / 'cann_baselines.cpp'
    module = load(name='_hgkda_cann_priority_v1', sources=[str(source)], build_directory=str(build),
        extra_include_paths=[str(npu / 'include'), '/usr/local/Ascend/cann-9.0.0/include'],
        extra_cflags=['-O2', '-g0'], extra_ldflags=['-L' + str(npu / 'lib'), '-ltorch_npu',
          '-Wl,-rpath,' + str(npu / 'lib'), '-L/usr/local/Ascend/cann-9.0.0/lib64', '-lopapi', '-lascendcl',
          '-Wl,-rpath,/usr/local/Ascend/cann-9.0.0/lib64'], with_cuda=False, verbose=True)
    target = BATCH / 'references/artifacts' / Path(module.__file__).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(module.__file__).read_bytes())
    headers = {}
    for line in source.read_text().splitlines():
        if line.startswith('#include <aclnnop/'):
            name = line.split('/')[-1].rstrip('>')
            p = Path('/usr/local/Ascend/cann-9.0.0/include/aclnnop') / name
            frozen = target.parent / name
            frozen.write_bytes(p.read_bytes())
            headers[str(frozen)] = sha(frozen)
    save(record, dict(status='built', module_name='_hgkda_cann_priority_v1', source=str(source),
         source_sha256=sha(source), binary=str(target), binary_sha256=sha(target), headers=headers,
         torch=torch.__version__, torch_npu=torch_npu.__version__, abi=torch._C._GLIBCXX_USE_CXX11_ABI,
         architecture=platform.machine(), scope='直接CANN API参考；分配和包装成本计入完整调用'))
    print('BUILT', target)


if __name__ == '__main__':
    main()
