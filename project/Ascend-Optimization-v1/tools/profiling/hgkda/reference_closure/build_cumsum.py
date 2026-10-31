"""单CPU线程构建独立CANN扫描桥，不运行设备。"""
import hashlib
import json
import os
from pathlib import Path
os.environ['MAX_JOBS']='1'
import torch,torch_npu
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'
BUILD=Path('/data/hanle/ascend-optimization/runtime/reference-closure-cumsum');BUILD.mkdir(parents=True,exist_ok=True)
npu=Path(torch_npu.__file__).parent;source=HERE/'cann_cumsum.cpp'
module=load(name='_hgkda_reference_cumsum_v1',sources=[str(source)],build_directory=str(BUILD),
    extra_include_paths=[str(npu/'include'),str(npu/'include/third_party/op-plugin'),'/usr/local/Ascend/cann-9.0.0/include'],
    extra_cflags=['-O2','-g0'],extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib'),
        '-L/usr/local/Ascend/cann-9.0.0/lib64','-lopapi','-lascendcl','-Wl,-rpath,/usr/local/Ascend/cann-9.0.0/lib64'],with_cuda=False,verbose=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=BATCH/'references/artifacts';out.mkdir(parents=True,exist_ok=True);binary=out/Path(module.__file__).name;binary.write_bytes(Path(module.__file__).read_bytes())
record=dict(module_name='_hgkda_reference_cumsum_v1',source=str(source),source_sha256=sha(source),binary=str(binary),binary_sha256=sha(binary),
    kind='CANN_cumsum_host_bridge',scope='仅厂商扫描部分；Triton仍负责相邻增量与其他布局快速路径；不是纯Triton自主算子')
(BATCH/'cann-cumsum-build.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
print(json.dumps(record,ensure_ascii=False))
