"""冻结Ascend C计算不变、C++任务队列启动的新候选。"""
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
batch=PROJECT/'operators/campaigns/threeway-20261030'
build=json.loads((batch/'ascendc-host-build.json').read_text())
old=batch/'native-candidates/log10_/ascendc-ln-v1'
out=batch/'proposals/log10_/ascendc-taskqueue-v2';out.mkdir(parents=True,exist_ok=True)
for source in [Path(build['binary']),Path(build['source']),old/'libhgkda_log10.so',old/'log10.cpp',old/'CMakeLists.txt']:
    target=out/source.name
    raw=source.read_bytes()
    if target.exists() and target.read_bytes()!=raw:raise FileExistsError(target)
    target.write_bytes(raw)
cpp=out/Path(build['binary']).name
kernel=out/'libhgkda_log10.so'
body=(batch/'starting-points/log10_.py').read_text(encoding='utf8').replace('def run(A):','def _triton_run(A):',1)
body+='''

# 计算来自本项目Ascend C源码；C++只负责元数据、任务队列及启动。
import hashlib as _hashlib
import importlib.util as _importlib
from pathlib import Path as _Path

_HOST_PATH=_Path(HOST_PATH)
_KERNEL_PATH=_Path(KERNEL_PATH)
if _hashlib.sha256(_HOST_PATH.read_bytes()).hexdigest()!=HOST_SHA:
    raise RuntimeError('Ascend C宿主二进制身份不符')
if _hashlib.sha256(_KERNEL_PATH.read_bytes()).hexdigest()!=KERNEL_SHA:
    raise RuntimeError('Ascend C计算二进制身份不符')
if torch.__version__!=TORCH_VERSION or torch._C._GLIBCXX_USE_CXX11_ABI!=TORCH_ABI:
    raise RuntimeError('宿主封装的Torch版本或ABI不符')
_spec=_importlib.spec_from_file_location('_hgkda_ascendc_host_v2',_HOST_PATH)
_host=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_host)
_host.configure_library(str(_KERNEL_PATH))


def run(A):
    return _host.run_log10(A,_triton_run)
'''
replacements=dict(HOST_PATH=repr(str(cpp)),KERNEL_PATH=repr(str(kernel)),
                  HOST_SHA=repr(hashlib.sha256(cpp.read_bytes()).hexdigest()),
                  KERNEL_SHA=repr(hashlib.sha256(kernel.read_bytes()).hexdigest()),
                  TORCH_VERSION=repr(build['torch']),TORCH_ABI=repr(build['abi']))
# 标记只出现在赋值右侧，不能替换变量名中的相同子串。
for key,value in replacements.items():
    body=body.replace('='+key+'\n','='+value+'\n').replace('('+key+')','('+value+')').replace('!='+key+':','!='+value+':').replace('!='+key+' or','!='+value+' or')
source=out/'main.py'
raw=body.encode('utf8')
if source.exists() and source.read_bytes()!=raw:raise FileExistsError(source)
source.write_bytes(raw)
files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name not in {'manifest.json','proposal.json'}}
manifest=dict(operator='log10_',backend='AscendC_with_Triton_layout_fallback',implementation_language='python',
              files=files,host_build=build,source_sha256=hashlib.sha256(raw).hexdigest(),
              attribution='Codex编写Ascend C及任务队列封装；布局回退复用历史Triton；原KGS及独立补测验收')
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
proposal=dict(operator='log10_',id=out.name,source='main.py',source_sha256=manifest['source_sha256'],
              implementation_language='python',backend=manifest['backend'],native_manifest='manifest.json',
              attribution=manifest['attribution'],summary='Ascend C计算不变，使用C++同stream任务队列启动，避免逐次Python元数据及刷新等待',
              prediction='连续小中输入完整调用下降；设备kernel及Ln/Muls数值顺序不变；新输入地址和当前stream每次获取')
(out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status='prepared',source=str(source),sha256=manifest['source_sha256'])))
