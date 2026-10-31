"""冻结来源可见Ascend C与Triton布局回退的候选；厂商目标库不参与计算。"""
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    device=json.loads((BATCH/'ascendc-build.json').read_text())
    host=json.loads((BATCH/'host-build.json').read_text())
    for op,kind in [('negative_',0),('log2_',1),('fix',2),('round_out',3),('special_round_out',3),('hardswish',4)]:
        out=BATCH/'proposals'/op/'ascendc-taskqueue-v1';out.mkdir(parents=True,exist_ok=True)
        assets=[Path(device['binary']),Path(host['binary'])]
        assets.extend(Path(p) for data in [device,host] for p in data['files'] if not p.endswith('.so'))
        copied=[]
        for p in assets:
            dest=out/p.name;raw=p.read_bytes()
            if dest.exists() and dest.read_bytes()!=raw:raise FileExistsError(dest)
            dest.write_bytes(raw);copied.append(dest)
        source=(BATCH/'starting-points'/(op+'.py')).read_text(encoding='utf8')
        source+='\n\n_original_run=run\n'
        source+='''
import hashlib as _hashlib
import importlib.util as _importlib
from pathlib import Path as _Path
_host_path=_Path(HOST_PATH)
_device_path=_Path(DEVICE_PATH)
if _hashlib.sha256(_host_path.read_bytes()).hexdigest()!=HOST_SHA:
    raise RuntimeError('宿主二进制身份不符')
if _hashlib.sha256(_device_path.read_bytes()).hexdigest()!=DEVICE_SHA:
    raise RuntimeError('计算二进制身份不符')
if torch.__version__!=TORCH_VERSION or torch._C._GLIBCXX_USE_CXX11_ABI!=TORCH_ABI:
    raise RuntimeError('Torch版本或ABI不符')
_spec=_importlib.spec_from_file_location('_hgkda_remaining_host_v1',_host_path)
_host=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_host)
_host.configure_library(str(_device_path))
'''
        values=dict(HOST_PATH=repr(str(out/Path(host['binary']).name)),DEVICE_PATH=repr(str(out/Path(device['binary']).name)),
                    HOST_SHA=repr(sha(out/Path(host['binary']).name)),DEVICE_SHA=repr(sha(out/Path(device['binary']).name)),
                    TORCH_VERSION=repr(host['torch']),TORCH_ABI=repr(host['abi']))
        for key,value in values.items():source=source.replace('('+key+')','('+value+')').replace('!='+key+':','!='+value+':').replace('!='+key+' or','!='+value+' or')
        if op=='round_out':
            source+='''
def _native_round_fallback(x, out=None):
    return _original_run(x,decimals=0,out=out)


def run(input, *, decimals=0, out=None):
    if decimals!=0:
        return _original_run(input,decimals=decimals,out=out)
    return _host.execute(input,out,3,_native_round_fallback)
'''
        elif op=='special_round_out':
            source+='''
def _native_round_fallback(x,out):
    return _original_run(x,out,decimals=0)


def run(self,out,*,decimals=0):
    if decimals!=0:
        return _original_run(self,out,decimals=decimals)
    return _host.execute(self,out,3,_native_round_fallback)
'''
        else:
            param='A' if op in {'negative_','log2_'} else 'self' if op=='fix' else 'x'
            source+='\ndef run('+param+'):\n    return _host.execute('+param+',None,'+str(kind)+',_original_run)\n'
        main_path=out/'main.py';raw=source.encode('utf8')
        if main_path.exists() and main_path.read_bytes()!=raw:raise FileExistsError(main_path)
        main_path.write_bytes(raw)
        files={str(path):sha(path) for path in [*copied,main_path]}
        manifest=dict(operator=op,backend='AscendC_with_Triton_layout_fallback',implementation_language='python',
            files=files,source_sha256=sha(main_path),host_build=host,device_build=device,
            attribution='Codex编写Ascend C计算及任务队列封装；非连续/非对齐/舍入小数位回退历史Triton；固定KGS及补测')
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        proposal=dict(operator=op,id=out.name,source='main.py',source_sha256=sha(main_path),backend=manifest['backend'],
            implementation_language='python',native_manifest='manifest.json',attribution=manifest['attribution'],
            summary='连续对齐输入用自写Ascend C，C++同stream任务队列启动；保留动态布局回退',
            prediction='降低Python元数据及启动开销；16384元素流式块；语义和性能由完整KGS与补测判定')
        (out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Prepared 6 Ascend C proposals')


if __name__=='__main__':main()
