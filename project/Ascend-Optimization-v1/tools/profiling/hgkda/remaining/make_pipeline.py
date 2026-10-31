"""冻结双缓冲版本；旧同步实现和二进制继续保留为对照。"""
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    host=json.loads((BATCH/'host-build-v4.json').read_text());device=json.loads((BATCH/'ascendc-build-v4.json').read_text())
    targets=[('negative_','ascendc-taskqueue-v1',''),('log2_','ascendc-taskqueue-v1',''),('fix','ascendc-taskqueue-v1',''),
        ('round_out','ascendc-taskqueue-v1',''),('special_round_out','ascendc-taskqueue-v1',''),('hardswish','ascendc-taskqueue-v1',''),
        ('expm1_','ascendc-series-taskqueue-v2','-v2'),('fmod_tensor_','ascendc-div-trunc-taskqueue-v3','-v3')]
    for op,previous,revision in targets:
        old=BATCH/'proposals'/op/previous
        out=BATCH/'proposals'/op/'ascendc-double-buffer-v4';out.mkdir(parents=True,exist_ok=True)
        files=[]
        for source in [Path(p) for data in [host,device] for p in data['files']]:
            dest=out/source.name
            if dest.exists() and dest.read_bytes()!=source.read_bytes():raise FileExistsError(dest)
            dest.write_bytes(source.read_bytes());files.append(dest)
        oldhost=json.loads((BATCH/('host-build'+revision+'.json')).read_text())
        olddevice=json.loads((BATCH/('ascendc-build'+revision+'.json')).read_text())
        body=(old/'main.py').read_text(encoding='utf8')
        replacements={str(old/Path(oldhost['binary']).name):str(out/Path(host['binary']).name),
            str(old/Path(olddevice['binary']).name):str(out/Path(device['binary']).name),
            sha(Path(oldhost['binary'])):sha(Path(host['binary'])),sha(Path(olddevice['binary'])):sha(Path(device['binary'])),
            '_hgkda_remaining_host_v'+(revision[-1] if revision else '1'):'_hgkda_remaining_host_v4'}
        for key,value in replacements.items():body=body.replace(key,value)
        source=out/'main.py';raw=body.encode('utf8')
        if source.exists() and source.read_bytes()!=raw:raise FileExistsError(source)
        source.write_bytes(raw);files.append(source)
        manifest=dict(operator=op,backend='AscendC_with_Triton_layout_fallback',implementation_language='python',
            files={str(p):sha(p) for p in files},source_sha256=sha(source),host_build=host,device_build=device,
            attribution='Codex自写Ascend C双缓冲输入/输出队列与C++任务队列启动；保留原Triton布局回退；固定KGS及补测')
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        proposal=dict(operator=op,id=out.name,source='main.py',source_sha256=sha(source),backend=manifest['backend'],
            implementation_language='python',native_manifest='manifest.json',attribution=manifest['attribution'],
            summary='4096元素双缓冲队列管理MTE2/Vector/MTE3依赖；去掉逐块PIPE_ALL全流水栅栏',
            prediction='改善大输入搬运与计算流水；保持数学公式；以完整调用、语义补测及对齐原生路径补测判断')
        (out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Prepared 8 double-buffer proposals')


if __name__=='__main__':main()
