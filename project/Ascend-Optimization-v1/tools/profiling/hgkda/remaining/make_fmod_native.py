"""冻结Ascend C余数候选；同shape连续路径外保留真实广播/stride回退。"""
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def main():
    host=json.loads((BATCH/'host-build-v3.json').read_text());device=json.loads((BATCH/'ascendc-build-v3.json').read_text())
    out=BATCH/'proposals/fmod_tensor_/ascendc-div-trunc-taskqueue-v3';out.mkdir(parents=True,exist_ok=True)
    files=[]
    for source in [Path(p) for data in [device,host] for p in data['files']]:
        dest=out/source.name
        if dest.exists() and dest.read_bytes()!=source.read_bytes():raise FileExistsError(dest)
        dest.write_bytes(source.read_bytes());files.append(dest)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    template=(BATCH/'proposals/expm1_/ascendc-series-taskqueue-v2/main.py').read_text(encoding='utf8')
    suffix=template[template.index('\n\n_original_run=run'):template.rindex('\ndef run(A):')]
    old=BATCH/'proposals/expm1_/ascendc-series-taskqueue-v2'
    oldhost=json.loads((BATCH/'host-build-v2.json').read_text());olddevice=json.loads((BATCH/'ascendc-build-v2.json').read_text())
    for key,value in {str(old/Path(oldhost['binary']).name):str(out/Path(host['binary']).name),
        str(old/Path(olddevice['binary']).name):str(out/Path(device['binary']).name),
        sha(Path(oldhost['binary'])):sha(Path(host['binary'])),sha(Path(olddevice['binary'])):sha(Path(device['binary'])),
        '_hgkda_remaining_host_v2':'_hgkda_remaining_host_v3'}.items():suffix=suffix.replace(key,value)
    body=(BATCH/'starting-points/fmod_tensor_.py').read_text(encoding='utf8')+suffix+'''
def run(A,B):
    return _host.execute(A,B,6,_original_run)
'''
    source=out/'main.py';raw=body.encode('utf8')
    if source.exists() and source.read_bytes()!=raw:raise FileExistsError(source)
    source.write_bytes(raw);files.append(source)
    manifest=dict(operator='fmod_tensor_',backend='AscendC_with_Triton_layout_fallback',implementation_language='python',
        files={str(p):sha(p) for p in files},source_sha256=sha(source),host_build=host,device_build=device,
        attribution='Codex自写Ascend C Div/Trunc/Mul/Sub；C++任务队列启动；布局/广播回退历史Triton；固定KGS及补测')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    proposal=dict(operator='fmod_tensor_',id=out.name,source='main.py',source_sha256=sha(source),backend=manifest['backend'],
        implementation_language='python',native_manifest='manifest.json',attribution=manifest['attribution'],
        summary='连续同shape使用自写Ascend C浮点余数，保持除法/截断/乘/减顺序；C++启动',
        prediction='降低宿主成本和program调度；精确倍数、零、非有限值及独立广播布局由补测判断')
    (out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Prepared fmod Ascend C candidate')


if __name__=='__main__':main()
