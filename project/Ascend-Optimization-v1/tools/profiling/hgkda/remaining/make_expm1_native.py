"""从v2构建冻结带小值相对精度保护的Ascend C expm1候选。"""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    spec=importlib.util.spec_from_file_location('native_maker',HERE/'make_native.py')
    maker=importlib.util.module_from_spec(spec);spec.loader.exec_module(maker)
    batch=maker.BATCH
    device=json.loads((batch/'ascendc-build-v2.json').read_text())
    host=json.loads((batch/'host-build-v2.json').read_text())
    out=batch/'proposals/expm1_/ascendc-series-taskqueue-v2';out.mkdir(parents=True,exist_ok=True)
    files=[]
    for source in [Path(p) for data in [device,host] for p in data['files']]:
        dest=out/source.name
        if dest.exists() and dest.read_bytes()!=source.read_bytes():raise FileExistsError(dest)
        dest.write_bytes(source.read_bytes());files.append(dest)
    # 与v1相同的动态回退及版本检查，模块与计算入口改为独立v2。
    template=(batch/'proposals/negative_/ascendc-taskqueue-v1/main.py').read_text(encoding='utf8')
    suffix=template[template.index('\n\n_original_run=run'):]
    previous=json.loads((batch/'host-build.json').read_text());previous_device=json.loads((batch/'ascendc-build.json').read_text())
    old_folder=batch/'proposals/negative_/ascendc-taskqueue-v1'
    replacements={str(old_folder/Path(previous['binary']).name):str(out/Path(host['binary']).name),
        str(old_folder/Path(previous_device['binary']).name):str(out/Path(device['binary']).name),
        maker.sha(Path(previous['binary'])):maker.sha(Path(host['binary'])),
        maker.sha(Path(previous_device['binary'])):maker.sha(Path(device['binary'])),
        '_hgkda_remaining_host_v1':'_hgkda_remaining_host_v2',
        '_host.execute(A,None,0,_original_run)':'_host.execute(A,None,5,_original_run)'}
    for key,value in replacements.items():suffix=suffix.replace(key,value)
    source=out/'main.py';raw=((batch/'starting-points/expm1_.py').read_text(encoding='utf8')+suffix).encode('utf8')
    if source.exists() and source.read_bytes()!=raw:raise FileExistsError(source)
    source.write_bytes(raw);files.append(source)
    manifest=dict(operator='expm1_',backend='AscendC_with_Triton_layout_fallback',implementation_language='python',
        files={str(p):maker.sha(p) for p in files},source_sha256=maker.sha(source),host_build=host,device_build=device,
        attribution='Codex编写来源可见Ascend C Exp/短级数选择与任务队列启动；特殊布局回退历史Triton；固定KGS及小值补测')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    proposal=dict(operator='expm1_',id=out.name,source='main.py',source_sha256=manifest['source_sha256'],
        backend=manifest['backend'],implementation_language='python',native_manifest='manifest.json',attribution=manifest['attribution'],
        summary='Ascend C Exp与小值补偿公式分流；8192流式块；C++同stream启动',
        prediction='降低宿主启动成本，保持接近零时相对精度和有符号零；由63项补测与正式合同验收')
    (out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Prepared expm1 Ascend C candidate')


if __name__=='__main__':main()
