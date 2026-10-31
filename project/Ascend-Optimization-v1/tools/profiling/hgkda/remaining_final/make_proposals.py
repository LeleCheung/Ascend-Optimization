"""把新设备库与已验证宿主启动组合成身份固定的独立候选。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PROJECT=Path(__file__).resolve().parents[4]
OLD=PROJECT/'operators/campaigns/remaining-nine-20261030'
BATCH=PROJECT/'operators/campaigns/remaining-final-20261031'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);a=p.parse_args()
    build=json.loads((BATCH/('device-build-'+a.variant+'.json')).read_text())
    for op,previous in [('fix','ascendc-python-allocation-v5'),('expm1_','ascendc-double-buffer-v4')]:
        old=OLD/'proposals'/op/previous;out=BATCH/'proposals'/op/a.variant
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True)
        record=json.loads((old/'manifest.json').read_text())
        copied=[]
        for name in ['_hgkda_remaining_host_v4.so','ascendc_host.cpp']:
            shutil.copyfile(old/name,out/name);copied.append(out/name)
        for name,path in [('libhgkda_remaining.so',Path(build['binary'])),('pointwise.cpp',BATCH/('native-build-'+a.variant)/'pointwise.cpp'),('CMakeLists.txt',BATCH/('native-build-'+a.variant)/'CMakeLists.txt')]:
            shutil.copyfile(path,out/name);copied.append(out/name)
        original_device_sha=sha(old/'libhgkda_remaining.so')
        body=(old/'main.py').read_text().replace(str(old),str(out)).replace(original_device_sha,sha(out/'libhgkda_remaining.so'))
        source=out/'main.py';source.write_text(body,encoding='utf8');copied.append(source)
        attribution='Codex自写Ascend C大分块与显式流水事件；保留已核验C++同stream启动和Triton特殊布局回退；固定KGS及补测'
        manifest=dict(record,files={str(f):sha(f) for f in copied},source_sha256=sha(source),device_build=build,attribution=attribution)
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        proposal=dict(operator=op,id=a.variant,source='main.py',source_sha256=sha(source),backend=manifest['backend'],
            implementation_language='python',native_manifest='manifest.json',attribution=attribution,
            summary='扩大连续块并复用读入缓冲写回，减少FP32片上中间复制；保留原数值路径和特殊布局回退',
            prediction='大输入设备时间减少；完整调用和快速路径补测决定是否保留，不由单一指标推断成功')
        (out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='prepared',variant=a.variant,operators=['fix','expm1_'])))

if __name__=='__main__':main()
