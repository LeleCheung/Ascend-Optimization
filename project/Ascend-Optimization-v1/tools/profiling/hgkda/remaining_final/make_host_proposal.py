"""将同一设备实现配合单次宿主入口，独立登记而不覆盖原提案。"""
import hashlib
import json
from pathlib import Path
import shutil

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-final-20261031'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    old=BATCH/'proposals/fix/event-large-v6';out=BATCH/'proposals/fix/host-python-allocation-v9'
    if out.exists():raise FileExistsError(out)
    host=json.loads((BATCH/'host-build-python-allocation-v9.json').read_text())
    previous=json.loads((old/'manifest.json').read_text());out.mkdir(parents=True)
    files=[]
    for name in ['libhgkda_remaining.so','pointwise.cpp','CMakeLists.txt']:
        shutil.copyfile(old/name,out/name);files.append(out/name)
    for path in host['files']:
        p=Path(path);shutil.copyfile(p,out/p.name);files.append(out/p.name)
    source=out/'main.py'
    body=(old/'main.py').read_text().replace(str(old),str(out))
    body=body.replace('_hgkda_remaining_host_v4','_hgkda_remaining_host_v9').replace(sha(old/'_hgkda_remaining_host_v4.so'),sha(out/'_hgkda_remaining_host_v9.so'))
    body=body.replace('    output=self.new_empty(self.shape)\n    return _host.execute(self,output,2,_allocated_fallback)','    return _host.execute_allocated(self,_allocated_fallback)')
    source.write_text(body,encoding='utf8');files.append(source)
    attribution='Codex合并Python torch.empty_like分配与C++启动入口；自写Ascend C设备源码不变；非连续布局回退Triton'
    frozen=dict(previous,source_sha256=sha(source),files={str(p):sha(p) for p in files},host_build=host,attribution=attribution)
    (out/'manifest.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    proposal=dict(operator='fix',id=out.name,source='main.py',source_sha256=sha(source),backend=frozen['backend'],
        implementation_language='python',native_manifest='manifest.json',attribution=attribution,
        summary='缓存Python分配接口并合并为一次扩展调用；每次新分配输出、重取地址和stream',
        prediction='小输入完整调用耗时下降；设备源码不变，设备收益不归于本次宿主改法')
    (out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='prepared',operator='fix',id=out.name)))

if __name__=='__main__':main()
