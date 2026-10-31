"""为fix冻结只改变宿主分配GIL范围的候选。"""
import hashlib
import json
from pathlib import Path
import shutil

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-final-20261031'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    old=BATCH/'proposals/fix/event-large-v6';out=BATCH/'proposals/fix/queue-gil-v12'
    if out.exists():raise FileExistsError(out)
    host=json.loads((BATCH/'host-build-queue-gil-v12.json').read_text())
    previous=json.loads((old/'manifest.json').read_text());out.mkdir(parents=True)
    files=[]
    for name in ['libhgkda_remaining.so','pointwise.cpp','CMakeLists.txt']:
        shutil.copyfile(old/name,out/name);files.append(out/name)
    for path in host['files']:
        p=Path(path);shutil.copyfile(p,out/p.name);files.append(out/p.name)
    body=(old/'main.py').read_text().replace(str(old),str(out)).replace('_hgkda_remaining_host_v4','_hgkda_remaining_host_v12')
    body=body.replace(sha(old/'_hgkda_remaining_host_v4.so'),sha(out/'_hgkda_remaining_host_v12.so'))
    body=body.replace('    output=self.new_empty(self.shape)\n    return _host.execute(self,output,2,_allocated_fallback)','    return _host.execute(self,None,2,_original_run)')
    source=out/'main.py';source.write_text(body,encoding='utf8');files.append(source)
    attribution='Codex保留原同stream任务队列；C++每次新分配输出并释放GIL；设备源码与Triton布局回退不变'
    manifest=dict(previous,source_sha256=sha(source),files={str(p):sha(p) for p in files},host_build=host,attribution=attribution)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    proposal=dict(operator='fix',id=out.name,source='main.py',source_sha256=sha(source),backend=manifest['backend'],
        implementation_language='python',native_manifest='manifest.json',attribution=attribution,
        summary='输出分配在C++中释放GIL；保留当前stream队列及独立输出',
        prediction='减少宿主调用耗时；先通过对齐路径检查，再运行完整KGS，设备源码不变')
    (out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status='prepared',operator='fix',id=out.name)))

if __name__=='__main__':main()
