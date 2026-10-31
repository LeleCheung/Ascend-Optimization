"""直接启动候选：设备沿用事件大块版，宿主变更独立登记。"""
import hashlib
import json
from pathlib import Path
import shutil

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-final-20261031'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    host=json.loads((BATCH/'host-build-direct-v11.json').read_text())
    for op in ['fix','expm1_']:
        old=BATCH/'proposals'/op/'event-large-v6';out=BATCH/'proposals'/op/'direct-host-v11'
        if out.exists():raise FileExistsError(out)
        previous=json.loads((old/'manifest.json').read_text());out.mkdir(parents=True)
        files=[]
        for name in ['libhgkda_remaining.so','pointwise.cpp','CMakeLists.txt']:
            shutil.copyfile(old/name,out/name);files.append(out/name)
        for path in host['files']:
            p=Path(path);shutil.copyfile(p,out/p.name);files.append(out/p.name)
        body=(old/'main.py').read_text().replace(str(old),str(out)).replace('_hgkda_remaining_host_v4','_hgkda_remaining_host_v11')
        body=body.replace(sha(old/'_hgkda_remaining_host_v4.so'),sha(out/'_hgkda_remaining_host_v11.so'))
        if op=='fix':
            body=body.replace('    output=self.new_empty(self.shape)\n    return _host.execute(self,output,2,_allocated_fallback)','    return _host.execute(self,None,2,_original_run)')
        source=out/'main.py';source.write_text(body,encoding='utf8');files.append(source)
        attribution='Codex直接启动Ascend C，分配及队列等待释放GIL，登记内存stream；原数值设备源码与Triton特殊布局回退保留'
        frozen=dict(previous,source_sha256=sha(source),files={str(p):sha(p) for p in files},host_build=host,attribution=attribution)
        (out/'manifest.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        proposal=dict(operator=op,id=out.name,source='main.py',source_sha256=sha(source),backend=frozen['backend'],
            implementation_language='python',native_manifest='manifest.json',attribution=attribution,
            summary='减少宿主调度层；先排空原stream任务保证顺序，记录分配器stream，当前地址每次重新读取',
            prediction='小输入完整调用延迟下降；原生路径特殊值、动态输入、stream及完整KGS必须通过')
        (out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='prepared',operators=['fix','expm1_'],id='direct-host-v11')))

if __name__=='__main__':main()
