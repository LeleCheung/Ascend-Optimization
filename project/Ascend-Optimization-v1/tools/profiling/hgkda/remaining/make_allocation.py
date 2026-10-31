"""输出先由正常PyTorch分配接口分配，再交C++启动自写设备计算；不调用目标参考。"""
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    withdrawn={}
    for op,kind,param in [('fix',2,'self'),('hardswish',4,'x')]:
        old=BATCH/'proposals'/op/'ascendc-double-buffer-v4';out=BATCH/'proposals'/op/'ascendc-python-allocation-v5'
        out.mkdir(parents=True,exist_ok=True);previous=json.loads((old/'manifest.json').read_text());files=[]
        for path in previous['files']:
            source=Path(path)
            if source.name=='main.py':continue
            dest=out/source.name
            if dest.exists() and dest.read_bytes()!=source.read_bytes():raise FileExistsError(dest)
            dest.write_bytes(source.read_bytes());files.append(dest)
        body=(old/'main.py').read_text(encoding='utf8').replace(str(old),str(out))
        begin=body.rindex('\ndef run('+param+'):')
        body=body[:begin]+'''\ndef _allocated_fallback(x,out):
    return _original_run(x)


def run(PARAM):
    output=PARAM.new_empty(PARAM.shape)
    return _host.execute(PARAM,output,KIND,_allocated_fallback)
'''.replace('PARAM',param).replace('KIND',str(kind))
        source=out/'main.py';raw=body.encode('utf8')
        if source.exists() and source.read_bytes()!=raw:raise FileExistsError(source)
        source.write_bytes(raw);files.append(source)
        manifest=dict(previous,files={str(p):sha(p) for p in files},source_sha256=sha(source),
            attribution='Codex自写Ascend C双缓冲计算；Python仅分配新输出，C++做同stream任务队列启动；原KGS与补测')
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        proposal=dict(operator=op,id=out.name,source='main.py',source_sha256=sha(source),backend=manifest['backend'],
            implementation_language='python',native_manifest='manifest.json',attribution=manifest['attribution'],
            summary='输出通过输入new_empty分配；双缓冲Ascend C计算及C++启动；避开C++分配的挂起路径',
            prediction='保留输出独立性与当前stream；调查C++empty_like路径停滞，不将待证实原因写成结论')
        (out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        for name in ['ascendc-taskqueue-v1','ascendc-double-buffer-v4']:
            withdrawn[op+':'+name]='同一C++内部输出分配路径已在fix出现隔离worker停滞；停止新提交，改用Python分配输出的独立候选。未修改已完成原始结果。'
    (BATCH/'withdrawn-proposals.json').write_text(json.dumps(withdrawn,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Prepared 2 Python-allocation proposals; 4 pending unsafe allocation variants withdrawn')


if __name__=='__main__':main()
