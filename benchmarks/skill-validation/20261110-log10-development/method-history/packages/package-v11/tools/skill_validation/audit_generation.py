"""核对方法包、起点及模型生成轨迹；不运行模型或改写候选源码。"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())


def audit(run):
    config=read(run/'config.json');workspace=run/'workspace';method=read(run/'method-manifest.json')
    package=run/'method-snapshot' if (run/'method-snapshot').exists() else Path(config['package'])
    for item in method['files']:
        assert sha(package/item['path'])==item['sha256'],item['path']
        if item['path'].startswith('skills/hgkda/'):
            assert sha(workspace/item['path'])==item['sha256'],'运行中的skill被修改'
    starter=read(workspace/'starter-provenance.json')
    assert sha(workspace/'starter.py')==starter['starter_sha256']==starter['upstream_sha256']
    calls=[];writes=[];outside_reads=[]
    for call in sorted((run/'agent-calls').iterdir()):
        invocation=read(call/'host-invocation.json')
        assert sha(call/'prompt.md')==invocation['prompt_sha256']
        tools={};results={}
        for line in (call/'agent-events.jsonl').read_bytes().splitlines():
            try:entry=json.loads(line)
            except json.JSONDecodeError:continue
            message=entry.get('message',{})
            if not isinstance(message,dict):continue
            for block in message.get('content',[]):
                if not isinstance(block,dict):continue
                if block.get('type')=='tool_use':tools[block['id']]=block
                if block.get('type')=='tool_result':results[block['tool_use_id']]=block
        for identity,tool in tools.items():
            payload=tool.get('input',{});path=payload.get('file_path',payload.get('path',''))
            success=identity in results and not results[identity].get('is_error',False)
            if tool['name'] in ['Write','Edit'] or (tool['name']=='Bash' and any(k in payload.get('command','') for k in ['cat >','cat <<','tee ','cp ','mv '])):
                writes.append(dict(invocation=call.name,tool=tool['name'],path=path,
                    command=payload.get('command','')[:180],success=success))
            if tool['name']=='Read' and path and Path(path).is_absolute() and not Path(path).is_relative_to(workspace):
                outside_reads.append(dict(invocation=call.name,path=path,success=success))
        calls.append(dict(invocation=call.name,model=invocation.get('model'),prompt_sha256=invocation['prompt_sha256'],
            runner_sha256=invocation['runner_sha256'],tool_bridge_sha256=invocation['tool_bridge_sha256'],tools=len(tools)))
    assert not any(row['success'] for row in outside_reads),'模型成功读取了workspace外文件，须审查信息来源'
    return dict(status='passed',operator=config['operator'],phase=config['phase'],
        method_manifest_sha256=sha(run/'method-manifest.json'),starter_sha256=starter['starter_sha256'],
        method_files=len(method['files']),model_calls=calls,model_write_actions=writes,
        outside_read_attempts=outside_reads,
        scope='核对原件及已记录的Read/Write/Edit/Bash调用；不将调用日志视作文件系统隔离证明')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    value=audit(a.directory)
    a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=value['status'],model_calls=len(value['model_calls']),method_files=value['method_files'])))


if __name__=='__main__':main()
