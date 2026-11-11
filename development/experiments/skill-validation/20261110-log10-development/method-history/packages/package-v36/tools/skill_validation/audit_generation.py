"""核对方法包、起点及模型生成轨迹；不运行模型或改写候选源码。"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())


def initial_context_identity(call):
    """核对调用前实际提供的知识、源码和方法身份，避免仅核对配置声明。"""
    path=call/'initial-context.json'
    if not path.exists():return dict(status='not_recorded_legacy_call')
    context=read(path);provenance=read(call/'source-provenance.json')
    assert context['source_sha256']==provenance['input_sha256'],'调用初始上下文不属于输入源码'
    assert context['method_manifest_sha256']==sha(call/'method-manifest.json'),'调用初始上下文不属于本次方法'
    assert read(call/'initial-knowledge.json')['mechanisms']==context['mechanisms'],'调用初始知识与上下文不同'
    return dict(status='verified',source_sha256=context['source_sha256'],
                method_manifest_sha256=context['method_manifest_sha256'],knowledge_sha256=sha(call/'initial-knowledge.json'))


def replay_candidate(call,workspace,tools,results):
    """由成功的Write/Edit重放候选字节，核对记录的调用前后快照。"""
    path=call/'source-provenance.json'
    if not path.exists():return dict(status='not_recorded_legacy_call')
    provenance=read(path)
    assert sha(call/'input-source.py')==provenance['input_sha256']
    text=(call/'input-source.py').read_bytes().decode('utf8') if provenance['candidate_existed'] else None
    count=0
    for identity,tool in tools.items():
        if identity not in results or results[identity].get('is_error',False):continue
        payload=tool.get('input',{});raw=payload.get('file_path','')
        if tool['name'] not in ['Write','Edit'] or not raw:continue
        target=Path(raw);target=target if target.is_absolute() else workspace/target
        if target.resolve()!=(workspace/'candidate.py').resolve():continue
        if tool['name']=='Write':text=payload['content']
        else:
            assert text is not None,'候选尚未创建却执行Edit'
            before,after=payload['old_string'],payload['new_string']
            assert before and before in text,'Edit旧文本与调用快照不一致'
            if payload.get('replace_all',False):text=text.replace(before,after)
            else:
                assert text.count(before)==1,'非全量Edit旧文本并不唯一'
                text=text.replace(before,after,1)
        count+=1
    if 'output_sha256' in provenance:
        assert text is not None
        digest=hashlib.sha256(text.encode('utf8')).hexdigest()
        assert digest==provenance['output_sha256']==sha(call/'output-source.py'),'模型工具记录无法重放输出源码'
    else:assert text is None and not (call/'output-source.py').exists()
    return dict(status='verified',candidate_write_actions=count,**provenance)


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
            if tool['name']=='Read' and path:
                target=Path(path);target=target if target.is_absolute() else workspace/target
                if not target.resolve().is_relative_to(workspace.resolve()):
                    outside_reads.append(dict(invocation=call.name,path=path,success=success))
        calls.append(dict(invocation=call.name,model=invocation.get('model'),started_at=invocation.get('started_at'),prompt_sha256=invocation['prompt_sha256'],
            runner_sha256=invocation['runner_sha256'],tool_bridge_sha256=invocation['tool_bridge_sha256'],tools=len(tools),
            candidate_replay=replay_candidate(call,workspace,tools,results),
            initial_context=initial_context_identity(call)))
    assert not any(row['success'] for row in outside_reads),'模型成功读取了workspace外文件，须审查信息来源'
    frozen_budget=None
    if config['phase']=='frozen':
        frozen_budget=read(run/'frozen-budget.json')
        assert len(calls)<=frozen_budget['maximum_model_calls']
        assert len({row['prompt_sha256'] for row in calls})==1,'冻结调用使用了不同提示词'
        assert config['maximum_preflight_sources']==frozen_budget['maximum_preflight_sources']
        assert config['maximum_formal_sources']==frozen_budget['maximum_formal_sources']
        for call in (run/'agent-calls').iterdir():
            invocation=read(call/'host-invocation.json');argv=invocation['argv']
            assert int(argv[argv.index('--max-turns')+1])==frozen_budget['max_turns_each']
            for key in ['thinking_tokens','max_output_tokens','effort']:
                assert invocation['context_settings'][key]==frozen_budget[key]
            assert invocation['model']==read(run/'model-settings.json')['model']
        assert all(row['candidate_replay']['status']=='verified' for row in calls)
        assert all(row['initial_context']['status']=='verified' for row in calls),'冻结调用缺少初始上下文身份原件'
        ordered=sorted(calls,key=lambda row:row['started_at'])
        previous=None
        for row in ordered:
            source=row['candidate_replay']
            assert source['candidate_existed']==(previous is not None)
            assert source['input_sha256']==(previous or starter['starter_sha256']),'调用之间出现无法解释的源码变化'
            previous=source.get('output_sha256')
        if previous is not None:assert sha(workspace/'candidate.py')==previous
        assert len(list(run.glob('formal-*/plan.json')))<=frozen_budget['maximum_formal_sources']
        assert len(list(run.glob('attempts/*/preflight.json')))<=frozen_budget['maximum_preflight_sources']
    return dict(status='passed',operator=config['operator'],phase=config['phase'],
        method_manifest_sha256=sha(run/'method-manifest.json'),starter_sha256=starter['starter_sha256'],
        method_files=len(method['files']),model_calls=calls,model_write_actions=writes,
        outside_read_attempts=outside_reads,frozen_budget_verified=frozen_budget,
        scope='核对原件及已记录的Read/Write/Edit/Bash调用；不将调用日志视作文件系统隔离证明')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    value=audit(a.directory)
    a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=value['status'],model_calls=len(value['model_calls']),method_files=value['method_files'])))


if __name__=='__main__':main()
