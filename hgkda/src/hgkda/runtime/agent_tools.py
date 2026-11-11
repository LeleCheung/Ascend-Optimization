"""独立skill试验的有限工具桥：整理上下文或预检候选，不启动模型或KG。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from hgkda.gate import inspect_text


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(command,log,*,cwd,timeout=1500):
    with log.open('xb') as output:
        result=subprocess.run(list(map(str,command)),stdout=output,stderr=subprocess.STDOUT,cwd=cwd,timeout=timeout)
    return result.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('action',choices=['context','preflight'])
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    config=read(args.config)
    run=Path(config['run_directory']);repo=Path(config.get('repository', config['run_directory']));workspace=run/'workspace'
    if args.action=='context':
        if config.get('context_kind')=='operator_transfer':
            from hgkda.runtime.transfer_context import build
            brief=build(config)
            print(json.dumps(dict(status='completed',source_sha256=brief['source_sha256'],
                output=str(workspace/'context-brief.json'),retrieval=brief['knowledge_retrieval']),ensure_ascii=False))
            return
        raise ValueError('Public runtime requires context_kind=operator_transfer')
    assert args.source is not None
    source=args.source.resolve()
    assert source.is_relative_to(workspace.resolve()) and source.suffix=='.py'
    if config.get('context_kind')=='operator_transfer' and source!=workspace.resolve()/'candidate.py':
        print(json.dumps(dict(status='source_rejected',reason='只接受workspace/candidate.py算子实现；不运行起点或测试／探测脚本'),ensure_ascii=False));return
    text=source.read_bytes().decode('utf8');inspection=inspect_text(text)
    if inspection['issues']:
        print(json.dumps(dict(status='static_rejected',issues=inspection['issues']),ensure_ascii=False));return
    # 本试验不接受未提供源码的二进制或动态外部执行依赖。
    import ast
    for node in ast.walk(ast.parse(text)):
        if isinstance(node,ast.Import):names=[item.name for item in node.names]
        elif isinstance(node,ast.ImportFrom):names=[node.module or '']
        else:continue
        assert all(name.split('.')[0] in {'torch','torch_npu','triton','math','functools','typing','os'} for name in names),names
    # os仅用于当前编译模式的环境元数据，不允许额外宿主执行或文件操作。
    for node in ast.walk(ast.parse(text)):
        if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='os':
            assert node.attr in {'environ','getenv'},'os仅允许读取环境模式元数据'
    digest=sha(source);attempt=run/'attempts'/digest
    if config.get('require_preflight_review'):
        from hgkda.gate import admit
        review=run/'source-reviews'/(digest+'.json')
        if not review.exists():
            pending=dict(status='awaiting_source_review',source_sha256=digest,source=str(source),
                reason='设备操作前等待外部只读路径与地址边界审查；保持当前候选不变并返回')
            (workspace/'latest-preflight.json').write_text(json.dumps(pending,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            print(json.dumps(pending,ensure_ascii=False));return
        admit(text,read(review))
    if (attempt/'preflight.json').exists():
        previous=read(attempt/'preflight.json')
        (workspace/'latest-preflight.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(previous,ensure_ascii=False));return
    limit=config.get('maximum_preflight_sources',3)
    assert len(list((run/'attempts').glob('*/preflight.json'))) < limit, '已达到本轮显式配置的设备预检数量限制'
    attempt.mkdir(parents=True,exist_ok=False)
    frozen=attempt/'candidate.py';frozen.write_bytes(source.read_bytes())
    checks=config['job']['checks'];environment={
        'ASCEND_RT_VISIBLE_DEVICES':str(config['device']),'PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':'1','TRITON_CACHE_DIR':str(attempt/'triton-cache'),
        'PYTHONPATH':config.get('device_pythonpath', '')}

    def device(label,command):
        shell='source '+shlex.quote(str(Path(config['sdk'])/'set_env.sh'))+'\n'
        shell+='\n'.join('export '+key+'='+shlex.quote(value) for key,value in environment.items())+'\n'
        shell+='exec '+shlex.join([config['device_python'],*map(str,command)])
        return execute(['docker','exec',config['container'],'bash','-c',shell],attempt/(label+'.log'),cwd=repo)

    import fcntl  # 仅Linux设备执行阶段需要文件锁；离线上下文与帮助不依赖它。
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        code=device('compiled',[Path(__file__).with_name('capture_compiled.py'),'--source',frozen,
             '--leaf',checks['supplementary']['leaf'],'--output',attempt/'compiled'])
        if code==0:
            code=device('supplementary',[checks['supplementary']['checker'],'--operator',config['job']['operator'],
                 '--source',frozen,'--leaf',checks['supplementary']['leaf'],'--output',attempt/'supplementary.json'])
    qualification=read(attempt/'supplementary.json') if (attempt/'supplementary.json').exists() else None
    expected=checks['supplementary']['previous_total']
    from hgkda.validation import supplement_status
    qualified=qualification and supplement_status(config['job']['operator'],qualification,digest)=='passed'
    result=dict(status='passed' if code==0 and qualified and qualification['passed']==qualification['total']==expected else 'failed',
        source_sha256=digest,frozen_source=str(frozen),
        supplementary=dict(passed=qualification['passed'],total=qualification['total']) if qualification else None,
        failed_cases=[dict(case=row['case'],error=row.get('error')) for row in (qualification or {}).get('cases',[]) if not row['passed']][:4],
        log_tail=(attempt/('supplementary.log' if qualification else 'compiled.log')).read_text(errors='replace')[-1800:] if code else '',
        formal_evaluation='待宿主模型完成逐路径源码审查后执行一次KGS完整评测')
    (attempt/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (workspace/'latest-preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
