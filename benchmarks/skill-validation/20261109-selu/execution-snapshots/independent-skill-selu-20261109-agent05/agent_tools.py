"""独立skill试验的有限工具桥：整理上下文或预检候选，不启动模型或KG。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from hgkda.artifacts import unwrap
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
    run=Path(config['run_directory']);repo=Path(config['repository']);workspace=run/'workspace'
    if args.action=='context':
        comparison=workspace/'comparison.json';context=workspace/'context.json'
        if not comparison.exists():
            subprocess.run([sys.executable,'-m','hgkda','compare','--directory',config['paired_directory'],
                            '--output',str(comparison)],check=True,stdout=subprocess.DEVNULL)
        if not context.exists():
            subprocess.run([sys.executable,'-m','hgkda','context','--operator','selu_',
                '--source',str(workspace/'starter.py'),'--comparison',str(comparison),
                '--ir-manifest',str(run/'baseline-ir/manifest.json'),'--output',str(context)],
                check=True,stdout=subprocess.DEVNULL)
        data=unwrap(read(context));paired=data['facts']['paired']
        local_ir=workspace/'actual-ir';local_ir.mkdir(exist_ok=True)
        for item in data['facts']['compiled_ir']:
            shutil.copyfile(item['path'],local_ir/Path(item['path']).name)
        selected=[row for row in paired['differences'] if any(k in row['metric'] for k in
                    ['ratio','time','instructions','datas'])]
        selected.sort(key=lambda row:abs((row.get('candidate_over_reference') or 1)-1),reverse=True)
        brief=dict(source_sha256=data['source']['sha256'],case=data['case'],reference_kind=data['reference_kind'],
            differences=selected[:12],transfer_granularity=paired['derived_differences'],hypotheses=data['hypotheses'],
            mechanisms=data['knowledge']['mechanisms'],full_context=str(context),
            actual_ir=[str(local_ir/Path(row['path']).name) for row in data['facts']['compiled_ir']],
            reference='CANN原生；正式性能只由KGS完整评测确定')
        (workspace/'context-brief.json').write_text(json.dumps(brief,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(brief,ensure_ascii=False));return
    assert args.source is not None
    source=args.source.resolve()
    assert source.is_relative_to(workspace.resolve()) and source.suffix=='.py'
    text=source.read_bytes().decode('utf8');inspection=inspect_text(text)
    if inspection['issues']:
        print(json.dumps(dict(status='static_rejected',issues=inspection['issues']),ensure_ascii=False));return
    # 本试验不接受未提供源码的二进制或动态外部执行依赖。
    import ast
    for node in ast.walk(ast.parse(text)):
        if isinstance(node,ast.Import):names=[item.name for item in node.names]
        elif isinstance(node,ast.ImportFrom):names=[node.module or '']
        else:continue
        assert all(name.split('.')[0] in {'torch','torch_npu','triton','math','functools','typing'} for name in names),names
    digest=sha(source);attempt=run/'attempts'/digest
    if (attempt/'preflight.json').exists():
        previous=read(attempt/'preflight.json')
        (workspace/'latest-preflight.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(previous,ensure_ascii=False));return
    assert len(list((run/'attempts').glob('*/preflight.json'))) < 3, '本轮最多预检三份不同源码'
    attempt.mkdir(parents=True,exist_ok=False)
    frozen=attempt/'candidate.py';frozen.write_bytes(source.read_bytes())
    checks=config['job']['checks'];environment={
        'ASCEND_RT_VISIBLE_DEVICES':str(config['device']),'PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':'1','TRITON_CACHE_DIR':str(attempt/'triton-cache'),
        'PYTHONPATH':str(repo/'third_party/FlagGems/src')}

    def device(label,command):
        shell='source '+shlex.quote(str(Path(config['sdk'])/'set_env.sh'))+'\n'
        shell+='\n'.join('export '+key+'='+shlex.quote(value) for key,value in environment.items())+'\n'
        shell+='exec '+shlex.join([config['device_python'],*map(str,command)])
        return execute(['docker','exec',config['container'],'bash','-c',shell],attempt/(label+'.log'),cwd=repo)

    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        code=device('compiled',[repo/'tools/regression/capture_compiled.py','--source',frozen,
             '--leaf',checks['supplementary']['leaf'],'--output',attempt/'compiled'])
        if code==0:
            code=device('supplementary',[checks['supplementary']['checker'],'--operator','selu_',
                 '--source',frozen,'--leaf',checks['supplementary']['leaf'],'--output',attempt/'supplementary.json'])
    qualification=read(attempt/'supplementary.json') if (attempt/'supplementary.json').exists() else None
    result=dict(status='passed' if code==0 and qualification and qualification['passed']==qualification['total']==42 else 'failed',
        source_sha256=digest,frozen_source=str(frozen),
        supplementary=dict(passed=qualification['passed'],total=qualification['total']) if qualification else None,
        failed_cases=[dict(case=row['case'],error=row.get('error')) for row in (qualification or {}).get('cases',[]) if not row['passed']][:4],
        log_tail=(attempt/('supplementary.log' if qualification else 'compiled.log')).read_text(errors='replace')[-1800:] if code else '',
        formal_evaluation='待宿主模型完成逐路径源码审查后执行一次KGS完整评测')
    (attempt/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (workspace/'latest-preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
