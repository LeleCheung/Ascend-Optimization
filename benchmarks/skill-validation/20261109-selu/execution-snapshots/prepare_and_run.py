"""在独立运行目录采集固定起点证据，并让Claude宿主加载skill生成候选。"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['repository','regression-run','run-directory','lock','sdk','model-settings']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--container',required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--device-python',required=True)
    args=parser.parse_args()
    repo=args.repository.resolve();run=args.run_directory.resolve();run.mkdir(parents=True,exist_ok=False)
    state=dict(status='preparing',started_at=datetime.now(timezone.utc).isoformat())
    save(run/'controller-state.json',state)
    workspace=run/'workspace';workspace.mkdir()
    original=read(args.regression_run/'plan.json')
    job=next(row for row in original['jobs'] if row['operator']=='selu_')
    config=dict(repository=str(repo),run_directory=str(run),lock=str(args.lock),sdk=str(args.sdk),
        container=args.container,device=args.device,device_python=args.device_python,job=job,
        paired_directory=str(run/'baseline-paired'))
    save(run/'config.json',config)
    shutil.copytree(repo/'skills/hgkda',workspace/'skills/hgkda')
    (workspace/'.claude/skills').mkdir(parents=True)
    (workspace/'.claude/skills/hgkda').symlink_to(workspace/'skills/hgkda',target_is_directory=True)
    upstream=repo/'third_party/FlagGems/src/flag_gems/ops/selu_.py'
    starter=workspace/'starter.py'
    starter.write_bytes(upstream.read_bytes()+b'\n\ndef run(A):\n    return selu_(A)\n')
    save(workspace/'starter-provenance.json',dict(upstream=str(upstream),upstream_sha256=sha(upstream),
         starter_sha256=sha(starter),operation='仅追加run(A)接口；原计算与非连续回退保持原样',
         prior_best_source_provided=False))
    leaf=run/'reference-catalog'
    shutil.copytree(job['checks']['cann_supplementary']['leaf'],leaf)
    oracle=leaf/'oracle.py';text=oracle.read_bytes().decode('utf8')
    for row in original['compatibility_files']:
        if row['path'] in text:
            assert sha(row['archived'])==row['sha256']
            text=text.replace(row['path'],row['archived'])
    oracle.write_bytes(text.encode())
    case=next(json.loads(line) for line in (leaf/'timing.jsonl').read_text().splitlines()
              if json.loads(line)['name']=='selu_-timing-3')
    paired=run/'baseline-paired';paired.mkdir()
    spec=dict(operator='selu_',source=str(starter),source_sha256=sha(starter),leaf=str(leaf),case=case,
              reference_kind='qualified_CANN_native_single_api',ir_directory=str(run/'baseline-ir'))
    save(paired/'spec.json',spec)
    configuration=read(args.model_settings)
    environment=dict(os.environ)
    for key,value in configuration.get('env',{}).items():environment[key]=str(value)
    if 'ANTHROPIC_API_KEY' not in environment and environment.get('ANTHROPIC_AUTH_TOKEN'):
        environment['ANTHROPIC_API_KEY']=environment['ANTHROPIC_AUTH_TOKEN']

    def device(label,command):
        env=dict(ASCEND_RT_VISIBLE_DEVICES=str(args.device),OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
                 PYTHONPATH=str(repo/'third_party/FlagGems/src'),TRITON_CACHE_DIR=str(run/'profiling-triton-cache'))
        shell='source '+shlex.quote(str(args.sdk/'set_env.sh'))+'\n'
        shell+='\n'.join('export '+key+'='+shlex.quote(value) for key,value in env.items())+'\nexec '+shlex.join(command)
        with (run/(label+'.log')).open('xb') as output:
            process=subprocess.run(['docker','exec',args.container,'bash','-c',shell],stdout=output,stderr=subprocess.STDOUT,timeout=900)
        if process.returncode:raise RuntimeError(label+'返回'+str(process.returncode))

    try:
        with args.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state['status']='collecting_paired_profiling';save(run/'controller-state.json',state)
            for side,kernel in [('candidate','selu_kernel_'),('reference','Elu_ccb78600df288a689f3cc4abbf7e007e_high_performance_210010000')]:
                output=paired/side;output.mkdir()
                application=shlex.join([args.device_python,str(repo/'tools/skill_validation/paired_driver.py'),
                                        '--spec',str(paired/'spec.json'),'--side',side])
                command=['msprof','op','--application='+application,'--output='+str(output/'profile'),
                         '--kernel-name='+kernel,'--launch-count=1','--warm-up=5','--aic-metrics=Roofline']
                device('collect-'+side,command)
                basic=list((output/'profile').rglob('OpBasicInfo.csv'))
                assert len(basic)==1,(side,basic)
                import csv
                with basic[0].open() as stream:rows=list(csv.DictReader(stream))
                assert len(rows)==1 and rows[0]['Op Name']==kernel
                save(output/'collection.json',dict(status='completed',side=side,expected_kernel=kernel,
                     groups='Roofline',spec_sha256=sha(paired/'spec.json'),command=command,driver_sha256=sha(repo/'tools/skill_validation/paired_driver.py')))
        tool_command=shlex.join([sys.executable,str(repo/'tools/skill_validation/agent_tools.py'),
                                '--config',str(run/'config.json')])
        save(workspace/'task.json',dict(operator='selu_',source='starter.py',
            required_formal_cases=36,required_supplementary_each=42,tool_command=tool_command,
            current_best_source_read_allowed=False,target='同CANN参考完整调用争取超过0.8，至少相对固定起点产生有效优化'))
        prompt='''加载并遵循skills/hgkda/SKILL.md。任务见task.json。仅在当前工作目录读取和写入，不读取历史最佳源码或其他实验目录，不使用其他agent。
先调用工具桥的context动作读取同输入CANN成对profiling及实际IR摘要；必要时读取完整context.json。
从starter.py优化selu_，输出candidate.py及diagnosis.json。候选所有计算和搬运必须Triton，禁止原生/CANN/ATen fallback，包括非连续输入。
诊断需包含证据、机制、具体修改、预期变化、未决问题。优先简化连续分块、类型精度、非连续寻址和启动；当前普通Triton启动可先形成有效版本。
调用工具桥preflight --source <当前目录candidate.py>，Bash timeout设置600000。根据失败信息修复，最多三份不同候选，不重复同源码测试。全部42项预检通过后停止。
先不要自行提交正式评测或虚构成绩；宿主完成逐路径审查后会调用KGS，并将正式结果反馈给你。最终简短说明候选文件及预检状态。
工具桥命令：'''+tool_command
        (workspace/'prompt.md').write_text(prompt,encoding='utf8')
        state['status']='agent_generating_and_preflighting';save(run/'controller-state.json',state)
        command=['claude','-p','--bare','--output-format','stream-json','--verbose','--max-turns','18',
                 '--tools','Read,Write,Edit,Bash','--allowedTools','Read','Write','Edit',
                 'Bash('+str(sys.executable)+' '+str(repo/'tools/skill_validation/agent_tools.py')+' *)',
                 '--permission-mode','acceptEdits','--no-session-persistence']
        save(run/'host-invocation.json',dict(argv=command,model=configuration.get('model'),
            secret_environment_recorded=False,knowledge=str(repo/'src/hgkda/data/knowledge'),
            skill_sha256=sha(workspace/'skills/hgkda/SKILL.md')))
        with (run/'agent-events.jsonl').open('xb') as out,(run/'agent-stderr.log').open('xb') as err:
            process=subprocess.run(command,input=prompt.encode(),stdout=out,stderr=err,env=environment,cwd=workspace,timeout=2400)
        records=[]
        for line in (run/'agent-events.jsonl').read_bytes().splitlines():
            try:records.append(json.loads(line))
            except json.JSONDecodeError:pass
        result=next((r for r in reversed(records) if r.get('type')=='result'),{})
        save(run/'model-usage.json',dict(returncode=process.returncode,
            result={key:result.get(key) for key in ['subtype','is_error','num_turns','duration_ms','usage','modelUsage','total_cost_usd','result']}))
        state.update(status='awaiting_source_review' if process.returncode==0 and
                     (workspace/'latest-preflight.json').exists() and read(workspace/'latest-preflight.json')['status']=='passed'
                     else 'requires_attention',returncode=process.returncode)
        save(run/'controller-state.json',state)
    except Exception as error:
        state.update(status='requires_attention',error=type(error).__name__+': '+str(error));save(run/'controller-state.json',state)
        raise


if __name__=='__main__':main()
