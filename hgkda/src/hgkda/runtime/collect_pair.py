"""为当前生成候选与冻结CANN参考采集同输入指标；先从设备任务发现实际kernel。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def resolve_reported_symbol(alias,log):
    """从采集器实际报告的符号中消歧，不把运行时任务别名当作二进制符号。"""
    reported=sorted(set(re.findall(r'Kernel (\S+) skipped: not selected via --kernel-name',log)))
    operation=alias.rsplit('_',1)[-1]
    matches=[name for name in reported if name.startswith(operation+'_')]
    if not alias.startswith('aclnn') or len(matches)!=1:
        raise ValueError('任务别名无法与唯一实测二进制符号对应：'+str(dict(alias=alias,reported=reported)))
    return matches[0]


def resolve_collected_symbol(alias, actual, log, side):
    """Accept a unique CANN binary specialization explicitly reported by msprof."""
    reported=set(re.findall(r'Start analyze kernel on device:[^\n]*?kernel:\s*([^,\s]+)',log))
    operation=alias.rsplit('_',1)[-1] if alias.startswith('aclnn') else alias
    if side!='reference' or reported!={actual} or not actual.startswith(operation+'_'):
        raise ValueError('Collected binary differs from the discovered task without a unique specialization binding')
    return actual


def partition_profile_tasks(tasks):
    """Keep DMA events explicit while counting AI-core metric reports separately."""
    compute=[];other=[]
    for task in tasks:
        kind=task.get('task_type','')
        if kind.startswith('KERNEL_AI'):compute.append(task)
        elif kind=='SDMA_SQE':other.append(task)
        else:raise ValueError('Unclassified device task requires explicit profiling support: '+kind)
    if not compute:raise ValueError('No AI-core task available for metric collection')
    return compute,other


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='只补齐已有采集，已完成侧不重复采集')
    a=p.parse_args()
    import fcntl
    c=read(a.config);package=Path(c['package']);run=Path(c['run_directory'])
    source=a.source.resolve();digest=sha(source)
    if a.resume:assert a.output.is_dir()
    else:a.output.mkdir(parents=True,exist_ok=False)
    leaf=run/'reference-catalog'
    case=next(json.loads(line) for line in (leaf/'timing.jsonl').read_text().splitlines() if json.loads(line)['name']==a.case)
    spec=a.output/'spec.json'
    if a.resume:
        previous=read(spec);assert previous['source_sha256']==digest and previous['case']==case
    else:
        frozen=a.output/'candidate-source.py';shutil.copyfile(source,frozen)
        save(spec,dict(operator=c['operator'],source=str(frozen),source_sha256=digest,leaf=str(leaf),case=case,
             reference_kind=c['job']['reference_kind'],entrypoint=c.get('capture_entrypoint','run'),
             warmup_before_capture=c.get('capture_warmup',0),ir_directory=str(a.output/'actual-ir')))
    environment=dict(ASCEND_RT_VISIBLE_DEVICES=str(c['device']),OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
          PYTHONPATH=c.get('device_pythonpath', ''),TRITON_CACHE_DIR=str(a.output/'profile-triton-cache'))
    def device(label,arguments):
        shell='source '+shlex.quote(str(Path(c['sdk'])/'set_env.sh'))+'\n'
        shell+='\n'.join('export '+k+'='+shlex.quote(v) for k,v in environment.items())+'\n'
        shell+='exec '+shlex.join(list(map(str,arguments)))
        with (a.output/(label+'.log')).open('xb') as log:
            result=subprocess.run(['docker','exec','-w',str(a.output),c['container'],'bash','-c',shell],stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        assert result.returncode==0,label+': '+(a.output/(label+'.log')).read_text(errors='replace')[-1800:]
    with Path(c['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for side in ['candidate','reference']:
            folder=a.output/side
            if (folder/'collection.json').exists():
                meta=read(folder/'collection.json')
                assert meta['status']=='completed' and meta['spec_sha256']==sha(spec)
                continue
            trace=a.output/(side+'-discovery.json')
            if not trace.exists():
                device(side+'-discover',[c['device_python'],Path(__file__).with_name('discover_kernel.py'),
                       '--spec',spec,'--side',side,'--output',trace])
            assert read(trace)['source_sha256']==digest and read(trace)['side']==side
            discovery=read(trace);names=discovery['kernels'];assert names
            if len(names)>1:
                # 两侧均保留完整调用中的所有任务；模拟器另行显式选定目标。
                # 同一API可以多次调用同名TransData，但每次对应不同二进制；
                # 不用别名猜映射，直接采集本次调用的全部实际二进制任务。
                all_profile=folder/'all-kernels'
                compute_tasks,non_metric_tasks=partition_profile_tasks(discovery['tasks'])
                task_count=len(compute_tasks)
                application=shlex.join([c['device_python'],str(Path(__file__).with_name('paired_driver.py')),'--spec',str(spec),'--side',side])
                command=['msprof','op','--application='+application,'--output='+str(all_profile),
                         '--launch-count='+str(task_count),'--warm-up=0','--aic-metrics=Roofline']
                if not list(all_profile.rglob('OpBasicInfo*.csv')):device(side+'-all-kernels',command)
                basics=sorted(all_profile.rglob('OpBasicInfo*.csv'))
                assert len(basics)==task_count,'未完整采集同一API全部任务：'+str((len(basics),task_count))
                children=[]
                for index,basic in enumerate(basics):
                    with basic.open() as stream:rows=list(csv.DictReader(stream))
                    assert len(rows)==1
                    symbol=rows[0]['Op Name'];child=folder/('task-'+str(index).zfill(3));child.mkdir(exist_ok=True)
                    profile=child/'profile';profile.mkdir(exist_ok=True)
                    for table in basic.parent.glob('*.csv'):
                        target=profile/table.name
                        if target.exists():assert sha(target)==sha(table)
                        else:shutil.copyfile(table,target)
                    save(child/'collection.json',dict(status='completed',side=side,expected_kernel=symbol,groups='Roofline',
                         task_alias=None,spec_sha256=sha(spec),command=command,discovery_sha256=sha(trace),
                         raw_basic=dict(path=str(basic),sha256=sha(basic))))
                    children.append(dict(directory=child.name,symbol=symbol,occurrences=1))
                save(folder/'collection.json',dict(status='completed',side=side,groups='Roofline',spec_sha256=sha(spec),
                    kernels=children,discovery_sha256=sha(trace),runtime_tasks=discovery['tasks'],
                    non_metric_tasks=non_metric_tasks,
                    scope='AI-core任务逐项保存指标；SDMA任务保留发现trace与耗时，完整调用由正式评测覆盖。'))
                continue
            folder.mkdir(exist_ok=True);symbol=names[0]
            application=shlex.join([c['device_python'],str(Path(__file__).with_name('paired_driver.py')),'--spec',str(spec),'--side',side])
            first_log=a.output/(side+'-collect.log')
            resolution=None
            if first_log.exists() and not list((folder/'profile').rglob('OpBasicInfo*.csv')):
                symbol=resolve_reported_symbol(symbol,first_log.read_text(errors='replace'))
                resolution=dict(task_alias=names[0],reported_binary_symbol=symbol,log_sha256=sha(first_log))
                if (folder/'profile').exists():(folder/'profile').rename(folder/'profile-unmatched')
            command=['msprof','op','--application='+application,'--output='+str(folder/'profile'),
                     '--kernel-name='+symbol,'--launch-count=1','--warm-up=5','--aic-metrics=Roofline']
            label=side+'-collect-symbol' if resolution else side+'-collect'
            if not (a.resume and list((folder/'profile').rglob('OpBasicInfo*.csv'))):
                device(label,command)
            if not list((folder/'profile').rglob('OpBasicInfo*.csv')) and resolution is None:
                symbol=resolve_reported_symbol(symbol,first_log.read_text(errors='replace'))
                resolution=dict(task_alias=names[0],reported_binary_symbol=symbol,log_sha256=sha(first_log))
                if (folder/'profile').exists():(folder/'profile').rename(folder/'profile-unmatched')
                command=[('--kernel-name='+symbol if word.startswith('--kernel-name=') else word) for word in command]
                device(side+'-collect-symbol',command)
            basic=list((folder/'profile').rglob('OpBasicInfo*.csv'));assert len(basic)==1
            with basic[0].open() as stream:rows=list(csv.DictReader(stream))
            assert len(rows)==1
            if rows[0]['Op Name']!=symbol:
                log_path=a.output/(label+'.log')
                symbol=resolve_collected_symbol(symbol,rows[0]['Op Name'],log_path.read_text(errors='replace'),side)
                resolution=dict(task_alias=names[0],reported_binary_symbol=symbol,log_sha256=sha(log_path),
                    basis='unique analyzed binary specialization in the same msprof invocation')
            save(folder/'collection.json',dict(status='completed',side=side,expected_kernel=symbol,groups='Roofline',
                task_alias=names[0],symbol_resolution=resolution,
                spec_sha256=sha(spec),command=command,discovery_sha256=sha(trace),
                driver_sha256=sha(Path(__file__).with_name('paired_driver.py'))))
    assert sha(source)==digest,'采集期间候选源码发生变化'
    comparison=a.output/'comparison.json';context=a.output/'context.json'
    subprocess.run([sys.executable,'-m','hgkda','compare','--directory',str(a.output),'--output',str(comparison)],check=True)
    subprocess.run([sys.executable,'-m','hgkda','context','--operator',c['operator'],'--source',str(source),
         '--comparison',str(comparison),'--ir-manifest',str(a.output/'actual-ir/manifest.json'),'--output',str(context)],check=True)
    save(run/'workspace/paired-context-pointer.json',dict(source_sha256=digest,context=str(context),sha256=sha(context)))
    print(json.dumps(dict(status='completed',source_sha256=digest,case=a.case,context=str(context))))


if __name__=='__main__':
    from hgkda.runtime.trace import traced_main
    traced_main(main, 'pair')
