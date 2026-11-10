"""按已发现的候选符号采集单核模拟器，绑定区域时间轴并刷新实际模型上下文。"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--paired-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--soc',required=True)
    p.add_argument('--simulator-library',type=Path,required=True)
    p.add_argument('--core-id',type=int,default=0)
    p.add_argument('--timeout-minutes',type=int,default=5)
    p.add_argument('--trace-unit',choices=['us','ns','cycles'],required=True,
                   help='原始trace ts/dur单位，须依据工具格式确认，displayTimeUnit不是换算依据')
    p.add_argument('--parse-only',action='store_true',help='仅解析该目录已完成的采集，不重新运行模拟器')
    p.add_argument('--allow-postcapture-exit',action='store_true',
                   help='仅允许目标同步及IR清单落盘之后的139退出；不接受超时或Python异常')
    a=p.parse_args()
    if a.timeout_minutes<1:p.error('timeout-minutes必须为正数')
    c=read(a.config);pair=a.paired_directory.resolve();out=a.output.resolve()
    spec=read(pair/'spec.json');source=Path(spec['source']);digest=sha(source)
    assert digest==spec['source_sha256']
    assert digest==sha(Path(c['run_directory'])/'workspace/candidate.py'),'只采当前候选'
    discovery=read(pair/'candidate-discovery.json');names=discovery['kernels'];assert len(names)==1
    kernel=names[0]
    from hgkda.gate import admit
    admit(source.read_text(encoding='utf8'),read(Path(c['run_directory'])/'source-reviews'/(digest+'.json')))
    if not a.parse_only:
        out.mkdir(parents=True,exist_ok=False)
        # 真机和模拟器的编译产物可能不同；保留相同源码和case，分别写入独立目录。
        capture_spec=out/'spec.json'
        save(capture_spec,dict(spec,ir_directory=str(out/'actual-ir')))
        app=shlex.join([c['device_python'],str(Path(c['package'])/'tools/skill_validation/paired_driver.py'),
                       '--spec',str(capture_spec),'--side','candidate'])
        command=['msprof','op','simulator','--application='+app,'--kernel-name='+kernel,
                 '--soc-version='+a.soc,'--core-id='+str(a.core_id),'--launch-count=1',
                 '--dump=on','--timeout='+str(a.timeout_minutes),'--output='+str(out/'raw/kernel-000')]
        save(out/'request.json',dict(source_sha256=digest,case=spec['case'],kernel=kernel,
             core_id=a.core_id,soc=a.soc,trace_time_unit=a.trace_unit,command=command,
             paired_spec_sha256=sha(pair/'spec.json'),capture_spec_sha256=sha(capture_spec),collector_sha256=sha(__file__)))
        environment=dict(ASCEND_RT_VISIBLE_DEVICES=str(c['device']),OMP_NUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1',TRITON_DISABLE_LINE_INFO='false',
            TRITON_CACHE_DIR=str(out/'compiler-cache'),
            PYTHONPATH=str(Path(c['repository'])/'third_party/FlagGems/src'))
        shell='source '+shlex.quote(str(Path(c['sdk'])/'set_env.sh'))+'\n'
        shell+='\n'.join('export '+k+'='+shlex.quote(v) for k,v in environment.items())+'\n'
        shell+='export LD_LIBRARY_PATH='+shlex.quote(str(a.simulator_library))+':"${LD_LIBRARY_PATH:-}"\n'
        shell+='exec '+shlex.join(command)
        with Path(c['lock']).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with (out/'collector.log').open('xb') as log:
                result=subprocess.run(['docker','exec','-w',str(out),c['container'],'bash','-c',shell],
                                      stdout=log,stderr=subprocess.STDOUT,timeout=60*a.timeout_minutes+120)
        log=(out/'collector.log').read_text(errors='replace')
        codes=[int(code) for code in re.findall(r'Child process exited with status (\d+)',log)]
        save(out/'execution.json',dict(returncode=result.returncode,child_exit_codes=codes))
    request=read(out/'request.json')
    assert request['source_sha256']==digest and request['case']==spec['case']
    assert request['trace_time_unit']==a.trace_unit and request['soc']==a.soc and request['core_id']==a.core_id
    execution=read(out/'execution.json')
    qualification=dict(process_clean_exit=execution['returncode']==0 and not any(execution['child_exit_codes']))
    if not qualification['process_clean_exit']:
        assert a.allow_postcapture_exit and execution['returncode']==0
        assert execution['child_exit_codes'] and set(execution['child_exit_codes'])=={139}
        assert 'Traceback (most recent call last)' not in (out/'collector.log').read_text(errors='replace')
        capture_spec=read(out/'spec.json');assert sha(out/'spec.json')==request['capture_spec_sha256']
        assert capture_spec['case']==spec['case'] and capture_spec['source_sha256']==digest
        assert Path(capture_spec['ir_directory'])==out/'actual-ir'
        manifest=out/'actual-ir/manifest.json';compiled=read(manifest)
        assert compiled['source_sha256']==digest and compiled['files']
        for item in compiled['files']:assert sha(manifest.parent/item['path'])==item['sha256']
        qualification.update(kernel_completed=True,completion_manifest_sha256=sha(manifest),
            basis='独立空目录的IR清单在候选调用和torch.npu.synchronize成功返回后写入；当前驱动源码规定该顺序',
            driver_sha256=sha(Path(c['package'])/'tools/skill_validation/paired_driver.py'),
            limitation='目标kernel已同步完成，但应用随后以139退出；保留退出异常，不表述为干净进程运行')
        save(out/'postcapture-exit-qualification.json',qualification)
    else:qualification['kernel_completed']=True
    from hgkda.simulator import parse, validate_time_unit
    listing=parse(out/'raw/kernel-000',kernel,source,digest,spec['case'])
    assert listing['kernels'] and all(k['name']==kernel for k in listing['kernels'])
    save(out/'listing.json',listing)
    from hgkda.regions import analyze
    reports=[]
    for core in listing['kernels'][0]['cores']:
        traces=list((out/'raw/kernel-000').glob('OPPROF_*/simulator/'+core['name']+'/trace.json'))
        assert len(traces)==1,'时间轴原件不唯一'
        raw=read(traces[0]);trace=dict(raw) if isinstance(raw,dict) else dict(traceEvents=raw)
        unit_validation=validate_time_unit(raw,core['instructions'],a.trace_unit)
        trace['hgkda_binding']=dict(source_sha256=digest,case=spec['case'],kernel=kernel,
            core=core['name'],time_unit=a.trace_unit,raw_trace_sha256=sha(traces[0]),
            raw_trace=str(traces[0]),unit_validation=unit_validation,
            scope='单次模拟器采集的单kernel、单硬件核文件；不是整卡trace')
        target=out/(core['name']+'-bound-trace.json');save(target,trace)
        result=analyze(listing,trace,kernel=kernel,core=core['name'],trace_unit=a.trace_unit,
                       sources={str(source):dict(text=source.read_text(encoding='utf8'),sha256=digest)})
        result['collection_qualification']=qualification
        if not qualification['process_clean_exit']:result['limitations'].append(qualification['limitation'])
        save(out/(core['name']+'-regions.json'),result)
        reports.append(dict(core=core['name'],regions=str(out/(core['name']+'-regions.json')),
                           trace_sha256=sha(target),timeline_status=result['timeline']['status']))
    assert reports and all(row['timeline_status']=='completed' for row in reports)
    # 多个AIV子核分别分析；不将它们混合为一条时间轴，主上下文明确选定其中一个。
    selected=reports[0];context=out/'context.json'
    subprocess.run([sys.executable,'-m','hgkda','context','--operator',c['operator'],
        '--source',str(source),'--comparison',str(pair/'comparison.json'),'--listing',str(out/'listing.json'),
        '--regions',selected['regions'],'--ir-manifest',str(pair/'actual-ir/manifest.json'),
        '--output',str(context)],check=True)
    save(Path(c['run_directory'])/'workspace/paired-context-pointer.json',
         dict(source_sha256=digest,context=str(context),sha256=sha(context)))
    save(out/'summary.json',dict(status='completed' if qualification['process_clean_exit'] else 'completed_with_postcapture_exit',
         collection_qualification=qualification,source_sha256=digest,case=spec['case'],
         kernel=kernel,cores=reports,selected_core=selected['core'],context=str(context),
         parser=listing['parser'],
         raw_files=[dict(path=str(p.relative_to(out)),sha256=sha(p)) for p in (out/'raw').rglob('*') if p.is_file()]))
    print(json.dumps(dict(status='completed',context=str(context),cores=len(reports))))


if __name__=='__main__':
    main()
