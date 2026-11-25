"""Capture a single-core simulator using discovered candidate symbols, bind region timelines, and refresh actual model context."""
import argparse
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


def select_kernel(names, requested=None):
    """Require an explicit observed kernel for a multi-kernel invocation."""
    names = list(dict.fromkeys(names))
    if requested is not None:
        if requested not in names:
            raise ValueError('Selected kernel absent from current source discovery')
        return requested
    if len(names) != 1:
        raise ValueError('Specify --kernel-name for the observed multi-kernel invocation')
    return names[0]


def freeze_autotune_replay(pair, out, digest, case):
    """Bind simulator configuration to the same source and measured input."""
    replay_path=pair/'actual-ir/autotune-cache.json'
    if not replay_path.exists():
        return {}, None
    replay=read(replay_path)
    if replay['source_sha256']!=digest or replay['case']!=case:
        raise ValueError('Autotune replay source or case differs from measured invocation')
    if not any(entry['cache'] for entry in replay['entries']):
        return {}, None
    frozen=out/'autotune-replay.json'
    frozen.write_bytes(replay_path.read_bytes())
    return dict(autotune_replay=str(frozen),warmup_before_capture=0),sha(frozen)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--paired-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--soc',required=True)
    p.add_argument('--simulator-library',type=Path,required=True)
    p.add_argument('--core-id',type=int,default=0)
    p.add_argument('--kernel-name',help='Explicitly select a discovered candidate symbol from a multi-kernel call')
    p.add_argument('--kernel-replay',type=Path,help='Target launch inputs saved on hardware, used only for isolated simulator analysis')
    p.add_argument('--timeout-minutes',type=int,default=5)
    p.add_argument('--trace-unit',choices=['us','ns','cycles'],required=True,
                   help='Raw trace ts/dur unit, confirmed from the tool format; displayTimeUnit does not define conversion')
    p.add_argument('--parse-only',action='store_true',help='Parse completed captures in this directory without rerunning the simulator')
    p.add_argument('--allow-postcapture-exit',action='store_true',
                   help='Allow exit 139 only after target synchronization and IR-manifest persistence; reject timeouts and Python exceptions')
    a=p.parse_args()
    if a.timeout_minutes<1:p.error('timeout-minutes must be positive')
    import fcntl
    c=read(a.config);pair=a.paired_directory.resolve();out=a.output.resolve()
    spec=read(pair/'spec.json');source=Path(spec['source']);digest=sha(source)
    assert digest==spec['source_sha256']
    assert digest==sha(Path(c['run_directory'])/'workspace/candidate.py'),'Capture only the current candidate'
    discovery=read(pair/'candidate-discovery.json');names=discovery['kernels']
    kernel=select_kernel(names,a.kernel_name)
    from hgkda.gate import admit
    admit(source.read_text(encoding='utf8'),read(Path(c['run_directory'])/'source-reviews'/(digest+'.json')))
    if not a.parse_only:
        out.mkdir(parents=True,exist_ok=False)
        # Hardware and simulator compiled artifacts may differ; keep the same source/case in separate directories.
        capture_spec=out/'spec.json'
        capture_values=dict(spec,ir_directory=str(out/'actual-ir'))
        if spec.get('ir_input'):
            from hgkda.runtime.ir_input import binary_bindings, freeze
            alias=c.get('simulator_target_alias')
            if alias is not None:
                assert alias['observed_arch']==a.soc
                alias=dict(alias,binary_sha256=binary_bindings(spec['ir_input']['manifest']))
            frozen,binding=freeze(spec['ir_input']['manifest'],out/'implementation',source,
                runtime_path=c.get('ir_capture_runtime'),analysis_target_alias=alias)
            assert sha(frozen)==digest
            capture_values.update(source=str(frozen),ir_input=binding)
        replay_values,replay_sha=freeze_autotune_replay(pair,out,digest,spec['case'])
        capture_values.update(replay_values)
        if a.kernel_replay:
            from hgkda.runtime.launch_replay import freeze
            replay=freeze(a.kernel_replay,out/'launch-replay',digest,spec['case'],kernel)
            capture_values['kernel_replay']=dict(replay,kernel=kernel)
        save(capture_spec,capture_values)
        app=shlex.join([c['device_python'],str(Path(__file__).with_name('paired_driver.py')),
                       '--spec',str(capture_spec),'--side','candidate'])
        command=['msprof','op','simulator','--application='+app,'--kernel-name='+kernel,
                 '--soc-version='+a.soc,'--core-id='+str(a.core_id),'--launch-count=1',
                 '--dump=on','--timeout='+str(a.timeout_minutes),'--output='+str(out/'raw/kernel-000')]
        save(out/'request.json',dict(source_sha256=digest,case=spec['case'],kernel=kernel,
             core_id=a.core_id,soc=a.soc,trace_time_unit=a.trace_unit,command=command,
             paired_spec_sha256=sha(pair/'spec.json'),capture_spec_sha256=sha(capture_spec),collector_sha256=sha(__file__),
             autotune_cache_sha256=replay_sha))
        environment=dict(ASCEND_RT_VISIBLE_DEVICES=str(c['device']),OMP_NUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1',TRITON_DISABLE_LINE_INFO='false',
            TRITON_CACHE_DIR=str(out/'compiler-cache'),
            PYTHONPATH=c.get('device_pythonpath', ''))
        if spec.get('ir_input'):
            # Preserve the original capture compiler-default line settings and compiler environment identity.
            environment.pop('TRITON_DISABLE_LINE_INFO')
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
    assert request['kernel']==kernel
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
            basis='An IR manifest in an isolated empty directory is written after the candidate call and torch.npu.synchronize return successfully, as ordered by the current driver source',
            driver_sha256=sha(Path(__file__).with_name('paired_driver.py')),
            limitation='Target kernel synchronized successfully, but the application later exited with 139; retain the abnormal exit rather than describing a clean process run')
        save(out/'postcapture-exit-qualification.json',qualification)
    else:qualification['kernel_completed']=True
    from hgkda.simulator import parse, validate_time_unit
    debug_source=Path(spec.get('debug_source',source))
    assert sha(debug_source)==digest,'Debug source differs from the frozen operator'
    verified_sources={str(path):dict(text=path.read_text(encoding='utf8'),sha256=digest)
        for path in [source,debug_source]}
    listing=parse(out/'raw/kernel-000',kernel,debug_source,digest,spec['case'])
    assert listing['kernels'] and all(k['name']==kernel for k in listing['kernels'])
    save(out/'listing.json',listing)
    from hgkda.regions import analyze
    reports=[]
    for core in listing['kernels'][0]['cores']:
        traces=list((out/'raw/kernel-000').glob('OPPROF_*/simulator/'+core['name']+'/trace.json'))
        assert len(traces)==1,'Original timeline is not unique'
        raw=read(traces[0]);trace=dict(raw) if isinstance(raw,dict) else dict(traceEvents=raw)
        unit_validation=validate_time_unit(raw,core['instructions'],a.trace_unit)
        trace['hgkda_binding']=dict(source_sha256=digest,case=spec['case'],kernel=kernel,
            core=core['name'],time_unit=a.trace_unit,raw_trace_sha256=sha(traces[0]),
            raw_trace=str(traces[0]),unit_validation=unit_validation,
            scope='One kernel on one hardware core from one simulator capture, not a whole-device trace')
        target=out/(core['name']+'-bound-trace.json');save(target,trace)
        result=analyze(listing,trace,kernel=kernel,core=core['name'],trace_unit=a.trace_unit,
                       sources=verified_sources)
        result['collection_qualification']=qualification
        if not qualification['process_clean_exit']:result['limitations'].append(qualification['limitation'])
        save(out/(core['name']+'-regions.json'),result)
        reports.append(dict(core=core['name'],regions=str(out/(core['name']+'-regions.json')),
                           trace_sha256=sha(target),timeline_status=result['timeline']['status']))
    assert reports and all(row['timeline_status']=='completed' for row in reports)
    # Analyze AIV subcores separately rather than mixing timelines; select one explicitly for the main context.
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
    from hgkda.runtime.trace import traced_main
    traced_main(main, 'simulate')
