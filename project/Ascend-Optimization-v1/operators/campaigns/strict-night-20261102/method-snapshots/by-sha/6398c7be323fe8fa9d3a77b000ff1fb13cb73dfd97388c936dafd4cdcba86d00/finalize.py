"""仅在队列终态后核验人工审查过的最佳源码；不自动签署人工审查。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from night_common import BATCH,ROOT,TOOLS,PY,read,save,sha
from admission import static


def compiled(manifest,source):
    value=read(manifest);assert value['spec']['source_sha256']==sha(source)
    assert value.get('entries'),'没有实际Triton编译对象'
    entries=[]
    for row in value['entries']:
        kinds=set()
        for artifact in row['artifacts']:
            p=Path(artifact['path']);assert p.is_file() and p.stat().st_size and sha(p)==artifact['sha256']
            kinds.add(artifact['kind'])
        assert {'ttir','npubin'}.issubset(kinds),(row['symbol'],kinds)
        entries.append(dict(symbol=row['symbol'],kinds=sorted(kinds)))
    return dict(status='verified',source_sha256=sha(source),manifest=str(manifest),manifest_sha256=sha(manifest),kernels=entries)


def verify(runner,op,best):
    source=Path(best['source']);out=source.parent;review=out/'review.json'
    assert sha(source)==best['source_sha256']
    manual=read(review)
    assert manual['status']=='reviewed_strict_triton' and manual['source_sha256']==sha(source)
    assert manual['all_branches_reviewed'] and manual['dependencies_reviewed']
    host_proof=None
    if manual.get('reviewed_host_bridges'):
        from host_glue import admit
        host_proof=admit(source,review)
    else:assert not static(source.read_text()),'静态入口尚未通过'
    original=BATCH/'original-catalog/ops'/op
    if op in {'round_out','special_round_out'}:
        from out_api_probe import qualified as qualified_out_api
        checker=HERE/'out_api_probe.py'
        frozen=out/('out-api-checker-'+sha(checker)+'.py')
        if frozen.exists():assert frozen.read_bytes()==checker.read_bytes()
        else:frozen.write_bytes(checker.read_bytes())
        probe=out/'out-api-probe.json'
        if not probe.exists():
            runner.cmd('selected-out-api-'+op,[PY,frozen,'--operator',op,
                '--source',source,'--leaf',original,'--output',probe],900)
        best['out_api_probe']=str(probe)
        best['out_api_probe_checker']=str(frozen)
        if not qualified_out_api(read(probe),sha(source)):
            best.update(qualification='failed_out_api_probe',compiled_status='not_qualified')
            return
    runtime=out/'selected-runtime.json'
    if not runtime.exists():runner.cmd('selected-runtime-'+op,[PY,TOOLS/'strict_triton/runtime_audit.py',
         '--operator',op,'--source',source,'--leaf',original,'--output',runtime],1200)
    assert read(runtime)['status']=='passed' and read(runtime)['source_sha256']==sha(source)
    cases=[json.loads(x) for x in (original/'timing.jsonl').read_text().splitlines() if x.strip()]
    formal=read(best['search_result'])
    times=[v for v in formal.get('per_workload',[]) if v.get('phase')=='timing' and v.get('latency_ms')]
    chosen=max(times,key=lambda v:v['latency_ms'])['uuid'] if times else cases[0]['name']
    case=next(v for v in cases if v['name']==chosen)
    directory=out/'actual-device';directory.mkdir(exist_ok=True)
    # 已有代表输入原件保持原样；新候选优先捕获实际最耗时路径，覆盖新persistent核。
    if (directory/'spec.json').exists():case=read(directory/'spec.json')['case']
    spec=directory/'spec.json';payload=dict(operator=op,source=str(source),source_sha256=sha(source),leaf=str(original),case=case)
    if spec.exists():assert read(spec)==payload
    else:save(spec,payload)
    manifest=directory/'ir/manifest.json'
    if not manifest.exists():
        assert not (directory/'ir').exists(),'未完成编译目录必须检查后再处理'
        runner.cmd('selected-ir-'+op,[PY,TOOLS/'next_phase/driver.py','--spec',spec,'--side','candidate','--ir',directory/'ir'],900)
    proof=compiled(manifest,source);save(directory/'compiled-verification.json',proof)
    trace=directory/'trace.json'
    if not trace.exists():runner.cmd('selected-trace-'+op,[PY,TOOLS/'next_phase/driver.py','--spec',spec,'--side','candidate','--trace',trace],600)
    events=read(trace);events=events.get('traceEvents',[]) if isinstance(events,dict) else events
    tasks=[e for e in events if e.get('ph')=='X' and ('Task Type' in e.get('args',{}) or 'kernel' in str(e.get('cat','')).lower())]
    assert tasks,'实际设备任务为空'
    proof=dict(status='verified',source_sha256=sha(source),review=str(review),review_sha256=sha(review),
               runtime=str(runtime),runtime_sha256=sha(runtime),compiled=str(directory/'compiled-verification.json'),
               trace=str(trace),trace_sha256=sha(trace),tasks=[{k:e.get(k) for k in ['name','cat','dur','args']} for e in tasks],
               scope='人工逐路径审查＋完整运行时＋代表输入实际Triton编译对象/设备trace；不冒称全形状trace')
    if host_proof:proof['host_glue']=host_proof
    save(out/'backend-verification.json',proof)
    if op=='fix' and manual.get('diagnostic_parent_source_sha256'):
        parent=BATCH/'admissions'/op/manual['diagnostic_parent_source_sha256']/'main.py'
        assert sha(parent)==manual['diagnostic_parent_source_sha256']
        probe_case=next(v for v in cases if v['name']==manual['diagnostic_case'])
        for label,path in [('parent',parent),('corrected',source)]:
            probe=out/'component-diagnostic'/label;probe_spec=probe.with_name(label+'-spec.json')
            payload=dict(operator=op,source=str(path),source_sha256=sha(path),leaf=str(original),case=probe_case)
            if probe_spec.exists():assert read(probe_spec)==payload
            else:save(probe_spec,payload)
            if not (probe/'result.json').exists():
                args=[PY,HERE/'component_probe.py','--spec',probe_spec,'--output',probe]
                key='selected-component-'+op+'-'+label
                if probe.exists():
                    assert (probe/'trace.json').exists() and not (probe/'host-samples.json').exists(), '只允许已核对的旧版本trace解析故障恢复'
                    failure=runner.q.runtime/(key+'.log')
                    assert failure.exists() and 'TypeError: can only concatenate str' in failure.read_text(), '不恢复未知故障'
                    args+=['--resume-after-parse-failure'];key+='-parse-recovery'
                runner.cmd(key,args,1200)
        best['component_diagnostic']=str(out/'component-diagnostic')
    best.update(review_status='reviewed_strict_triton',compiled_status='verified',backend_verification=str(out/'backend-verification.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--operator',action='append');args=parser.parse_args()
    state=read(BATCH/'state.json')
    assert state['status'] in {'search_completed_pending_review','backend_verified'},'搜索尚未结束，不与设备队列并发'
    spec=importlib.util.spec_from_file_location('strict_night_final_controller',HERE/'run.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        runner=controller.Runner()
        for op,row in runner.state['operators'].items():
            if args.operator and op not in args.operator:continue
            best=row.get('best') or {}
            if not best:continue
            verify(runner,op,best);runner.persist()
        remaining=[op for op,row in runner.state['operators'].items() if row.get('best') and row['best'].get('compiled_status')!='verified']
        if not remaining:runner.state['status']='backend_verified'
        runner.persist()
        print(json.dumps(dict(status=runner.state['status'],remaining=remaining),ensure_ascii=False))


if __name__=='__main__':main()
