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
from run import Runner,BATCH,ROOT,TOOLS,PY,read,save,sha
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
    assert not static(source.read_text()),'静态入口尚未通过'
    original=BATCH/'original-catalog/ops'/op
    runtime=out/'selected-runtime.json'
    if not runtime.exists():runner.cmd('selected-runtime-'+op,[PY,TOOLS/'strict_triton/runtime_audit.py',
         '--operator',op,'--source',source,'--leaf',original,'--output',runtime],1200)
    assert read(runtime)['status']=='passed' and read(runtime)['source_sha256']==sha(source)
    cases=[json.loads(x) for x in (original/'timing.jsonl').read_text().splitlines() if x.strip()]
    case=cases[0];directory=out/'actual-device';directory.mkdir(exist_ok=True)
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
    save(out/'backend-verification.json',proof)
    best.update(review_status='reviewed_strict_triton',compiled_status='verified',backend_verification=str(out/'backend-verification.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--operator',action='append');args=parser.parse_args()
    state=read(BATCH/'state.json')
    assert state['status'] in {'search_completed_pending_review','backend_verified'},'搜索尚未结束，不与设备队列并发'
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        runner=Runner()
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
