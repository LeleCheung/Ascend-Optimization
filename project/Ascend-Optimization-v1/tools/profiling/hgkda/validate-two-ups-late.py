#!/usr/bin/env python3
"""在己方搜索暂停且持卡锁时，独立验收本批后续上采样候选；不改主队列状态。"""
import hashlib, importlib.util, json, math, shutil, urllib.request
from pathlib import Path

ROOT=Path('/data/hanle/ascend-optimization')
TOOLS=Path(__file__).resolve().parent
BATCH=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024'


def save(p,data):
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    spec=importlib.util.spec_from_file_location('late_validation_queue',TOOLS/'run-nightly.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    q=module.Queue(BATCH.name,ROOT,'tle_yy')
    out=BATCH/'results/upsample_bilinear2d/codex-late'
    out.mkdir(parents=True,exist_ok=False)
    candidates=[]
    for p in (BATCH/'research').glob('*probes*/*/timings.json'):
        d=json.loads(p.read_text())
        response=p.with_name('response.json')
        if not response.exists() or json.loads(response.read_text()).get('status')!='SUCCEEDED':continue
        times={r['case']:r['median_ms'] for r in d['rows'] if r['variant']=='parent'}
        rows=[r for r in d['rows'] if r['variant']!='parent']
        if len(times)!=2 or len(rows)!=2 or {r['case'] for r in rows}!=set(times):continue
        gain=math.exp(sum(math.log(times[r['case']]/r['median_ms']) for r in rows)/len(rows))
        name=p.parent.name
        source=BATCH/'knowledge'/('upsample-'+name+'.py')
        if source.is_file() and gain>1.08:
            assert hashlib.sha256(source.read_bytes()).hexdigest()==d['source_sha256']
            candidates.append((gain,name,source))
    # Two strongest mechanisms plus safe compiled-launch control, each source
    # receives one formal run.  Diagnostic numbers never become reported scores.
    selected=sorted(candidates,reverse=True)[:2]
    raw=next((r for r in candidates if r[1]=='safe_compiled_launch'),None)
    if raw and all(r[1]!=raw[1] for r in selected):selected.append(raw)
    records=[]
    for gain,name,source in selected:
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        leaf=out/digest;leaf.mkdir()
        target=leaf/'main.py';shutil.copyfile(str(source),str(target))
        script='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util,json
from kernelgen_client.catalog import Catalog
from kernelgen_client.protocol.schema import Implementation
from kernelgen_server.evaluation.engine import EvaluationEngine
from kernelgen_server.backends.ascend.device import NpuDevice
batch=Path(%r);source=Path(%r);out=Path(%r);tools=Path(%r)
catalog=Catalog(batch/'catalog');op=catalog.load('upsample_bilinear2d')
impl=Implementation(name='upsample-late-'+%r,definition='upsample_bilinear2d',language='triton',entrypoint='main.py::run',sources=[dict(path='main.py',content=source.read_text())])
engine=EvaluationEngine(NpuDevice(perf_mode='walltime',num_trials=3),'npu:0','ascend',oracle_path=op.oracle_path)
result=engine.preflight(catalog.request('upsample_bilinear2d',impl))
(out/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10))
if result['status']!='PASSED':raise SystemExit('preflight failed')
for filename,output in [('check-remaining-five.py','remaining-semantics.json'),('check-remaining-special.py','special.json')]:
 spec=importlib.util.spec_from_file_location('late_gate_'+output,tools/filename)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 if module.check('upsample_bilinear2d',source,out/output):raise SystemExit('semantic gate failed')
''' % (str(BATCH),str(target),str(leaf),str(TOOLS),digest[:12])
        request=dict(command=['{python}','gate.py'],files=[dict(path='gate.py',content=script)],timeout_seconds=1200)
        save(leaf/'gate-request.json',request)
        q.idle()
        response=json.load(q.opener.open(urllib.request.Request(q.server+'/debug/jobs',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=1245))
        save(leaf/'gate-response.json',response);q.idle()
        record=dict(name=name,source=str(target),source_sha256=digest,diagnostic_gain=gain,
            contribution='Codex实现与诊断；KGS独立准入、补充语义及一次正式评测',status='semantic_failed')
        records.append(record)
        if response['status']=='SUCCEEDED':
            assert all(json.loads((leaf/p).read_text())['status']=='passed' for p in ['remaining-semantics.json','special.json'])
            try:q.evaluate('upsample_bilinear2d',target,leaf/'formal',name,1)
            except (RuntimeError,AssertionError) as error:record['error']=str(error)
            path=leaf/'formal'/(name+'-1.result.json')
            if path.exists():
                result=json.loads(path.read_text())
                record.update(status='passed' if result['status']=='PASSED' else 'formal_failed',result=str(path),geo_mean=result.get('geo_mean'))
                if result['status']=='PASSED':
                    timing=[r['latency_ms'] for r in result['per_workload'] if r['phase']=='timing']
                    assert len(timing)==6 and all(v>0 for v in timing)
                    record['latency_ms']=math.exp(sum(math.log(v) for v in timing)/len(timing))
        save(out/'summary.json',dict(scope='本批后续候选独立验收；最终须与父版及CC有效候选按实际延迟共同选版',records=records))
        print(json.dumps(record,ensure_ascii=False),flush=True)
    save(out/'summary.json',dict(status='completed',records=records))


if __name__=='__main__':main()
