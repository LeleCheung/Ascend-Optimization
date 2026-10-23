"""仅在修复队列尚未开始时，导入真实通过全部门禁和正式评测的候选。"""
from pathlib import Path
import datetime,hashlib,importlib.util,json,math
root=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
batch=root/'operators/campaigns/scatter-segmented-20261023'
main=root/'operators/campaigns/remaining-five-20261023-v2'
tools=root/'tools/profiling/hgkda'
state_path=batch/'endurance-state.json'
assert not state_path.exists(),'只允许初始化尚未启动的修复批次'
assert json.loads((main/'endurance-state.json').read_text())['status']=='running'
assert json.loads((batch/'wait-state.json').read_text())['status']=='waiting_for_main_and_postflight'
digest='72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e'
candidate=batch/'results/scatter_add/codex-manual/candidates'/digest
source=candidate/'main.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
gates=[candidate/'semantics.json',candidate/'dynamic-range.json']
for p in gates:
 d=json.loads(p.read_text());assert d['source_sha256']==digest and d['status']=='passed' and d['passed']==d['total']==18
 assert all(r['status']=='passed' for r in d['cases'])
normal=json.loads(gates[0].read_text())
assert normal['timing_input_correctness']['passed']==normal['timing_input_correctness']['total']==9
assert all(r['status']=='passed' for r in normal['timing_input_correctness']['cases'])
formal=candidate/'formal';result_path=formal/'codex-manual-1.result.json'
result=json.loads(result_path.read_text())
assert result['status']=='PASSED' and result['num_passed']==result['num_workloads']==18
request=json.loads((formal/'codex-manual-1.request.json').read_text())
assert len(request['implementation']['sources'])==1
assert request['implementation']['sources'][0]['content']==source.read_text()
plan=json.loads((batch/'plan.json').read_text());record=plan['records'][0]
spec=importlib.util.spec_from_file_location('scatter_import_audit',tools/'audit-endurance.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
helper.verify_formal_contract(record,formal,result)
rows=[r for r in result['per_workload'] if r['phase']=='timing']
assert len(rows)==9 and all(r['latency_ms']>0 for r in rows)
latency=math.exp(sum(math.log(r['latency_ms']) for r in rows)/len(rows))
best=dict(status='passed',source=str(source),source_sha256=digest,result=str(result_path),
          geo_mean=result['geo_mean'],latency_ms=latency,gates=[str(p) for p in gates],
          contribution='Codex独立诊断、显式分段扫描与真实完整验证；不是CC独立生成')
now=datetime.datetime.utcnow().isoformat()+'Z'
key='validate:scatter_add:codex-manual:'+digest[:12]
state=dict(status='created',operators={'scatter_add':dict(status='queued',best=best)},
           jobs={key:dict(status='completed',result=best,started_at=now,finished_at=now)},
           created_at=now,updated_at=now,
           imported_candidate=dict(policy='真实结果导入，不跳过父版反例核查；启动时才开始总预算',
                                   source_sha256=digest,source_sha256_origin='Codex candidate',
                                   original_formal=str(result_path)))
assert 'started_epoch' not in state
payload=(json.dumps(state,ensure_ascii=False,indent=2)+'\n').encode()
# 排队器此时仍被主批次完成条件和同一卡锁阻挡；独占创建，不覆盖活跃或历史状态。
with state_path.open('xb') as stream:stream.write(payload)
entry=dict(status='passed',imported_job=key,source_sha256=digest,score=result['geo_mean'],
           formal=18,supplementary=18,timing_numerical=9,dynamic_range=18,
           state_sha256=hashlib.sha256(payload).hexdigest(),created_at=now,
           contribution=best['contribution'])
target=batch/'codex-candidate-import.json'
assert not target.exists();target.write_text(json.dumps(entry,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(entry,ensure_ascii=False))
