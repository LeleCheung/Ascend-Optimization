"""在910B离线核验最终选中源码、合同、数值门禁和真实路径，导出逐case表。"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math

PROJECT=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
BATCH=PROJECT/'operators/campaigns/remaining-five-20261023-v2'
TOOLS=PROJECT/'tools/profiling/hgkda'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def local(value):
 p=Path(value);p.resolve().relative_to(PROJECT.resolve());assert not p.is_symlink();return p


def execute(output=None):
 output=output or BATCH/'final-delivery-audit.json';assert not output.exists()
 output.parent.mkdir(parents=True,exist_ok=True)
 assert read(BATCH/'finalization-state.json')['status']=='completed'
 assert read(BATCH/'packed-v2-state.json')['status']=='completed'
 assert read(BATCH/'packed-v3-state.json')['status']=='completed'
 manifest=read(BATCH/'final-delivery.json')
 records={r['operator']:r for r in read(BATCH/'plan.json')['records']}
 spec=importlib.util.spec_from_file_location('final_formal_contract',TOOLS/'audit-endurance.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 seen=set();proofs=[];timings=[]
 for row in manifest['records']:
  op=row['operator'];assert op not in seen;seen.add(op)
  selected=row['selected'];digest=selected['source_sha256'];record=records[op]
  assert sha(local(row['source']))==sha(local(selected['source']))==digest
  result_path=local(selected['result']);result=read(result_path)
  assert result['status']=='PASSED' and result['num_passed']==result['num_workloads']==record['correctness_cases']+record['timing_cases']
  assert row['score']==result['geo_mean'] and math.isfinite(row['score']) and row['score']>0
  module.verify_formal_contract(record,result_path.parent,result)
  request=read(result_path.with_name(result_path.name.replace('.result.json','.request.json')))
  assert len(request['implementation']['sources'])==1 and request['implementation']['sources'][0]['content']==local(row['source']).read_text()
  expected={'scatter_add':[18,18],'upsample_bilinear2d':[36,48],'fractional_max_pool3d':[27,30],'_conj_physical':[34]}[op]
  gates=[read(local(name)) for name in row['gates']]
  assert sorted(g['total'] for g in gates)==sorted(expected)
  for g in gates:
   assert g['status']=='passed' and g['passed']==g['total']==len(g['cases']) and g['source_sha256']==digest and g['physical_device']==7
   assert all(case['status']=='passed' for case in g['cases'])
   valid_tools=['check-scatter-dynamic-range.py','check-endurance.py'] if op=='scatter_add' else ['check-remaining-special.py','check-remaining-five.py']
   assert g['checker_sha256'] in [sha(TOOLS/name) for name in valid_tools]
  leaf=BATCH/record['catalog_leaf']
  contract={phase:[json.loads(line) for line in (leaf/(phase+'.jsonl')).read_text().splitlines() if line.strip()] for phase in ['correctness','timing']}
  if op=='scatter_add':
   g=next(g for g in gates if 'timing_input_correctness' in g);extra=g['timing_input_correctness']
   assert extra['passed']==extra['total']==len(contract['timing'])
   assert [r['case'] for r in extra['cases']]==[r['name'] for r in contract['timing']]
   assert all(r['status']=='passed' and r['repetitions']==2 and r['stream_switch_checked'] for r in g['cases']+extra['cases'])
  else:
   g=next(g for g in gates if any(r.get('phase')=='timing' for r in g['cases']))
   assert g['contract_sha256']==record['contract_sha256']
   for phase in ['correctness','timing']:
    assert [r['case'] for r in g['cases'] if r.get('phase')==phase]==[r['name'] for r in contract[phase]]
   assert all(r['repetitions']==2 and r['stream_switch'] for r in g['cases'])
  for key in ['facts','complex_facts']:
   if key not in selected:continue
   facts=read(local(selected[key]));assert facts['source_sha256']==digest and facts['physical_device']==7
   for side in facts['sides'].values():
    if side['status']=='completed':assert sha(local(side['trace']))==side['trace_sha256']
   for entry in facts['ir']:
    for artifact in entry['artifacts']:assert sha(local(artifact['path']))==artifact['sha256']
  latency=[r['latency_ms'] for r in result['per_workload'] if r['phase']=='timing']
  assert len(latency)==record['timing_cases'] and all(math.isfinite(v) and v>0 for v in latency)
  actual=math.exp(sum(math.log(v) for v in latency)/len(latency))
  assert math.isclose(actual,selected['latency_ms'],rel_tol=1e-12)
  for r in result['per_workload']:
   if r['phase']=='timing':timings.append(dict(operator=op,case=r['uuid'],candidate_ms=r['latency_ms'],reported_speedup=r.get('speedup',''),source_sha256=digest))
  proofs.append(dict(operator=op,score=row['score'],formal=result['num_workloads'],gates=expected,source_sha256=digest,reference_scope=row['reference_scope']))
 assert seen=={'scatter_add','upsample_bilinear2d','fractional_max_pool3d','_conj_physical'}
 stability=read(local(manifest['cumulative_reference_counterexample']))
 assert stability['physical_device']==7 and stability['contract_sha256']==records['cumulative_trapezoid']['contract_sha256']
 unstable=[r for r in stability['cases'] if r.get('empty_tolerance_intersections',0)>0]
 assert len(unstable)==1 and unstable[0]['spec']['dtype']=='float16' and unstable[0]['spec']['dim']==0
 assert unstable[0]['empty_tolerance_intersections']==380
 csv_path=output.with_suffix('.cases.csv')
 assert not csv_path.exists()
 with csv_path.open('w',encoding='utf-8-sig',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=['operator','case','candidate_ms','reported_speedup','source_sha256']);writer.writeheader();writer.writerows(timings)
 result=dict(status='passed',records=proofs,timing_cases=len(timings),manifest_sha256=sha(BATCH/'final-delivery.json'),
  checker_sha256=sha(Path(__file__)),cumulative_reference_empty_intersections=380,unclosed=manifest['unclosed'],under_target=manifest['under_target'],
  policy='证据一致性通过不表示五项原生NPU性能全部达标；两项CPU回退对照继续单列')
 output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,help='新审计文件；拒绝覆盖历史输出')
 execute(parser.parse_args().output)
