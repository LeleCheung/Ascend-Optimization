from pathlib import Path
import json,hashlib
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117');src=r/'capture/simulator/summary.json'
d=json.loads(src.read_bytes());out={'source_sha256':d['source_sha256'],'case':d['case'],'kernel':d['kernel'],'scope':d['scope'],
 'qualification':'Application timeout after 15 minutes; no completion manifest. Observed windows only, not full-kernel total or causal proof. Non-scalar/non-sync PC timings agree with CSV in microseconds.',
 'original_summary_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'cores':[]}
for c in d['reports']:
    t=c['timeline'];units=c['unit_validation'];assert units['status']=='verified'
    pipes=[{k:p[k] for k in ['pipeline','pipeline_kind','productive_union','wait_only_union','idle_union','productive_ratio','wait_only_ratio','idle_ratio']} for p in t['pipelines'] if p['pipeline_kind']!='auxiliary_trace_lane']
    regions=[{k:x.get(k) for k in ['kind','start','end','instructions','largest_instruction_us','trace_interval_union_us']} for x in c['regions'][:3]]
    out['cores'].append({'core':c['core'],'capture_duration_us':t['capture_duration'],'time_unit_validation':{k:units[k] for k in ['status','unit','compared_pcs','matching_pcs','median_ratio']},'pipelines':pipes,
        'top_instructions':[{k:i.get(k) for k in ['pc','instruction','pipeline','calls','running_time_us','source']} for i in c['top_instructions'][:8]],'regions':regions})
p=r/'with-simulator-v2/workspace/simulator-summary.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2))
manifest=r/'contrast-manifest-v2.json';m=json.loads(manifest.read_bytes());m['simulator_scope']='partial timeout capture, observed window only';m['simulator_summary_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(r/'simulator-input-manifest.json').write_text(json.dumps(m,indent=2))
print(len(p.read_bytes()))
