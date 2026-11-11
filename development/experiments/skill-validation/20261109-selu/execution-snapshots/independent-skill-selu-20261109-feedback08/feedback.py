import json,subprocess,sys,time
from pathlib import Path
b=Path('/data/hanle/ascend-optimization');repo=b/'Ascend-Optimization';r=b/'runs/independent-skill-selu-20261109-v1'
formal=r/'formal-3cec1212c3b7'
for i in range(90):
 if (formal/'audit-final.json').exists() and (formal/'service-stop.json').exists():break
 time.sleep(5)
else:raise RuntimeError('正式评测及两侧补测尚未完成')
d=json.loads((formal/'operators/selu_/evaluation/result.json').read_bytes())
feedback=dict(status=d['status'],cann_score=d['geo_mean'],passed=d['num_passed'],total=d['num_workloads'],
 source_sha256='3cec1212c3b7630bbf548188693f4cd9fef8747ee78cb1f822a42dcf685a23ed',
 retained_delivery_cann_score=1.0392916039566351,
 supplemental=json.loads((formal/'audit-final.json').read_bytes())['rows'][0]['checks'],
 timing=[{k:row.get(k) for k in ['uuid','latency_ms','reference_latency_ms','speedup']} for row in d['per_workload'] if row['phase']=='timing'])
(r/'workspace/formal-feedback.json').write_text(json.dumps(feedback,ensure_ascii=False,indent=2)+'\n')
subprocess.run([sys.executable,str(repo/'tools/skill_validation/agent_runner.py'),'--config',str(r/'config-entry-repair.json'),
 '--model-settings','/root/.claude/settings.json','--invocation','08-formal-result-feedback',
 '--max-turns','10','--prompt-file',str(b/'runs/.queued/independent-skill-selu-20261109-feedback08/prompt.md')],check=True)
