from pathlib import Path
import json,sys,hashlib
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117');arm=sys.argv[1];run=r/arm
assert arm in ['no-simulator-v2','with-simulator-v2']
current=json.loads((run/'evaluation-round-01/results.json').read_bytes());base=json.loads((r/'official-check-01.json').read_bytes())
cases=[]
for c,b in zip(current['cases'],base['cases']):
    assert c['case_id']==b['case_id']
    timing=c.get('candidate_timing',{}).get('full_call_us');old=b['candidate_timing']['full_call_us']
    cases.append({'case_id':c['case_id'],'baseline_candidate_us':old,'candidate_us':timing,'baseline_over_candidate':old/timing if timing else None,
        'official_checker_passed':c.get('candidate_correctness',{}).get('official_checker',{}).get('passed'),
        'error':c.get('candidate_error')})
target=next(c for c in cases if c['case_id']==4)
feedback={'source_sha256':current['source_sha256'],'baseline_source_sha256':base['source_sha256'],'cases':cases,
 'accepted':all(c['official_checker_passed'] for c in cases),'target_case':4,
 'decision':'target_met' if all(c['official_checker_passed'] for c in cases) and target['baseline_over_candidate']>1.01 else 'continue',
 'scope':'one complete four-case evaluation per source; <1% target change is inconclusive rather than demonstrated gain',
 'original_result_sha256':hashlib.sha256((run/'evaluation-round-01/results.json').read_bytes()).hexdigest()}
review=run/'source-reviews'/(current['source_sha256']+'.json')
if review.exists():feedback['source_review']=json.loads(review.read_bytes())
(run/'workspace/next-performance-feedback.json').write_text(json.dumps(feedback,indent=2))
prompt=(r/'comparison-prompt-v2.md').read_text()
prompt='先读取next-performance-feedback.json，这是你上一轮源码的完整四项结果。根据实际结果选择保留或修正一个机制。若目标已有明确收益且全部通过，可保留；否则提出下一项改动。原始profiling/IR/模拟器属于起点源码，引用时使用role=background，不作为本轮新源码的实测。当前评测反馈引用category=evaluation。请在最多6次读取后登记decision并Edit。\n'+prompt
p=r/'comparison-prompt-round02.md'
if p.exists():assert p.read_text()==prompt
else:p.write_text(prompt)
print(json.dumps(feedback))
