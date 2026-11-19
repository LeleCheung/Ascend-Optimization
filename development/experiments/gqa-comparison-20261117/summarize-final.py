from pathlib import Path
import json,hashlib,csv,math,collections
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117')
read=lambda p:json.loads(p.read_bytes())
base=read(r/'official-check-01.json');target=next(c for c in base['cases'] if c['case_id']==4)['candidate_timing']['full_call_us']
out={'baseline_source_sha256':base['source_sha256'],'target_case':4,'baseline_target_us':target,'arms':{},'scope':'single task, one model trajectory per arm; fixed four-case subset, not full official platform acceptance'}
for arm in ['no-simulator-v2','with-simulator-v2']:
    run=r/arm;rounds=[]
    for invocation in ['round-01','round-02']:
        folder=run/('evaluation-'+invocation);result=read(folder/'results.json')
        assert result['source_sha256']==hashlib.sha256((folder/'source.py').read_bytes()).hexdigest()
        cases=[]
        for c,b in zip(result['cases'],base['cases']):
            assert c['case_id']==b['case_id']
            passed=c.get('candidate_correctness',{}).get('official_checker',{}).get('passed',False)
            us=c.get('candidate_timing',{}).get('full_call_us')
            cases.append({'case_id':c['case_id'],'passed':passed,'candidate_us':us,'native_us':c.get('native_timing',{}).get('full_call_us'),
               'baseline_over_candidate':b['candidate_timing']['full_call_us']/us if us else None,'error':c.get('candidate_error')})
        usage=read(run/'agent-calls'/invocation/'model-usage.json')['result']
        t=next(c for c in cases if c['case_id']==4)
        rounds.append({'round':invocation,'source_sha256':result['source_sha256'],'passed':sum(c['passed'] for c in cases),'total':len(cases),
          'cases':cases,'target_us':t['candidate_us'],'target_speedup':t['baseline_over_candidate'],
          'all_case_candidate_speedup':math.exp(sum(math.log(c['baseline_over_candidate']) for c in cases)/len(cases)) if all(c['passed'] and c['candidate_us'] for c in cases) else None,
          'model':{k:usage.get(k) for k in ['num_turns','duration_ms','usage','total_cost_usd','subtype']}})
    choices=[x for x in rounds if x['passed']==x['total'] and x['target_speedup']>1.01]
    best=min(choices,key=lambda x:x['target_us']) if choices else None
    out['arms'][arm]={'rounds':rounds,'retained':'baseline' if best is None else best['round'],
      'retained_target_us':target if best is None else best['target_us'],'retained_speedup':1 if best is None else best['target_speedup']}
out['device_recapture']={};out['simulator_comparison']={}
for phase,folder in [('before','capture'),('after','capture-after')]:
    basic=next((r/folder/'candidate/raw').glob('OPPROF_*/OpBasicInfo.csv'))
    with basic.open() as f:out['device_recapture'][phase]=list(csv.DictReader(f))[0]
    ir=r/folder/'candidate/actual-ir/_gqa_fwd_kernel.bcmlir'
    out['device_recapture'][phase]['matmul_ir_occurrences']=ir.read_text().count('linalg.matmul')
    listing=read(r/folder/'simulator/listing.json');summary=read(r/folder/'simulator/summary.json');cores=[]
    for core,report in zip(listing['kernels'][0]['cores'],summary['reports']):
        assert core['name']==report['core']
        counts=collections.Counter();times=collections.Counter()
        for i in core['instructions']:
            counts[i['instruction']]+=i['calls'];times[i['instruction']]+=i['running_time_us']
        cores.append({'core':core['name'],'capture_us':report['timeline']['capture_duration'],
         'unit_validation':report['unit_validation']['status'],
         'instruction_calls':{k:counts[k] for k in ['WAIT_FLAG_DEVI','FIX_L0C_TO_DST','MMAD','BAR']},
         'instruction_report_time_us':{k:times[k] for k in ['WAIT_FLAG_DEVI','FIX_L0C_TO_DST','MMAD','BAR']},
         'qualification':report['qualification']})
    out['simulator_comparison'][phase]=cores
out['cautions']=['Simulator applications timed out; counts/times describe observed windows, not full invocation or reconstructed critical path.',
 'One trajectory per arm; no multi-seed statistical claim or assertion that the improvement is impossible without a simulator.',
 'Codex supplied read-only review feedback about a missing residual scaling factor in the simulator arm first candidate; model chose two-term approximation in round 2. This contribution is retained.',
 'The final code comments contain inaccurate BF16 roundoff claims; correctness is established by the fixed official checker subset, not those comments.']
(r/'comparison-results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:{x:v[x] for x in ['retained','retained_target_us','retained_speedup']} for k,v in out['arms'].items()}))
