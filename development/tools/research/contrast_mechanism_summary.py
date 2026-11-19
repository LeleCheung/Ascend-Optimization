"""Extract measured mechanism facts; never equate partial simulator windows with latency."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--simulator',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();pair=json.loads(a.comparison.read_bytes());pair=pair.get('result',pair)
    sim=json.loads(a.simulator.read_bytes());cand=pair['candidate']
    fields={};coverage={}
    selected={'aiv_vec_ratio','aiv_vec_misc_ratio','aiv_vec_fp32_ratio','aiv_mte2_ratio','aiv_mte3_ratio',
              'aiv_mte2_instructions','aiv_mte3_instructions','GM_to_UB_datas(KB)','UB_to_GM_datas(KB)'}
    for name,table in cand['tables'].items():
        coverage[name]=table.get('rows')
        for key,stats in table['fields'].items():
            if key in selected:fields[key]=stats
    timelines=[]
    for report in sim['reports']:
        t=report['timeline']
        timelines.append(dict(core=report['core'],window_us=t['capture_duration'],
                              pipelines=[{k:row.get(k) for k in ['pipeline','pipeline_kind','event_count','productive_union','wait_only_union','idle_union']}
                                         for row in t['pipelines'] if row.get('pipeline_kind')=='hardware_execution'],
                              top_instructions=report.get('top_instructions')))
    result=dict(source_sha256=sim['source_sha256'],candidate_basic=cand['basic'],metric_rows=coverage,
                fields=fields,simulator_timelines=timelines,capture_scope=sim['capture_scope'],
                limits=['Different capture windows cannot establish speedup or total waits eliminated.',
                        'Metric sums describe available program rows only; compare coverage before using totals.',
                        'Kernel timing is not a valid optimization result unless complete correctness checks pass.'])
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['simulator_timelines','limits']},ensure_ascii=False))


if __name__=='__main__':main()
