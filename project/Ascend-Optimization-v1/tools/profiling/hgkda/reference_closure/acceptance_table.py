"""汇总30项原合同与本轮新参考，先核验既有交付源码/请求/补测。"""
import hashlib
import json
import math
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'
sys.path.insert(0,str(HERE.parent/'next_phase'))
from validation_status import supplement_status
FIRST=['selu_','rad2deg_','log10_','cumulative_trapezoid','uniform','value_selecting_reduction_backward','scatter_add','upsample_bilinear2d','fractional_max_pool3d','_conj_physical']
SECOND=['negative_','xlogy_scalar_tensor','nan_to_num_','cross_out','expm1_','hardshrink_backward','subtract_','native_group_norm','log2_','transform_bias_rescale_qkv','fix','round_out','hardswish_out','fake_quantize_per_tensor_affine_cachemask_out','special_round_out','special_xlogy_','hardswish','fmod_tensor_','linalg_inv_ex','batch_norm_with_update_functional']
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    final=read(BATCH.parent/'remaining-final-20261031/delivery.json');previous=read(BATCH.parent/'remaining-nine-20261030/delivery.json')
    current=read(BATCH/'delivery.json');accepted=set(final['accepted']);sources=dict(previous['sources']);tracked={};rows=[]
    def track(p,expected=None):
        p=Path(p);digest=sha(p)
        if expected:assert digest==expected,'源码或证据身份不同 '+str(p)
        tracked[str(p)]=digest
    for r in final['rows']:
        sources[r['operator']]=dict(source=r['source'],source_sha256=r['source_sha256'],score=r['score'],complete=[r['best']['num_passed'],r['best']['num_workloads']],
            backend=r['backend'],supplementary=r['best']['extra_checks'],results=[r['best']['result'],r['confirmation']['result']],method=r['method'])
    assert len(accepted)==24 and len(set(FIRST+SECOND))==30
    for index,op in enumerate(FIRST+SECOND):
        row=dict(order=index+1,operator=op,target_date='1015' if op in FIRST else '1029',regular_accepted=op in accepted)
        if op in accepted:
            item=sources[op];source=Path(item['source']);source=source if source.is_absolute() else PROJECT/source;track(source,item['source_sha256'])
            complete=[]
            for f in item['results']:
                result=Path(f);raw=read(result);request=result.with_name(result.name.replace('.result.json','.request.json'));payload=read(request)
                assert raw['status']=='PASSED' and raw['num_passed']==raw['num_workloads'] and raw['num_workloads']>0 and not raw.get('is_hack')
                assert payload['implementation']['sources'][0]['content']==source.read_text(encoding='utf8')
                track(result);track(request);complete.append([raw['num_passed'],raw['num_workloads']])
            gates=read(item['supplementary']);track(item['supplementary']);assert supplement_status(op,gates,item['source_sha256'])=='passed'
            row.update(status='原参考常规达标',score=item['score'],source=str(source),source_sha256=item['source_sha256'],backend=item['backend'],
                complete_tests=complete,supplementary_tests=[gates['passed'],gates['total']],results=item['results'],method=item.get('method'))
        else:
            names={'linalg_inv_ex':'原参考正确性未通过；奇异数值与info待确认','cumulative_trapezoid':'默认参考重复不自洽，确定性条件单列',
                'fractional_max_pool3d':'原参考含CPU回退，命名NPU分解对照单列','_conj_physical':'原参考含CPU回退，命名NPU分解对照单列',
                'fake_quantize_per_tensor_affine_cachemask_out':'原参考含CPU回退；CANN兼容组合对照单列','batch_norm_with_update_functional':'原参考含CPU回退；CANN兼容组合对照单列'}
            row.update(status=names[op],score=None)
            comp=next((r for r in current['comparisons'] if r['operator']==op),None)
            if comp:row['named_reference_comparison']=comp
            if op in current.get('searches',{}):row['new_search']=current['searches'][op]
        rows.append(row)
    lock=read(BATCH/'software-lock.json')
    table=dict(schema='hgkda-30-reference-qualified-table-v1',regular_accepted=sum(r['regular_accepted'] for r in rows),total=30,
        named_reference_qualified=sum(r.get('named_reference_comparison',{}).get('threshold_passed',False) for r in rows),
        rows=rows,software_lock=lock,evidence_sha256=tracked,queue_status=current['queue_status'],current_job=current.get('current_job'),
        scope='既有原合同交付与本轮命名NPU参考分列；固定原实现的再次对照不是新优化；完整调用walltime')
    (BATCH/'30项验收总表.json').write_text(json.dumps(table,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    lines=['# 30项验收总表','','固定FlagGems：'+lock['flaggems']['commit']+'；FlagTree：'+lock['flagtree']['distribution_version']+'；CANN：9.0.0。',
        '当前24/30项原参考常规达标。其余对象保留CPU回退、默认不稳定和正确性问题；新参考对照独立列示。']
    for label,names in [('1015目标',FIRST),('1029新增目标',SECOND)]:
        lines+=['','## '+label,'','|算子|原参考成绩|本轮命名NPU参考成绩|状态|','|---|---|---|---|']
        for r in rows:
            if r['operator'] not in names:continue
            score='%.3f×'%r['score'] if r['score'] is not None else '—';comp=r.get('named_reference_comparison',{});f=comp.get('formal',{});e=comp.get('supplementary',{})
            named=('%.3f×'%f['score']+' / '+comp['reference']) if isinstance(f.get('score'),(int,float)) else '—'
            if named!='—':named+='；补测'+e.get('status','未完成')
            lines.append('|'+r['operator']+'|'+score+'|'+named+'|'+r['status']+'|')
    lines+=['','常规24项沿用已核验交付，不重跑或挑选更高分。本轮CANN参考和旧候选复测分别记录；0.8×为验收线，超过1.0×才快于对应参考。']
    (BATCH/'30项验收总表.md').write_text(chr(10).join(lines)+chr(10),encoding='utf8')
    print(json.dumps(dict(regular_accepted=table['regular_accepted'],named_reference_qualified=table['named_reference_qualified'],total=30,evidence=len(tracked))))
if __name__=='__main__':main()
