"""汇总原合同、新CANN参考及命名NPU分解，禁止混成全30项默认达标。"""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    state=read(BATCH/'next-state.json');initial=read(BATCH/'starting-acceptance.json');evidence={};rows=[]
    for rec in read(BATCH/'plan.json')['records']:
        for name,digest in rec['contract_sha256'].items():
            p=BATCH/rec['catalog_leaf']/name;assert sha(p)==digest;evidence[str(p)]=digest
    for request in BATCH.rglob('*.request.json'):
        payload=read(request)
        if payload.get('api_version')!='v6.2' or not payload.get('implementation'):continue
        result=request.with_name(request.name.replace('.request.json','.result.json'))
        assert result.exists(),'未知正式请求 '+str(request)
        evidence[str(request)]=sha(request);evidence[str(result)]=sha(result)
    refs=read(BATCH/'reference-index.json')
    for op,ref in refs.items():
        assert sha(ref['source'])==ref['source_sha256'];assert sha(ref['binary'])==ref['binary_sha256']
        qpath=BATCH/'references'/op/'qualification.json';q=read(qpath) if qpath.exists() else {}
        formal=state['jobs'].get('cann-formal:'+op,{}).get('result') or {}
        extra=state['jobs'].get('cann-extra:'+op,{}).get('result')
        comp=state['jobs'].get('candidate-vs-cann:'+op,{}).get('result') or {}
        rows.append(dict(operator=op,kind=ref['kind'],qualification=q.get('status','未完成'),qualified_cases=[q.get('passed'),q.get('total')],
            cann_vs_original=formal,cann_supplementary=extra,candidate_vs_cann=comp,scope=ref['scope']))
    new_regular=[]
    for op in ['linalg_inv_ex','cumulative_trapezoid']:
        value=state['operators'].get(op,{}).get('best') or {}
        if value.get('status')=='PASSED' and value.get('supplementary_status')=='passed' and value.get('geo_mean',0)>.8:
            source=Path(value['source']);assert sha(source)==value['source_sha256'];new_regular.append(op)
    supplemental={}
    for op in ['fractional_max_pool3d','_conj_physical']:
        supplemental[op]=state['jobs'].get('supplemental-full:'+op,{}).get('result')
    accepted=list(dict.fromkeys(initial['accepted']+new_regular))
    data=dict(status='completed_with_explicit_reference_status',initial_regular=24,regular_accepted=len(accepted),total=30,new_regular=new_regular,accepted=accepted,
      references=rows,supplemental=supplemental,evidence_sha256=evidence,
      rule='CANN/自建NPU参考成绩不替代原PyTorch合同；资格失败或参考语义冲突不能计为无条件达标',
      attribution='Codex实现CANN参考封装与控制器；KG/CC/DeepSeek追加搜索以真实请求及源码记录为准')
    save(BATCH/'delivery.json',data)
    lines=['# 夜间六项与CANN参考交付','','原合同常规达标：'+str(len(accepted))+'/30；本轮新增：'+('、'.join(new_regular) or '暂无原合同新增达标')+'。补充参考成绩独立列示。','',
      '|算子|CANN资格|通过数|CANN相对原PyTorch|现有候选相对CANN|','|---|---|---|---|---|']
    def score(v):return '%.3f×'%v['geo_mean'] if v and v.get('status')=='PASSED' and isinstance(v.get('geo_mean'),(int,float)) else (v or {}).get('status','未完成')
    for r in rows:lines.append('|'+r['operator']+'|'+r['qualification']+'|'+str(r['qualified_cases'])+'|'+score(r['cann_vs_original'])+'|'+score(r['candidate_vs_cann'])+'|')
    lines+=['','## 其余对象','']
    for op,v in supplemental.items():lines.append('- '+op+'：命名NPU分解对照 '+score(v)+'，保留原PyTorch CPU回退说明。')
    for op in ['linalg_inv_ex','cumulative_trapezoid']:
        job=state['jobs'].get('search:'+op+':1',{});lines.append('- '+op+'：'+job.get('status','未完成')+'；'+score(job.get('result'))+'。详细失败与补测保留。')
    lines+=['','CANN直接调用、BF16兼容组合、自写Ascend C和KG生成Triton分别记账；不按新参考成绩重写历史默认验收。']
    (BATCH/'交付报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(regular_accepted=len(accepted),new_regular=new_regular,reference_rows=len(rows),evidence=len(evidence))))
if __name__=='__main__':main()
