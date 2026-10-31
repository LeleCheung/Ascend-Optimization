"""冻结最佳源码并独立确认；保留未过线项及失败原件。"""
import hashlib
import json
import math
from pathlib import Path
import shutil


def finalize(r):
    from run import quote,PY
    # 导入的run是原next_phase模块，保持原正式/补测合同。
    for op in r.q.records:
        row=r.state['operators'].get(op,{})
        best=row.get('best') or {}
        if best.get('status')!='PASSED' or not best.get('extra_checks'):
            # 未合格项也交付可定位的起点和失败证据，文件名明确其资格。
            starting=r.starting_source(op)
            out=r.batch/'deliverables'/op;out.mkdir(parents=True,exist_ok=True)
            dest=out/'unqualified-starting-point.py'
            if dest.exists() and dest.read_bytes()!=starting.read_bytes():raise FileExistsError(dest)
            dest.write_bytes(starting.read_bytes())
            failures=[]
            for path in (r.batch/'proposals'/op).glob('*/assessment/formal/*.result.json'):
                value=json.loads(path.read_text(encoding='utf8'))
                failures.append(dict(result=str(path),status=value.get('status'),passed=value.get('num_passed'),total=value.get('num_workloads')))
            record=dict(operator=op,status='尚无通过完整正确性的可交付实现',source=str(starting),
                source_sha256=hashlib.sha256(starting.read_bytes()).hexdigest(),starting_result=row.get('starting_point'),
                attempted_results=failures,diagnostics=[str(p) for p in (r.batch/'analysis'/op).glob('*probe*.json')],
                scope='保留可复现起点与真实失败，不作为达标源码或合格实现使用')
            (out/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            continue
        from validation_status import supplement_status
        gates=Path(best['extra_checks'])
        if supplement_status(op,json.loads(gates.read_text()),best['source_sha256'])!='passed':continue
        digest=best['source_sha256'];source=Path(best['source'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise RuntimeError('最终源码身份不同 '+op)
        result=None
        if best.get('geo_mean',0)>.8:
            result=r.job('final-confirm:'+op+':'+digest[:12],lambda op=op,source=source,digest=digest:
                         r.evaluate(op,source,r.batch/'final-confirmation'/op/digest[:12],'HG-KDA-final-confirm'))
            if not result:raise r.module.Attention('确认结果未就绪 '+op)
            result.update(extra_checks=str(gates),supplementary_status='passed')
            row['final_confirmation']=result
            stable=result.get('status')=='PASSED' and result.get('geo_mean',0)>.8
            row['status']='两次完整评测及补测通过，稳定超过0.8×' if stable else '确认未稳定超过0.8×'
            row['reported_speedup']=math.sqrt(best['geo_mean']*result['geo_mean']) if result.get('status')=='PASSED' else None
            additional=row.get('additional_confirmations',[])
            if additional:
                scores=[best.get('geo_mean'),result.get('geo_mean'),*[item.get('geo_mean') for item in additional]]
                if all(isinstance(v,(int,float)) and v>0 for v in scores):
                    row['reported_speedup']=math.exp(sum(math.log(v) for v in scores)/len(scores))
                stable=stable and all(item.get('status')=='PASSED' and item.get('geo_mean',0)>.8 and item.get('source_sha256')==sha for item in additional)
                row['status']='追加完整评测均过0.8×，补测通过' if stable else '追加确认低于0.8×，不计稳定达标'
        else:
            row['status']='完整及补测通过，性能仍未达标'
            row['reported_speedup']=best.get('geo_mean')
        out=r.batch/'deliverables'/op;out.mkdir(parents=True,exist_ok=True)
        dest=out/'main.py'
        if dest.exists() and dest.read_bytes()!=source.read_bytes():raise FileExistsError(dest)
        dest.write_bytes(source.read_bytes())
        record=dict(operator=op,source=str(source),source_sha256=digest,formal_result=best['result'],
                    supplementary=str(gates),confirmation=result,additional_confirmations=row.get('additional_confirmations',[]),status=row['status'],speedup=row.get('reported_speedup'),
                    backend=best.get('backend','Triton'),original_contract=r.q.records[op]['contract_sha256'])
        (out/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        r.save()
