"""证据约束的知识整理：去重、作用域筛选、验证门禁、有限上下文检索。"""
import argparse
import hashlib
import json
from pathlib import Path
from validation_status import supplement_status


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def organize(records, *, root=Path('.')):
    groups={};rejected=[]
    for record in records:
        row=dict(record)
        required=['operator_type','action_id','scope','summary','evidence']
        if any(not row.get(key) for key in required):
            rejected.append(dict(record=row,reason='缺少结构化作用域或证据'));continue
        if row.get('status') in {'validated','observed_failure'}:
            try:
                expect_failure=row['status']=='observed_failure';failures=0
                ev=row['evidence'];source=root/ev['source'];formal=root/ev['formal_result']
                result=json.loads(formal.read_text(encoding='utf8'))
                assert digest(source)==ev['source_sha256']
                assert digest(formal)==ev['formal_result_sha256']
                assert result['status']=='PASSED' and result['num_passed']==result['num_workloads'] and result['num_workloads']>0
                # 评测请求必须绑定这份源码；不能用另一候选成绩为经验背书。
                request=root/ev['formal_request'];payload=json.loads(request.read_text(encoding='utf8'))
                sources=payload['implementation']['sources']
                assert len(sources)==1 and sources[0]['content']==source.read_text(encoding='utf8')
                for extra in ev.get('supplementary',[]):
                    p=root/extra['path'];assert digest(p)==extra['sha256'];d=json.loads(p.read_text(encoding='utf8'))
                    assert d['source_sha256']==ev['source_sha256']
                    if expect_failure:
                        assert d['status'] in {'passed','failed'}
                        failures+=d['status']=='failed'
                    else:
                        assert d['status']=='passed'
                        operator=row['scope'].get('operator',d.get('operator'))
                        assert supplement_status(operator,d,ev['source_sha256'])=='passed','补测未满足当前算子门禁'
                if expect_failure:assert failures>0,'失败约束没有可核验失败结果'
            except (KeyError,AssertionError,OSError,ValueError) as error:
                row['status']='draft';row['validation_error']=type(error).__name__+': '+str(error)
        elif row.get('status') not in {'draft','rejected','stale'}:row['status']='draft'
        key=json.dumps([row['operator_type'],row['action_id'],row['scope']],sort_keys=True,ensure_ascii=False)
        groups.setdefault(key,[]).append(row)
    active=[];history=[]
    for key,rows in groups.items():
        rows.sort(key=lambda r:(r.get('status') in {'validated','observed_failure'},r.get('updated_at','')),reverse=True)
        chosen=rows[0];chosen['dedup_key']=hashlib.sha256(key.encode()).hexdigest();active.append(chosen)
        for row in rows[1:]:
            row['superseded_by']=chosen.get('id',chosen['dedup_key']);history.append(row)
    return dict(schema='hgkda-knowledge-v1',active=active,superseded=history,rejected=rejected,evidence_root=str(root.resolve()),
                policy='相同类型/动作/作用域去重；源码/请求/结果核验后标validated或observed_failure；失败约束必须有匹配的失败补测；旧原件保留')


def retrieve(index, operator_type, scope, max_chars=6000):
    selected=[];used=0
    checked=organize(index['active'],root=Path(index.get('evidence_root','.')))
    for row in checked['active']:
        if row['status'] not in {'validated','observed_failure'} or row['operator_type']!=operator_type:continue
        if any(scope.get(k)!=v for k,v in row['scope'].items() if v!='*'):continue
        text=json.dumps(row,ensure_ascii=False)
        if used+len(text)>max_chars:continue
        selected.append(row);used+=len(text)
    return dict(records=selected,characters=used,budget=max_chars,policy='仅检索证据核验后的成功实现或失败约束，且作用域兼容；不截断单条证据')


def implementation_examples(index, kind, scope, *, max_chars=3000):
    """跨算子仅推荐可查阅源码，不迁移成绩或将实现事实升级为通用规律。"""
    from validation_status import supplement_status
    checked=organize(index.get('active',[]));selected=[];used=0
    for row in sorted(checked['active'],key=lambda r:r.get('updated_at',''),reverse=True):
        if row.get('status')!='validated' or row['operator_type']!=kind:continue
        if row['scope'].get('operator')==scope.get('operator'):continue
        if any(scope.get(k)!=v for k,v in row['scope'].items() if k!='operator' and v!='*'):continue
        ev=row['evidence'];extras=[json.loads(Path(x['path']).read_text(encoding='utf8')) for x in ev.get('supplementary',[])]
        if not extras or any(supplement_status(row['scope'].get('operator'),e,ev['source_sha256'])!='passed' for e in extras):continue
        entry=dict(operator=row['scope'].get('operator'),source=ev['source'],source_sha256=ev['source_sha256'],
            qualification='来源实现已通过自己的完整合同与补测；新算子仍需独立评测',
            usage='按需查看启动、分块、布局处理；数学公式和可用分块不能直接照搬，性能不继承',
            evidence=ev)
        size=len(json.dumps(entry,ensure_ascii=False))
        if used+size>max_chars:continue
        selected.append(entry);used+=size
        if len(selected)==2:break
    return dict(examples=selected,characters=used,budget=max_chars)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    records=json.loads(a.records.read_text(encoding='utf8'));result=organize(records,root=a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
