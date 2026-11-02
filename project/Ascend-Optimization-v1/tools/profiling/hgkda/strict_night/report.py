"""离线合并冻结交付与本轮已核验候选，保留完整测量身份，不触发设备评测。"""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
BATCH=PROJECT/'operators/campaigns/strict-night-20261102'
REMOTE='/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def qualified(best):
    return bool(best.get('qualification')=='passed' and best.get('compiled_status')=='verified'
                and best.get('review_status')=='reviewed_strict_triton')


def build(previous,state,materialize):
    """只让已完成全部门禁的新候选替换冻结源码；待核验分数不改变交付数。"""
    rows=[]
    for frozen in previous['rows']:
        row=copy.deepcopy(frozen);op=row['operator']
        best=(state.get('operators',{}).get(op) or {}).get('best') or {}
        row['selection']='reused_frozen_delivery'
        row['starting_cann_score']=(frozen.get('cann') or {}).get('score')
        if best and qualified(best):
            row=materialize(row,best)
            row['selection']='new_verified_candidate'
            row['accepted']=bool(best.get('reference')=='qualified_CANN' and best['search_score']>.8)
            row['status']='严格Triton达标' if row['accepted'] else (
                '严格Triton正确，性能未过0.8×' if best.get('reference')=='qualified_CANN'
                else '原语义通过，CANN参考不合格或缺失，单列')
        elif best:
            row['pending_candidate']={k:best.get(k) for k in ['source','source_sha256','search_score',
                'qualification','review_status','compiled_status']}
        rows.append(row)
    assert len(rows)==30 and len({r['operator'] for r in rows})==30
    groups={g:dict(total=sum(r['group']==g for r in rows),accepted=sum(r['group']==g and r['accepted'] for r in rows))
            for g in ['1015','1029']}
    return dict(schema='hgkda-strict-night-delivery-v1',queue_status=state['status'],current_job=state.get('current_job'),
        accepted=sum(r['accepted'] for r in rows),total=30,groups=groups,rows=rows,
        policy='完整调用相对合格CANN超过0.8×；原语义/补测、人工审查、实际Triton编译与设备路径均通过；冻结版本复用不重测')


def main():
    sys.path.insert(0,str(HERE))
    from measurements import score
    from validation_status import supplement_status
    evidence={}
    def path(name):
        raw=str(name)
        return PROJECT/Path(raw[len(REMOTE)+1:]) if raw.startswith(REMOTE+'/') else Path(raw)
    def track(name):
        p=path(name);evidence[str(name)]=sha(p);return p
    def capture(file,out):
        original=path(file);target=out/original.name
        raw=original.read_bytes()
        if target.exists():assert target.read_bytes()==raw,'已保存原始证据不一致'
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        track(target)
        return target
    def materialize(row,best):
        source=track(best['source']);digest=sha(source);assert digest==best['source_sha256']
        if best.get('out_api_probe'):
            import importlib
            import out_api_probe
            importlib.reload(out_api_probe)
            out_api=read(track(best['out_api_probe']))
            assert out_api_probe.qualified(out_api,digest),'out接口未完成补测'
            assert sha(track(best['out_api_probe_checker']))==out_api['checker_sha256']
        manual=read(track(source.with_name('review.json')))
        assert manual['source_sha256']==digest and manual['status']=='reviewed_strict_triton'
        assert manual['all_branches_reviewed'] and manual['dependencies_reviewed']
        proof=read(track(best['backend_verification']))
        assert proof['status']=='verified' and proof['source_sha256']==digest
        if manual.get('reviewed_host_bridges'):
            host=proof['host_glue']
            assert host['status']=='reviewed_host_only' and host['source_sha256']==digest
            for file,identity in host['files'].items():
                assert sha(track(file))==identity['sha256']
        for file,key in [('review','review_sha256'),('runtime','runtime_sha256'),('trace','trace_sha256')]:
            assert sha(track(proof[file]))==proof[key]
        compiled=read(track(proof['compiled']));assert compiled['source_sha256']==digest
        manifest=read(track(compiled['manifest']))
        assert sha(path(compiled['manifest']))==compiled['manifest_sha256']
        assert manifest['spec']['source_sha256']==digest and manifest['entries']
        for kernel in manifest['entries']:
            kinds=set()
            for artifact in kernel['artifacts']:
                assert sha(track(artifact['path']))==artifact['sha256'];kinds.add(artifact['kind'])
            assert {'ttir','npubin'}<=kinds
        assert proof['tasks'],'设备任务为空'
        runtime=read(path(proof['runtime']));assert runtime['status']=='passed' and runtime['source_sha256']==digest
        formal=path(best['search_result']);v=read(formal);score(v)
        snapshots=source.parent/'formal-reference'
        result=capture(formal,snapshots)
        if best.get('kind')=='Codex_proposal':
            request=formal.with_name(formal.name.replace('.result.json','.request.json'))
            payload=read(capture(request,snapshots))
            assert payload['implementation']['sources'][0]['content']==source.read_text(encoding='utf8')
            observed=score(v)
        else:
            solution=read(capture(formal.with_name('solution.json'),snapshots))
            assert len(solution['sources'])==1 and solution['sources'][0]['content']==source.read_text(encoding='utf8')
            capture(formal.with_name('identity.json'),snapshots)
            workspace=next(p for p in formal.parents if (p/'.kernelgen/run-process.json').exists())
            assert read(workspace/'.kernelgen/run-process.json').get('finished_at'),'搜索尚未结束'
            contract=read(capture(workspace/'stages/optimize/work/.kernelgen/evaluation-contract.json',snapshots))
            assert contract['definition']['reference']==(BATCH/'catalog/ops'/row['operator']/'oracle.py').read_text(encoding='utf8')
            ledger=read(track(best['measurement_ledger']))
            assert ledger['status']=='passed' and ledger['source_sha256']==digest and ledger['meaningful_change']
            for i,entry in enumerate(ledger['observations']):
                original=path(entry['path']);assert sha(original)==entry['sha256']
                copied=capture(original,snapshots/('measurement-'+str(i)))
                data=read(copied);observed=score(data.get('result') or data)
                assert math.isclose(observed,entry['score'],rel_tol=1e-9)
            observed=math.exp(sum(math.log(e['score']) for e in ledger['observations'])/len(ledger['observations']))
        assert math.isclose(observed,best['search_score'],rel_tol=1e-9)
        timing_leaf=BATCH/'catalog/ops'/row['operator']
        expected={(json.loads(line)['name'],phase) for name,phase in [('correctness.jsonl','correctness'),('timing.jsonl','timing')]
                  for line in (timing_leaf/name).read_text(encoding='utf8').splitlines() if line.strip()}
        assert {(x['uuid'],x['phase']) for x in v['per_workload']}==expected
        pt_path=result if path(best['pytorch_result'])==formal else track(best['pytorch_result'])
        pt=read(pt_path);pt_score=score(pt)
        if path(best['pytorch_result'])!=formal:
            pt_request=read(track(Path(best['pytorch_result']).with_name(Path(best['pytorch_result']).name.replace('.result.json','.request.json'))))
            assert pt_request['implementation']['sources'][0]['content']==source.read_text(encoding='utf8')
        extras={}
        for side,item in best['extras'].items():
            value=read(track(item['path']))
            assert value['source_sha256']==digest and supplement_status(row['operator'],value,digest)=='passed'
            extras[side]=dict(status='passed',path=item['path'],source_sha256=digest,passed=value.get('passed'),total=value.get('total'))
        row.update(source=best['source'],source_sha256=digest,review=manual,strict_backend=True,
            runtime=dict(status='passed',path=proof['runtime'],passed=runtime.get('passed'),total=runtime.get('total')),
            compiled_path=dict(status='verified',path=proof['compiled'],source_sha256=digest),
            device_path=dict(status='captured',trace=proof['trace'],trace_sha256=proof['trace_sha256']),
            pytorch=dict(status='PASSED',score=pt_score,path=str(pt_path),passed=pt['num_passed'],total=pt['num_workloads']),
            supplementary=extras['pytorch'],cann_supplementary=extras.get('cann',{}),provenance=best['provenance'])
        if best['reference']=='qualified_CANN':
            row['cann']=dict(status='PASSED',score=observed,path=str(result),passed=v['num_passed'],total=v['num_workloads'],
                measurement_ledger=best.get('measurement_ledger'),measurement='existing_valid_measurements_geometric_mean')
        else:row['cann']={}
        return row
    # 验证冻结交付身份；冻结的详细门禁与请求核验仍由原审计提供。
    previous=read(track(BATCH/'starting-delivery.json'))
    plan=read(track(BATCH/'plan.json'))
    frozen=BATCH.parent/'strict-triton-20261102/delivery.json'
    assert sha(track(frozen))==plan['starting_delivery_sha256']==sha(BATCH/'starting-delivery.json')
    audit=read(track(frozen.with_name('delivery-audit.json')))
    assert audit['status']=='verified' and audit['delivery_sha256']==sha(frozen)
    for row in previous['rows']:assert sha(track(row['source']))==row['source_sha256']
    state=read(BATCH/'state.json')
    result=build(previous,state,materialize)
    result['evidence_sha256']=evidence
    save(BATCH/'delivery.json',result)
    lines=['# 严格Triton夜间优化30项总表','',
        f'冻结起点16/30；本轮已核验{result["accepted"]}/30。队列：{state["status"]}。',
        '加速比为KGS完整调用；CANN参考类型沿用冻结审计，PyTorch另列。待核验候选不计数。','']
    for group in ['1015','1029']:
        g=result['groups'][group]
        lines += [f'## {group}（{g["accepted"]}/{g["total"]}）','',
            '|算子|起点相对CANN|HG-KDA相对CANN|相对PyTorch|原合同|状态|',
            '|---|---:|---:|---:|---|---|']
        for row in result['rows']:
            if row['group']!=group:continue
            fmt=lambda x:f'{x:.3f}×' if x is not None else '—'
            pt=row['pytorch'];relative=Path(row['source']).as_posix()[len(REMOTE)+1:]
            link=Path(__import__('os').path.relpath(PROJECT/relative,BATCH)).as_posix()
            lines.append('|['+row['operator']+']('+link+')|'+fmt(row['starting_cann_score'])+'|'+
                fmt(row['cann'].get('score'))+'|'+fmt(pt.get('score'))+'|'+
                str(pt.get('passed','—'))+'/'+str(pt.get('total','—'))+'|'+row['status']+'|')
        lines.append('')
    lines += ['参考不合格或缺失的算子单列；`fractional_max_pool3d`、`_conj_physical`和`linalg_inv_ex`的PyTorch参考包含CPU回退。',
        '完整源码身份、原始测量、补测、归属、参考类型和后端证明见`delivery.json`；本脚本不触发任何设备评测。','']
    (BATCH/'严格Triton30项总表.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(status='snapshot_verified',accepted=result['accepted'],groups=result['groups']),ensure_ascii=False))


if __name__=='__main__':
    sys.path.insert(0,str(HERE.parent/'next_phase'))
    main()
