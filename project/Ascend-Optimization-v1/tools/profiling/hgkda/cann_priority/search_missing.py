"""根据实测CANN差距继续KG搜索，最终仍验证原PyTorch语义。"""
import math
import shutil
from pathlib import Path
from common import BATCH, PROJECT, NEXT, load, read, save, sha


def prepare(r, op, entry):
    """独立子批次只包含一个已核验CANN合同，避免改动原合同。"""
    child = BATCH/'searches'/(BATCH.name+'-'+op)
    reference = Path(entry['source'])
    leaf = r.alternate(op, reference)
    seed = r.starting_source(op)
    record = dict(r.q.records[op])
    if not (child/'plan.json').exists():
        dest = child/'catalog/ops'/op
        dest.mkdir(parents=True, exist_ok=True)
        for name in ['definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl']:
            (dest/name).write_bytes((leaf/name).read_bytes())
        frozen = child/'seeds'/('seed-'+op+'.py')
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(seed.read_bytes())
        record.update(seed=frozen.relative_to(child).as_posix(), source_sha256=sha(frozen),
             catalog_leaf='catalog/ops/'+op, contract_sha256={n:sha(dest/n) for n in ['definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl']})
        save(child/'plan.json', dict(r.q.plan, records=[record],
             passes=[dict(rounds=3,budget_seconds=2400)], duration_hours=1,
             measurement_policy='固定本批CANN参考，3轮2400秒；最后按实际延迟保留完整候选，另做原PyTorch语义与CANN独立确认'))
        save(child/'catalog/manifest.json', dict(api_version='v6.2', name='cann-search-'+op,
             evaluator='native', layout='per-operator'))
        save(child/'targets.json',dict(targets=[dict(operator=op,status='相对本批CANN参考未达标，继续优化')]))
        save(child/'bootstrap.json', {op:dict(path=frozen.relative_to(child).as_posix(),sha256=sha(frozen),origin=str(seed))})
        save(child/'search-provenance.json', dict(parent_batch=str(BATCH),operator=op,
             source=str(seed),source_sha256=sha(seed),reference=str(reference),reference_sha256=sha(reference),
             original_contract_sha256=r.q.records[op]['contract_sha256'],
             cann_contract_sha256=record['contract_sha256'],
             attribution='Codex按当前实测差距组织上下文；KG/HG-KDA＋CC/DeepSeek实际搜索；KGS评测，最终再验原PyTorch语义'))
        shutil.copyfile(BATCH/'software-lock.json',child/'software-lock.json')
        raw = read(entry['comparison']['result'])
        cases = [{k:v.get(k) for k in ['uuid','phase','status','latency_ms','reference_latency_ms','speedup']}
                 for v in raw.get('per_workload',[]) if v.get('phase')=='timing']
        context = dict(schema='hgkda-cann-search-context-v1',status='measured_CANN_gap',
             spec=dict(operator=op,source_sha256=sha(seed),reference_kind=entry['kind']),
             source=dict(path=str(seed),sha256=sha(seed),text=seed.read_text(encoding='utf8'),full_characters=len(seed.read_text(encoding='utf8'))),
             execution_guidance=dict(reference='直接CANN或明确命名的兼容组合；不把PyTorch历史分数当成本批CANN分数',
                 current_geo_mean=raw.get('geo_mean'),cases=cases,
                 instructions='根据实际低分输入检查分块、连续性、重复搬运与host分派；先用get_profile_context获得冻结输入，必要时采集真机metrics/IR。每次改法写明预测与验证。禁止调用CANN目标API代替候选，也禁止调用PyTorch目标算子。原地/out/布局/特殊值均保留；不按UUID或预知数据缓存输出。',
                 evidence=dict(path=entry['comparison']['result'],sha256=sha(entry['comparison']['result']))))
        save(child/'analysis'/op/'agent-context.json',context)
    return child


def process(r, maximum=3):
    # 按本批真实差距立项；已新达标的hardswish_out不会再次搜索。
    choices=[]
    for op,entry in r.state.get('references',{}).items():
        comp=entry.get('comparison') or {}
        if (entry.get('supplementary') or {}).get('status')!='passed' or comp.get('status')!='PASSED':continue
        if not isinstance(comp.get('geo_mean'),(int,float)) or comp['geo_mean']>.8:continue
        valid_new=False
        for proposal in (BATCH/'proposals'/op).glob('*/proposal.json'):
            p=proposal.parent/'CANN/HG-KDA-new-vs-CANN-1.result.json'
            gates=proposal.parent/'CANN/extra-checks.json'
            if p.exists() and gates.exists():
                v=read(p);g=read(gates)
                spec=read(proposal)
                native=proposal.parent/'native-checks.json'
                requires_native=op in {'hardswish','hardswish_out','rad2deg_'} and spec['backend'].startswith('AscendC')
                native_ok=not requires_native or (native.exists() and read(native).get('status')=='passed')
                if v.get('status')=='PASSED' and v.get('geo_mean',0)>.8 and g.get('status')=='passed' and native_ok:valid_new=True
        if not valid_new:choices.append((comp['geo_mean'],op,entry))
    choices.sort()
    selection=BATCH/'search-selection.json'
    selected=[dict(operator=op,score=score,reference_sha256=entry['source_sha256']) for score,op,entry in choices[:maximum]]
    if selection.exists():
        # 恢复时固定已选择对象与预算，不因其他测量的分数改变搜索预算。
        selected=read(selection)['operators']
    else:
        save(selection,dict(maximum_operators=maximum,rounds=3,budget_seconds_per_operator=2400,
             selection='本批CANN合格且当前候选低于0.8，已有新达标者排除，按实际差距排序',operators=selected))
    Runner=load('cann_priority_search_runner',NEXT/'run.py').Runner
    for item in selected:
        op=item['operator'];entry=r.state['references'][op]
        assert entry['source_sha256']==item['reference_sha256']
        child=prepare(r,op,entry)
        worker=Runner(child.relative_to(BATCH.parent).as_posix())
        worker.state['operators'].setdefault(op,{})
        worker.save()
        result=r.job('KG-CANN-search:'+op,lambda op=op,worker=worker:worker.search(op,1))
        if not result or result.get('status')!='PASSED' or result.get('supplementary_status')!='passed':continue
        source=Path(result['source']);target=BATCH/'proposals'/op/'kg-cann-v1'
        target.mkdir(parents=True,exist_ok=True)
        code=target/'main.py'
        if code.exists():assert code.read_bytes()==source.read_bytes()
        else:code.write_bytes(source.read_bytes())
        save(target/'proposal.json',dict(operator=op,id=target.name,source='main.py',source_sha256=sha(code),
             backend=result.get('backend') or 'KG_HG-KDA_Triton',child_result=result,
             attribution='KG/HG-KDA＋CC/DeepSeek生成，Codex组织CANN对照证据；最终另验原PyTorch与CANN',
             prediction='模型根据本批CANN差距和实际profiling提出优化，以原合同和补测确认'))
        worker.state.update(status='search_completed',current_job=None)
        worker.save()
    return dict(status='completed',selected=[v['operator'] for v in selected])
