"""离线核验CANN参考、原合同与新增候选；不启动设备任务。"""
import ast
import json
import math
from pathlib import Path
from common import BATCH, read, save, sha


def geometric(values):
    return math.exp(sum(math.log(v) for v in values)/len(values)) if values else None


def main():
    state = read(BATCH/'next-state.json')
    initial = read(BATCH/'starting-acceptance.json')
    refs = read(BATCH/'reference-index.json')
    evidence, issues, pending = {}, [], []

    def track(path, expected=None):
        p = Path(path)
        value = sha(p)
        if expected is not None and value != expected:
            raise AssertionError('证据身份不同 '+str(p))
        evidence[str(p)] = value
        if p.suffix == '.py':
            for node in ast.walk(ast.parse(p.read_text(encoding='utf8'))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith('/data/hanle/ascend-optimization/') and node.value.endswith('.so'):
                    binary = Path(node.value)
                    evidence[str(binary)] = sha(binary)
        return p

    def formal(path, source):
        p = track(path)
        result = read(p)
        req = track(p.with_name(p.name.replace('.result.json','.request.json')))
        payload = read(req)
        track(source)
        assert payload['implementation']['sources'][0]['content'] == Path(source).read_text(encoding='utf8')
        assert payload['settings'] == dict(warmup_ms=1000, benchmark_ms=100, num_trials=3, timeout_seconds=1800)
        ok = result['status']=='PASSED' and result.get('num_passed')==result.get('num_workloads') and not result.get('is_hack')
        cases = [v for v in result.get('per_workload',[]) if v.get('phase')=='timing' and v.get('latency_ms') and v.get('reference_latency_ms')]
        score = geometric([v['reference_latency_ms']/v['latency_ms'] for v in cases])
        if ok:
            assert cases and math.isclose(score,result['geo_mean'],rel_tol=1e-9,abs_tol=1e-12)
        return dict(status=result['status'], complete=[result.get('num_passed'),result.get('num_workloads')],
                    passed=ok, score=score if ok else None, result=str(p),
                    case_details=[dict(uuid=v['uuid'],candidate_us=v['latency_ms']*1000,
                         reference_us=v['reference_latency_ms']*1000,ratio=v['reference_latency_ms']/v['latency_ms']) for v in cases],
                    cases_below_threshold=[v['uuid'] for v in cases if v['reference_latency_ms']/v['latency_ms']<=.8])

    def extra(path, source):
        p = track(path)
        value = read(p)
        assert value['source_sha256']==sha(source)
        if value.get('checker_source'):
            track(value['checker_source']['path'],value['checker_source']['sha256'])
        return dict(status=value.get('status'),passed=value.get('passed'),total=value.get('total'),path=str(p))

    rows = []
    for old in initial['table']['rows']:
        op = old['operator']
        row = dict(order=old['order'],operator=op,previous_regular=old['regular_accepted'],
                   previous_pytorch_score=old.get('score'),cann_status='尚无合格同语义CANN参考')
        entry = state.get('references',{}).get(op)
        if entry:
            source = track(entry['source'],entry['source_sha256'])
            row.update(reference_kind=entry['kind'],reference_source=str(source),reference_source_sha256=sha(source))
            if entry.get('build'):
                buildpath = track(entry['build'],entry['build_sha256'])
                build = read(buildpath)
                track(build['source'],build['source_sha256'])
                track(build['binary'],build['binary_sha256'])
                for name,digest in build.get('headers',{}).items():track(name,digest)
            f = entry.get('formal')
            if f:
                row['reference_validation'] = formal(f['result'],source)
            gates = entry.get('extra_checks')
            if gates and Path(gates).exists():row['reference_supplementary'] = extra(gates,source)
            qualified = row.get('reference_validation',{}).get('passed') and row.get('reference_supplementary',{}).get('status')=='passed'
            row['cann_reference_qualified'] = bool(qualified)
            row['cann_status'] = 'CANN参考已通过资格' if qualified else 'CANN参考待完成或未通过资格'
            comp = entry.get('comparison')
            if comp:
                row['starting_vs_cann'] = formal(comp['result'],comp['source'])
                p = Path(comp['result']).parent/'extra-vs-cann.json'
                if p.exists():row['starting_vs_cann_supplementary'] = extra(p,comp['source'])
            candidates = []
            for proposal in sorted((BATCH/'proposals'/op).glob('*/proposal.json')):
                spec = read(track(proposal))
                if spec.get('disabled_reason'):continue
                code = track(proposal.parent/spec['source'],spec['source_sha256'])
                item = dict(source=str(code),source_sha256=sha(code),backend=spec['backend'],proposal=str(proposal))
                manifest = proposal.parent/'manifest.json'
                if manifest.exists():
                    m = read(track(manifest))
                    for p,digest in m.get('files',{}).items():track(p,digest)
                    item['attribution'] = m.get('attribution')
                p = proposal.parent/'original/HG-KDA-new-1.result.json'
                if p.exists():item['pytorch'] = formal(p,code)
                p = proposal.parent/'extra-checks.json'
                if p.exists():item['supplementary'] = extra(p,code)
                p = proposal.parent/'CANN/HG-KDA-new-vs-CANN-1.result.json'
                if p.exists():item['cann'] = formal(p,code)
                p = proposal.parent/'CANN/extra-checks.json'
                if p.exists():item['cann_supplementary'] = extra(p,code)
                item['qualified'] = bool(qualified and item.get('pytorch',{}).get('passed') and
                    item.get('supplementary',{}).get('status')=='passed' and item.get('cann',{}).get('passed') and
                    item.get('cann_supplementary',{}).get('status')=='passed')
                starting = row.get('starting_vs_cann',{})
                if starting.get('passed') and item.get('cann',{}).get('passed'):
                    prior = {v['uuid']:v for v in starting['case_details']}
                    newer = {v['uuid']:v for v in item['cann']['case_details']}
                    assert set(prior)==set(newer)
                    item['candidate_latency_gain'] = geometric([prior[k]['candidate_us']/newer[k]['candidate_us'] for k in prior])
                    item['reference_latency_change'] = geometric([prior[k]['reference_us']/newer[k]['reference_us'] for k in prior])
                candidates.append(item)
            row['new_candidates'] = candidates
            good = [v for v in candidates if v['qualified']]
            if good:
                best = min(good,key=lambda v:geometric([c['candidate_us'] for c in v['cann']['case_details']]))
                row['selected_new'] = best
                row['selected_vs_cann'] = best['cann']['score']
                row['cann_accepted'] = row['selected_vs_cann']>.8
                row['cann_status'] = '新候选CANN对照达标' if row['cann_accepted'] else '新候选CANN对照未达标'
            else:
                start = row.get('starting_vs_cann',{})
                if start.get('passed'):
                    row['selected_vs_cann'] = start['score']
                    row['cann_accepted'] = bool(qualified and start['score']>.8 and row.get('starting_vs_cann_supplementary',{}).get('status')=='passed')
                    row['cann_status'] = ('既有源码CANN对照达标' if row['cann_accepted'] else
                        '既有源码CANN对照未达标或补测待完成')
        rows.append(row)
    for p in BATCH.rglob('*.request.json'):
        if not p.with_name(p.name.replace('.request.json','.result.json')).exists():pending.append(str(p))
    for name in ['plan.json','bootstrap.json','software-lock.json','reference-index.json','starting-acceptance.json']:
        track(BATCH/name)
    report = dict(schema='hgkda-cann-priority-report-v1',queue_status=state['status'],current_job=state.get('current_job'),
        previous_pytorch_regular=24,total=30,cann_references_qualified=sum(v.get('cann_reference_qualified',False) for v in rows),
        cann_accepted=sum(v.get('cann_accepted',False) for v in rows),
        newly_crossed_cann=[v['operator'] for v in rows if v.get('cann_accepted') and v.get('starting_vs_cann',{}).get('score',math.inf)<=.8 and v.get('selected_new')],
        rows=rows,pending_requests=pending,audit_issues=issues,evidence_sha256=evidence,
        rule='原PyTorch24项保留，CANN单算子和组合分列；本批复测不计新优化，逐case低分保留；>0.8通过，>1才更快')
    save(BATCH/'delivery.json',report)
    labels={'CANN_native_single_api':'原生CANN','CANN_native_uniform_with_generator_adapter':'原生CANN＋随机状态适配',
        'CANN_dtype_compatibility':'CANN＋dtype兼容','CANN_FP32_compatibility_composition':'CANN＋FP32转换',
        'CANN_FP32_state_compatibility_composition':'CANN＋FP32／状态适配',
        'CANN_zeros_scatter_composition':'CANN组合：清零＋Scatter',
        'CANN_Add_Muls_Cumsum_composition':'CANN组合：加法＋缩放＋扫描',
        'CANN_Copy_Neg_storage_composition':'CANN组合：复制＋虚部取负'}
    lines=['# CANN优先30项对照','','固定FlagGems d6a8eec、FlagTree 0.6.0+ascend.gitc286cba6、CANN 9.0.0。',
       '原PyTorch参考常规达标24/30；本批已核验CANN参考%d项、相对CANN达标%d项。'%(report['cann_references_qualified'],report['cann_accepted']),
       '本批使用KGS完整调用计时；CANN兼容组合包含转换与状态适配成本。历史列仅保留前批相同源码的结果。',
       '','|算子|CANN参考|本批相对CANN|本批相对PyTorch|历史相对PyTorch|状态|','|---|---|---|---|---|---|']
    for row in rows:
        old='%.3f×'%row['previous_pytorch_score'] if row['previous_pytorch_score'] is not None else '—'
        score='%.3f×'%row['selected_vs_cann'] if row.get('selected_vs_cann') is not None else '—'
        new=row.get('selected_new',{}).get('pytorch',{}).get('score')
        pytorch='%.3f×'%new if new is not None else '—'
        kind=row.get('reference_kind','未核验')
        lines.append('|'+row['operator']+'|'+labels.get(kind,kind)+'|'+score+'|'+pytorch+'|'+old+'|'+row['cann_status']+'|')
    lines+=['','本批新跨过CANN验收线：'+('、'.join(report['newly_crossed_cann']) or '尚无完整交付')+'。',
        '待返回正式请求%d项；证据核验%d项。'%(len(pending),len(evidence)),
        '原参考含CPU回退、确定性条件及求逆奇异语义继续单列；不将30项写成默认全部达标。']
    (BATCH/'CANN优先30项总表.md').write_text(chr(10).join(lines)+chr(10),encoding='utf8')
    print(json.dumps({k:report[k] for k in ['queue_status','cann_references_qualified','cann_accepted','newly_crossed_cann','pending_requests']},ensure_ascii=False))


if __name__=='__main__':main()
