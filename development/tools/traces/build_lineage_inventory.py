"""汇总FlagGems原始送测起点、历史版本和当前交付；不推测缺失继承关系。"""
import argparse
import ast
import collections
import hashlib
import html
import json
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


EARLY = {'selu_', 'rad2deg_', 'log10_', 'uniform', 'value_selecting_reduction_backward',
         'scatter_add', 'upsample_bilinear2d', '_conj_physical'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path.cwd(); evidence=args.evidence.resolve(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    discovered=read(evidence/'inventory.json')
    selected=[x for x in read(root/'operators/index.json')['rows'] if x['scope']=='main_30' and x['accepted']]
    matrix={r['operator']:r for r in read(root/'development/reports/trace-attribution-matrix-20261116.json')['rows']}
    rows=[]

    def local_link(path,label):
        return '['+label+']('+Path(os.path.relpath(path,out)).as_posix()+')'

    def verify_export(record):
        if record.get('export'):
            assert sha((evidence/record['export']).read_bytes())==record['sha256'],record

    def baseline_identity(record, upstream):
        if not record.get('sources'):return dict(status='missing_submitted_source')
        source=record['sources'][0];verify_export(source)
        actual=ast.parse((evidence/source['export']).read_text(encoding='utf-8-sig')).body
        original=ast.parse(upstream.read_text(encoding='utf-8-sig')).body
        unchanged=len(actual)>=len(original) and all(ast.dump(a)==ast.dump(b) for a,b in zip(original,actual))
        return dict(status='original_module_plus_adapter' if unchanged else 'requires_review',
                    original_module_ast_unchanged=unchanged,
                    appended_statements=[ast.unparse(x) for x in actual[len(original):]] if unchanged else [],
                    scope='模块主体静态一致；不等于已核验所有动态导入和后端派发路径')

    for selection in selected:
        op=selection['operator']
        records=[r for r in discovered['records'] if r['operator']==op]
        campaign='night-20261021-v3' if op in EARLY else 'next25-20261029-v2'
        plan=next(x for x in discovered['plans'] if x['operator']==op and x['campaign']==campaign)
        upstream=root/'development/dependencies'/plan['record']['upstream_source'].replace('\\','/')
        assert sha(upstream.read_bytes())==plan['record']['upstream_source_sha256']
        baseline_records=[r for r in records if r['baseline_path']]
        canonical=[r for r in baseline_records if '/'+campaign+'/' in r['result']['path'] and
                   ('/FlagGems/' in r['result']['path'] if op in EARLY else ('/baseline/' if op=='negative_' else '/baseline-fixed/') in r['result']['path'])]
        assert len(canonical)==1,(op,len(canonical))
        baseline=canonical[0]
        identity=baseline_identity(baseline,upstream)
        assert identity['original_module_ast_unchanged'],op
        for r in records:
            verify_export(r['result'])
            for s in r.get('sources',[]):verify_export(s)
            for s in r.get('provenance',[]):verify_export(s)
            if r.get('request'):verify_export(r['request'])
        baseline_raw=read(evidence/baseline['result']['export'])
        baseline['result_log_excerpt']=str(baseline_raw.get('log',''))[-2200:]
        baseline['identity']=identity
        for r in baseline_records:
            r['identity']=baseline_identity(r,upstream)
        manifest=read(root/'operators'/op/'manifest.json')
        delivery=dict(manifest['selected'])
        assert sha((root/delivery['path']).read_bytes())==delivery['sha256']
        result_path=root/'operators'/op/'evidence/retained/evaluation/result.json'
        result=read(result_path)
        assert result['status']=='PASSED' and result['num_passed']==result['num_workloads']
        assert abs(result['geo_mean']-delivery['cann_score'])<1e-12
        delivery.update(result=result_path.relative_to(root).as_posix(),result_sha256=sha(result_path.read_bytes()),
                        passed=result['num_passed'],total=result['num_workloads'])
        matches=[r for r in records if any(s['sha256']==delivery['sha256'] for s in r.get('sources',[]))]
        assert matches,op
        exact=[r for r in matches if r['measurement']['geo_mean']==delivery['cann_score']]
        delivery['remote_exact_measurements']=[r['result'] for r in exact]
        delivery['remote_same_source_measurements']=[r['result'] for r in matches]
        noncompliant=manifest['versions'].get('historical_noncompliant',{}).get('source',{}).get('sha256')
        for r in records:
            names=[s['sha256'] for s in r.get('sources',[])]
            path=r['result']['path'];impl=r.get('implementation',{}).get('name') or ''
            r['role']='baseline' if r['baseline_path'] else ('reference' if '/references/' in path or '/references-v' in path or impl.startswith('CANN-') else 'candidate')
            r['constraint_status']='known_noncompliant' if noncompliant and noncompliant in names else ('selected_strict_delivery' if delivery['sha256'] in names else 'historical_not_reaudited')
        candidate_records=[r for r in records if r['role']=='candidate']
        historical=matrix[op]['historical_rounds']
        source_refs=collections.defaultdict(list)
        for r in records:
            for s in r.get('sources',[]):source_refs[s['sha256']].append(s['export'])
        ledger_entries=[]
        for h in historical:
            item=dict(h)
            item['recovered_sources']=sorted(set(source_refs.get(h.get('source_sha256'),[])))
            ledger_entries.append(item)
        # Reference bundle IDs are compared as recorded; relocation can also change IDs.
        base_bundle=baseline.get('contract',{}).get('binding') or {}
        same_bundle=[r for r in exact if (r.get('contract',{}).get('binding') or {})==base_bundle]
        gap=[]
        if baseline['measurement']['status']!='PASSED':gap.append('原生起点未完整通过；保留失败节点，不填0分')
        if not same_bundle:gap.append('原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测')
        gap.append('源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对')
        row=dict(operator=op,upstream=dict(commit='d6a8eec473517a3d68157b208eb9c057eb1d4c50',
                    path=upstream.relative_to(root).as_posix(),sha256=sha(upstream.read_bytes()),plan=plan),
                 baseline=baseline,baseline_alternatives=baseline_records,delivery=delivery,
                 historical_ledger_entries=ledger_entries,measurements=records,
                 counts=dict(measurements=len(records),candidate_measurements=len(candidate_records),
                     candidate_sources=len(set(s['sha256'] for r in candidate_records for s in r.get('sources',[]))),
                     ledger_points=len(historical),baseline_records=len(baseline_records)),gaps=gap)
        rows.append(row)

    summary=dict(operators=len(rows),original_sources_verified=len(rows),baseline_records=sum(r['counts']['baseline_records'] for r in rows),
        canonical_baseline_passed=sum(r['baseline']['measurement']['status']=='PASSED' for r in rows),
        delivery_source_and_result_verified=len(rows),measurements=sum(r['counts']['measurements'] for r in rows),
        retrieval_errors=discovered['errors'])
    dump(out/'inventory.json',dict(schema='hgkda.native-to-delivery-inventory/v1',summary=summary,operators=rows))
    head=['# 26项：FlagGems原始起点—历史版本—交付终点','','日期：2026-11-17。只读查档，未补跑设备实验。',
          '', '固定FlagGems：`d6a8eec473517a3d68157b208eb9c057eb1d4c50`。26项的原始源文件均与当时计划记录SHA一致，26份起点送测源码的原始模块AST均未改变，末尾仅追加评测入口。该核验不自动证明动态后端派发路径。',
          '', '**已找到全部26项的基线送测记录和交付源码／结果。** 原始FlagGems送测基线与优化账本中首次修改后的baseline是两类记录，旧图只读取后者。',
          '', '下表起点分数为当时参考下的历史记录，终点为当前CANN参考；两列不能直接相除解释为优化收益。失败保留原状态。中间测量数量不等于模型轮次，包含不同批次及重复核验；厂商参考测量单独标记。',
          '', '| 算子 | 原生送测起点 | 候选测量／源码数 | 交付 CANN | 完整评测 | 证据 |',
          '| --- | --- | ---: | ---: | ---: | --- |']
    pages=[]
    for row in rows:
        op=row['operator']; b=row['baseline']; d=row['delivery'];m=b['measurement'];c=row['counts']
        score=f"{m['geo_mean']:.3f}×" if m['status']=='PASSED' and m.get('geo_mean') is not None else m['status']
        head.append(f"| `{op}` | {score} | {c['candidate_measurements']}／{c['candidate_sources']} | {d['cann_score']:.3f}× | {d['passed']}/{d['total']} | [{op}]({op}.md) |")
        lines=['# '+op,'','## 原始起点','',
               '- '+local_link(root/row['upstream']['path'],'固定版FlagGems源码')+'；SHA `'+row['upstream']['sha256']+'`。',
               '- '+local_link(evidence/b['sources'][0]['export'],'实际送测源码')+'；原始模块AST一致，追加入口：`'+'; '.join(b['identity']['appended_statements']).replace('\n',' ')+'`。',
               '- '+local_link(evidence/b['result']['export'],'起点原始结果')+'：**'+score+'**；'+str(m.get('num_passed'))+'/'+str(m.get('num_workloads'))+'。',
               '- '+local_link(evidence/b['request']['export'],'起点请求／合同绑定')+'；服务器原件：`'+b['result']['path']+'`。',
               '', '## 关键阶段','',
               '| 阶段 | 已有证据 |','| --- | --- |',
               '| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |',
               '| 早期优化 | '+str(c['ledger_points'])+'条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |',
               '| 后续修改与回归 | '+str(c['candidate_measurements'])+'份候选测量、'+str(c['candidate_sources'])+'份请求源码；详细位置列在下表 |',
               '| 交付终点 | '+f"{d['cann_score']:.6f}× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |", 
               '', '## 全部测量入口','',
               '按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。','',
               '| 类型 | 实验位置 | 结果 | 源码 |','| --- | --- | --- | --- |']
        for r in row['measurements']:
            path=r['result']['path'];short=path.split('/campaigns/')[-1].split('/runs/')[-1]
            val=r['measurement'];v=f"{val['geo_mean']:.6f}×" if val.get('geo_mean') is not None else str(val['status'])
            if val.get('geo_mean') is not None and val['status']!='PASSED':v+='（'+str(val['status'])+'）'
            sources=' · '.join(local_link(evidence/s['export'],s['sha256'][:12]) for s in r.get('sources',[])) or '请求源码未定位'
            role={'baseline':'基线目录','reference':'厂商参考','candidate':'候选'}[r['role']]
            if r['constraint_status']=='known_noncompliant':role+='／不符合当前约束'
            lines.append('| '+role+' | `'+short+'` | '+v+' | '+sources+' |')
        lines+=['','## 交付终点','',
                '- '+local_link(root/d['path'],'选用源码')+'：`'+d['sha256']+'`。',
                '- '+local_link(root/d['result'],'保留版完整结果')+f"：**{d['cann_score']:.6f}× CANN，{d['passed']}/{d['total']}**。",
                '- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。',
                '', '## 待补齐的连接','']+['- '+g for g in row['gaps']]
        (out/(op+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf8')
        pages.append(f'<tr><td><a href="{op}.md">{op}</a></td><td>{score}</td><td>{c["candidate_measurements"]} / {c["candidate_sources"]}</td><td>{d["cann_score"]:.3f}×</td><td>{d["passed"]}/{d["total"]}</td></tr>')
    head+=['','## 本轮查清的事实','',
           '- 原生送测源码、优化器第一轮和后续重生成起点必须分别记录。26项原生送测记录都已找到，不能再以优化账本第一点代替它。10月21日批次覆盖早期8项交付算子，10月29日批次覆盖其余18项；按固定批次选起点，没有挑选最低分。',
           '- 原来单点的QKV、log2、hardswish_out已找到多份中间候选；cachemask与batch_norm也找到基线、参考切换和严格Triton恢复记录。',
           '- 并非26项都从很低的有效成绩开始。cachemask原历史参考下为8.651×、batch_norm为100.912×，均需结合参考执行路径解释，不与CANN交付分数直接比较。',
           '- 原生失败不统一归为算子错误：scatter_add早期记录是相对导入失败，上采样早期记录包含参考接口错误；后续修正基线保留在各项完整清单。',
           '- 本轮只完成证据清单，不声称497条测量已经形成26条无缺口的继承链。下一步按源码SHA和父版本记录接线；不同参考的段可同图展示，不能把参考切换当作性能收益。',
           '', '## 范围与复现','',
           f"检索两个服务器历史根目录，找到{summary['measurements']}份测量、{summary['baseline_records']}份基线目录测量，检索解析错误{len(summary['retrieval_errors'])}项。所有导出的源码／基线原件SHA已核验，26份交付源码和结果与当前索引一致。",
           '', '源码与基线原件：'+local_link(evidence/'inventory.json','原始检索索引')+'；完整结构化清单：[inventory.json](inventory.json)。',
           '', '```powershell',
           'python -m development.tools.traces.build_lineage_inventory --evidence development/experiments/lineage/20261117/source-evidence --output development/reports/native-to-delivery-26-20261117',
           '```','']
    (out/'README.md').write_text('\n'.join(head),encoding='utf8')
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>26项起点与交付证据</title><style>body{background:#fff7e8;color:#1b2434;font:20px/1.6 Microsoft YaHei,sans-serif;margin:40px}table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #d8d5c8;text-align:left}a{color:#2368b1}h1{font-size:34px}</style><h1>26项：原始起点 → 历史版本 → 交付终点</h1><p>26项基线记录与交付结果已定位。历史起点与CANN终点的参考不同，分段理解。</p><p><a href="README.md">完整说明</a> · <a href="inventory.json">结构化证据清单</a></p><table><tr><th>算子</th><th>历史原生起点</th><th>候选测量 / 源码数</th><th>交付 CANN</th><th>完整评测</th></tr>'+''.join(pages)+'</table></html>',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
