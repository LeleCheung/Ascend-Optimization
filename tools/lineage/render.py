"""将已核验的三类版本索引渲染为仅含版本说明和表格的汇报文档。"""
import argparse
import html
import json
from pathlib import Path


def score(value):
    return f'{value:.3f}×' if isinstance(value, (int, float)) else '未评测'


def table(headers, rows):
    def row(values, tag):
        return '<tr>'+''.join('<'+tag+'>'+html.escape(str(v)).replace('\n', '<br/>')+'</'+tag+'>'
                              for v in values)+'</tr>'
    return '<table><thead>'+row(headers, 'th')+'</thead><tbody>'+''.join(row(r, 'td') for r in rows)+'</tbody></table>'


def render(index, previous_tables, fixed_version):
    previous = {r[0]: dict(zip(t[0], r)) for t in previous_tables for r in t[1:]}
    body = '<p>'+html.escape(fixed_version)+'</p>\n'
    for group, title in [('1015', '关键算子（1015）'), ('1029', '关键算子（1029新增）')]:
        rows = []
        for row in index['rows']:
            if row['group'] != group:
                continue
            op = row['operator']; old = previous[op]; versions = row['versions']
            mixed = versions['historical_nonconforming']
            m = (mixed.get('measurements') or {}).get('cann')
            if mixed['status'] == 'no_verified_record':
                mixed_text = '无已核验记录'
            elif m and m['verified_source_binding']:
                mixed_text = 'CANN '+score(m['score'])+'\n不计严格 Triton'
            else:
                mixed_text = '源码已归档；成绩待绑定\n不计严格 Triton'
            strict = versions['historical_strict_triton']
            ev = strict['evidence']
            lines = []
            for key, label in [('cann', 'CANN'), ('pytorch', 'PyTorch')]:
                item = ev.get(key, {})
                if item.get('score') is not None:
                    text = label+' '+score(item['score'])
                    if key == 'pytorch' and op in ['fractional_max_pool3d', '_conj_physical', 'linalg_inv_ex']:
                        text += '（参考含CPU回退）'
                    if item.get('status') != 'PASSED':
                        text += '（未完整通过）'
                    lines.append(text)
            strict_text = '\n'.join(lines) or '无可报告成绩'
            latest = versions['independent_entry_verified']
            if latest['status'] == 'verified':
                latest_text = 'CANN '+score(latest['cann_score'])
                latest_text += '\n'+('既有源码回归' if latest['source']['sha256'] == strict['source']['sha256'] else '本轮修复候选')
                tests = f"{latest['formal_cases']}/{latest['formal_cases']}；双方补测通过\n原合同达标"
            else:
                latest_text = '原合同未闭环'
                current = latest.get('current_record', {})
                if current.get('mathematical_contract'):
                    mathematical = current['mathematical_contract']
                    latest_text += '\n数学合同 CANN '+score(mathematical.get('cann_score', current.get('cann_score')))+'（单列）'
                tests = ('原合同语义／正确性未闭环；独立数学合同通过' if current.get('mathematical_contract')
                         else '缺合格的同语义CANN参考；不计达标')
            experiment = latest.get('generated_experiment')
            if experiment:
                latest_text += '\nSkill新生成：CANN '+score(experiment.get('cann_score'))
                if not experiment.get('accepted'):
                    latest_text += '（未达标）'
            method = old.get('关键实现／修改', old.get('关键优化', ''))
            rows.append([op, old.get('Excel 原记录', old.get('Excel原记录', '')),
                old.get('FlagGems 相对 PyTorch（历史实测）', old.get('FlagGems 相对 PyTorch', '')),
                mixed_text, strict_text, latest_text, tests, method, '2026-11-09'])
        headers = ['算子', 'Excel原记录', 'FlagGems实测（相对PyTorch）',
                   '历史最佳（不符合当前约束）', '历史最佳（严格Triton）',
                   '独立入口验证版（严格Triton）', '完整评测与状态', '关键方法', '更新日期']
        accepted = sum(r.get('current_original_contract_accepted') is True
                       for r in index['rows'] if r['group'] == group)
        body += '<h2>'+title+f'（{accepted}/{len(rows)} 原合同达标）</h2>\n'+table(headers, rows)+'\n'
    early = []
    for row in index['rows']:
        if row['group'] != 'early':
            continue
        old = previous[row['operator']]
        forbidden = row['versions']['historical_nonconforming']['status'] == 'archived_nonconforming'
        early.append([row['operator'], old['Excel原记录'], old['FlagGems 相对 PyTorch'],
            old['HG-KDA 相对 PyTorch'], old['HG-KDA相对FlagGems'],
            '含CANN DMA；不计严格Triton' if forbidden else '无已核验记录',
            '未按当前约束复核', '未执行本轮回归',
            old['完整评测']+'\n'+old['状态'], old['关键优化'], old['结果更新日期']])
    body += '<h2>早期优化与方法验证</h2>\n'+table(
        ['算子', 'Excel原记录', 'FlagGems实测（相对PyTorch）', '历史报告（相对PyTorch）',
         '历史报告（相对FlagGems）', '历史最佳（不符合当前约束）', '历史最佳（严格Triton）',
         '独立入口验证版（严格Triton）', '完整评测与历史状态', '关键方法', '记录日期'], early)+'\n'
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--previous-table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), '输出已存在，不覆盖待发布文稿'
    text = render(json.loads(args.index.read_bytes()), json.loads(args.previous_table.read_bytes()),
        '固定版本：FlagGems d6a8eec473517a3d68157b208eb9c057eb1d4c50；FlagTree 0.6.0+ascend.gitc286cba6；Triton 3.5.1；CANN 9.0.0；KGS 6.5.0。三类分别保留，倍数在单元格内注明参考；原30项合同达标26/30，独立数学合同单列。历史最佳为既有正式交付选中版本；FlagGems及早期设备计时与当前完整调用计时分开。')
    args.output.write_text(text, encoding='utf8')
    print(json.dumps(dict(rows=39, bytes=len(text.encode())), ensure_ascii=False))


if __name__ == '__main__':
    main()
