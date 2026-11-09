"""将已核验的三类版本索引渲染为仅含版本说明和表格的汇报文档。"""
import argparse
import html
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote


EARLY_REPORT_SOURCES={
    'mse_loss_backward':'archive/comparisons/flaggems-comparison-20261009/measurements/mse_loss_backward/candidate.py',
    't_copy':'archive/comparisons/flaggems-comparison-20261009/measurements/t_copy/candidate.py',
    'prelu_kernel_backward':'archive/comparisons/flaggems-repaired-comparison-20261009/validated-v5/measurements/_prelu_kernel_backward/candidate.py',
    'batch_norm_backward':'archive/comparisons/flaggems-repaired-comparison-20261009/validated-v5/measurements/batch_norm_backward/candidate.py',
    'smooth_l1_loss_backward':'archive/comparisons/flaggems-repaired-comparison-20261009/validated-v5/measurements/smooth_l1_loss_backward/candidate.py',
}


class Markup(str):
    """仅承载本模块生成并转义过的链接，表格普通数据仍按文本转义。"""


def escaped(value):
    return html.escape(str(value)).replace('\n', '<br/>')


def github_link(label, path, revision, *, directory=False):
    relative=PurePosixPath(path)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('GitHub链接必须使用仓库内相对路径')
    kind='tree' if directory else 'blob'
    url='https://github.com/LeleCheung/Ascend-Optimization/'+kind+'/'+quote(revision,safe='')+'/'+quote(relative.as_posix(),safe='/')
    return Markup('<a href="'+html.escape(url,quote=True)+'">'+escaped(label)+'</a>')


def score(value):
    return f'{value:.3f}×' if isinstance(value, (int, float)) else '未评测'


def table(headers, rows):
    def row(values, tag):
        return '<tr>'+''.join('<'+tag+'>'+(str(v) if isinstance(v,Markup) else escaped(v))+'</'+tag+'>'
                              for v in values)+'</tr>'
    # 语雀YMD的HTML表格要求table直接包含tr，不接受thead/tbody。
    return '<table>'+row(headers, 'td')+''.join(row(r, 'td') for r in rows)+'</table>'


def render(index, previous_tables, fixed_version, *, github_ref=None):
    def with_source(text, source, label='源码'):
        if not github_ref or not source:
            return Markup(escaped(text))
        return Markup(escaped(text)+'<br/>'+github_link(label,source['path'],github_ref))

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
            mixed_text=with_source(mixed_text,mixed.get('source'))
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
            strict_text=with_source(strict_text,strict.get('source'))
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
            latest_source=latest.get('source') or latest.get('current_record',{}).get('source')
            latest_text=with_source(latest_text,latest_source,
                                    '源码' if latest.get('source') else '候选源码（未验收）')
            experiment = latest.get('generated_experiment')
            if experiment:
                generated_text='Skill新生成：CANN '+score(experiment.get('cann_score'))
                if not experiment.get('accepted'):
                    generated_text+='（未达标）'
                latest_text=Markup(str(latest_text)+'<br/>'+with_source(generated_text,experiment.get('source'),'新生成源码'))
            method = old.get('关键实现／修改', old.get('关键优化', ''))
            operator_link=op
            if github_ref:
                operator_link=github_link(op,(latest_source or strict['source'])['path'],github_ref)
            rows.append([operator_link, old.get('Excel 原记录', old.get('Excel原记录', '')),
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
        mixed=row['versions']['historical_nonconforming']
        forbidden = mixed['status'] == 'archived_nonconforming'
        operator_link=row['operator']
        if github_ref:
            if forbidden:
                operator_link=github_link(row['operator'],mixed['source']['path'],github_ref)
            elif row['operator'] in EARLY_REPORT_SOURCES:
                operator_link=github_link(row['operator'],'project/Ascend-Optimization-v1/'+EARLY_REPORT_SOURCES[row['operator']],
                                          index['archive_commit'])
            else:
                operator_link=github_link(row['operator'],'project/Ascend-Optimization-v1/operators/'+row['operator'],
                                          index['archive_commit'],directory=True)
        early.append([operator_link, old['Excel原记录'], old['FlagGems 相对 PyTorch'],
            old['HG-KDA 相对 PyTorch'], old['HG-KDA相对FlagGems'],
            with_source('含CANN DMA；不计严格Triton',mixed.get('source')) if forbidden else '无已核验记录',
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
    parser.add_argument('--github-ref', required=True,help='固定的已发布Git提交，三类版本各自链接到对应原件')
    args = parser.parse_args()
    assert not args.output.exists(), '输出已存在，不覆盖待发布文稿'
    text = render(json.loads(args.index.read_bytes()), json.loads(args.previous_table.read_bytes()),
        '固定版本：FlagGems d6a8eec473517a3d68157b208eb9c057eb1d4c50；FlagTree 0.6.0+ascend.gitc286cba6；Triton 3.5.1；CANN 9.0.0；KGS 6.5.0。三类分别保留，倍数在单元格内注明参考；原30项合同达标26/30，独立数学合同单列。历史最佳为既有正式交付选中版本；FlagGems及早期设备计时与当前完整调用计时分开。',github_ref=args.github_ref)
    args.output.write_text(text, encoding='utf8')
    print(json.dumps(dict(rows=39, bytes=len(text.encode())), ensure_ascii=False))


if __name__ == '__main__':
    main()
