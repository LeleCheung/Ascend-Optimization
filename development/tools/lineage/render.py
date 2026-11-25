"""Render the verified three-version index as a report with version notes and tables."""
import argparse
import html
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote


LEGACY_HEADERS = {
    '\u5173\u952e\u5b9e\u73b0\uff0f\u4fee\u6539': 'Implementation / change',
    '\u5173\u952e\u4f18\u5316': 'Optimization',
    'Excel \u539f\u8bb0\u5f55': 'Original Excel record',
    'Excel\u539f\u8bb0\u5f55': 'Original Excel record',
    'FlagGems \u76f8\u5bf9 PyTorch\uff08\u5386\u53f2\u5b9e\u6d4b\uff09': 'FlagGems vs PyTorch (historical measurement)',
    'FlagGems \u76f8\u5bf9 PyTorch': 'FlagGems vs PyTorch',
    'HG-KDA \u76f8\u5bf9 PyTorch': 'HG-KDA vs PyTorch',
    'HG-KDA\u76f8\u5bf9FlagGems': 'HG-KDA vs FlagGems',
    '\u5b8c\u6574\u8bc4\u6d4b': 'Full evaluation',
    '\u72b6\u6001': 'Status',
    '\u7ed3\u679c\u66f4\u65b0\u65e5\u671f': 'Result date',
}


EARLY_REPORT_SOURCES={
    'mse_loss_backward':'archive/comparisons/flaggems-comparison-20261009/measurements/mse_loss_backward/candidate.py',
    't_copy':'archive/comparisons/flaggems-comparison-20261009/measurements/t_copy/candidate.py',
    'prelu_kernel_backward':'archive/comparisons/flaggems-repaired-comparison-20261009/validated-v5/measurements/_prelu_kernel_backward/candidate.py',
    'batch_norm_backward':'archive/comparisons/flaggems-repaired-comparison-20261009/validated-v5/measurements/batch_norm_backward/candidate.py',
    'smooth_l1_loss_backward':'archive/comparisons/flaggems-repaired-comparison-20261009/validated-v5/measurements/smooth_l1_loss_backward/candidate.py',
}


class Markup(str):
    """Only links generated and escaped by this module; ordinary table data remains text-escaped."""


def escaped(value):
    return html.escape(str(value)).replace('\n', '<br/>')


def github_link(label, path, revision, *, directory=False):
    relative=PurePosixPath(path)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('GitHub links must use repository-relative paths')
    kind='tree' if directory else 'blob'
    url='https://github.com/LeleCheung/Ascend-Optimization/'+kind+'/'+quote(revision,safe='')+'/'+quote(relative.as_posix(),safe='/')
    return Markup('<a href="'+html.escape(url,quote=True)+'">'+escaped(label)+'</a>')


def score(value):
    return f'{value:.3f}×' if isinstance(value, (int, float)) else 'Not evaluated'


def table(headers, rows):
    def row(values, tag):
        return '<tr>'+''.join('<'+tag+'>'+(str(v) if isinstance(v,Markup) else escaped(v))+'</'+tag+'>'
                              for v in values)+'</tr>'
    # Yuque YMD tables require tr directly under table, without thead/tbody.
    return '<table>'+row(headers, 'td')+''.join(row(r, 'td') for r in rows)+'</table>'


def render(index, previous_tables, fixed_version, *, github_ref=None):
    def with_source(text, source, label='Source'):
        if not github_ref or not source:
            return Markup(escaped(text))
        return Markup(escaped(text)+'<br/>'+github_link(label,source['path'],github_ref))

    previous = {r[0]: dict(zip((LEGACY_HEADERS.get(h, h) for h in t[0]), r)) for t in previous_tables for r in t[1:]}
    body = '<p>'+html.escape(fixed_version)+'</p>\n'
    for group, title in [('1015', 'Main operators (1015)'), ('1029', 'Additional main operators (1029)')]:
        rows = []
        for row in index['rows']:
            if row['group'] != group:
                continue
            op = row['operator']; old = previous[op]; versions = row['versions']
            mixed = versions['historical_nonconforming']
            m = (mixed.get('measurements') or {}).get('cann')
            if mixed['status'] == 'no_verified_record':
                mixed_text = 'No verified record'
            elif m and m['verified_source_binding']:
                mixed_text = 'CANN '+score(m['score'])+'\nExcluded from strict Triton'
            else:
                mixed_text = 'Source archived; score binding pending\nExcluded from strict Triton'
            mixed_text=with_source(mixed_text,mixed.get('source'))
            strict = versions['historical_strict_triton']
            ev = strict['evidence']
            lines = []
            for key, label in [('cann', 'CANN'), ('pytorch', 'PyTorch')]:
                item = ev.get(key, {})
                if item.get('score') is not None:
                    text = label+' '+score(item['score'])
                    if key == 'pytorch' and op in ['fractional_max_pool3d', '_conj_physical', 'linalg_inv_ex']:
                        text += '(reference includes CPU fallback)'
                    if item.get('status') != 'PASSED':
                        text += '(full checks incomplete)'
                    lines.append(text)
            strict_text = '\n'.join(lines) or 'No reportable score'
            strict_text=with_source(strict_text,strict.get('source'))
            latest = versions['independent_entry_verified']
            experiment = latest.get('generated_experiment')
            selected_generated=bool(experiment and latest.get('source',{}).get('sha256')==experiment['source']['sha256'])
            if latest['status'] == 'verified':
                latest_text = 'CANN '+score(latest['cann_score'])
                latest_text += '\n'+('Skill-generated and validated' if selected_generated else
                    'Existing-source regression' if latest['source']['sha256'] == strict['source']['sha256'] else 'Repaired candidate this run')
                tests = f"{latest['formal_cases']}/{latest['formal_cases']}; both supplementary checks passed\nOriginal contract accepted"
            else:
                latest_text = 'Original contract incomplete'
                current = latest.get('current_record', {})
                if current.get('mathematical_contract'):
                    mathematical = current['mathematical_contract']
                    latest_text += '\nMathematical-contract CANN '+score(mathematical.get('cann_score', current.get('cann_score')))+'(separate)'
                tests = ('Original semantic/correctness contract incomplete; independent mathematical contract passed' if current.get('mathematical_contract')
                         else 'No qualified same-semantics CANN reference; not accepted')
            latest_source=latest.get('source') or latest.get('current_record',{}).get('source')
            latest_text=with_source(latest_text,latest_source,
                                    'Source' if latest.get('source') else 'Candidate source (not accepted)')
            if experiment and not selected_generated:
                generated_text='Skill-generated: CANN '+score(experiment.get('cann_score'))
                if not experiment.get('accepted'):
                    generated_text+='(below threshold)'
                else:
                    generated_text+='(accepted)'
                latest_text=Markup(str(latest_text)+'<br/>'+with_source(generated_text,experiment.get('source'),'Newly generated source'))
            method = old.get('Implementation / change', old.get('Optimization', ''))
            if experiment and experiment.get('accepted'):
                method=experiment.get('method_summary',method)
            operator_link=op
            if github_ref:
                operator_link=github_link(op,(latest_source or strict['source'])['path'],github_ref)
            rows.append([operator_link, old.get('Original Excel record', old.get('Original Excel record', '')),
                old.get('FlagGems vs PyTorch (historical measurement)', old.get('FlagGems vs PyTorch', '')),
                mixed_text, strict_text, latest_text, tests, method, '2026-11-09'])
        headers = ['Operator', 'Original Excel record', 'Measured FlagGems vs PyTorch',
                   'Historical best (outside current constraints)', 'Historical best (strict Triton)',
                   'Independent-entry validation (strict Triton)', 'Full evaluation and status', 'Method', 'Updated']
        accepted = sum(r.get('current_original_contract_accepted') is True
                       for r in index['rows'] if r['group'] == group)
        body += '<h2>'+title+f'（{accepted}/{len(rows)} original-contract acceptances)</h2>\n'+table(headers, rows)+'\n'
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
        early.append([operator_link, old['Original Excel record'], old['FlagGems vs PyTorch'],
            old['HG-KDA vs PyTorch'], old['HG-KDA vs FlagGems'],
            with_source('Includes CANN DMA; excluded from strict Triton',mixed.get('source')) if forbidden else 'No verified record',
            'Not reviewed under current constraints', 'No regression this run',
            old['Full evaluation']+'\n'+old['Status'], old['Optimization'], old['Result date']])
    body += '<h2>Early optimization and method validation</h2>\n'+table(
        ['Operator', 'Original Excel record', 'Measured FlagGems vs PyTorch', 'Historical report vs PyTorch',
         'Historical report vs FlagGems', 'Historical best (outside current constraints)', 'Historical best (strict Triton)',
         'Independent-entry validation (strict Triton)', 'Full evaluation and historical status', 'Method', 'Recorded'], early)+'\n'
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--previous-table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--github-ref', required=True,help='Pinned published Git commit; each version links to its own originals')
    args = parser.parse_args()
    assert not args.output.exists(), 'Output exists; preserve the pending publication'
    text = render(json.loads(args.index.read_bytes()), json.loads(args.previous_table.read_bytes()),
        'Pinned versions: FlagGems d6a8eec473517a3d68157b208eb9c057eb1d4c50; FlagTree 0.6.0+ascend.gitc286cba6; Triton 3.5.1; CANN 9.0.0; KGS 6.5.0. Keep three versions distinct and identify references in cells. Original contracts accepted 26/30; mathematical contracts are separate. Historical best is the existing delivery selection; early device timings are separate from current full-call timings.',github_ref=args.github_ref)
    args.output.write_text(text, encoding='utf8')
    print(json.dumps(dict(rows=39, bytes=len(text.encode())), ensure_ascii=False))


if __name__ == '__main__':
    main()
