"""生成可跨Windows/910B核验的最佳源码索引；不运行设备任务，不复制候选代码。"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

from report import verify, reference_stability


REMOTE_PROJECT = '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1'
PROJECT = Path(__file__).resolve().parents[4]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def io_path(path):
    path = Path(path)
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        return Path('\\\\?\\' + str(path.resolve()))
    return path


def read(path):
    return json.loads(io_path(path).read_bytes())


def resolve(project, name):
    text = str(name)
    if text.startswith('\\\\?\\'):
        text = text[4:]
    if text.startswith(REMOTE_PROJECT + '/'):
        path = project / text[len(REMOTE_PROJECT) + 1:]
    else:
        path = Path(text)
        if not path.is_absolute():
            path = project / path
    path = path.resolve()
    path.relative_to(project.resolve())
    return path


def build(project, batch):
    project = project.resolve()
    state_bytes = io_path(batch / 'next-state.json').read_bytes()
    state = json.loads(state_bytes)
    targets = read(batch / 'targets.json')['targets']
    plan = read(batch / 'plan.json')
    contracts = {r['operator']: r for r in plan['records']}
    files = {}

    def include(path, expected=None):
        path = resolve(project, path)
        content = io_path(path).read_bytes()
        sha = digest(content)
        if expected and sha != expected:
            raise ValueError('证据SHA不符：' + str(path))
        relative = path.relative_to(project).as_posix()
        files[relative] = dict(sha256=sha, bytes=len(content))
        return relative

    for name in ['software-lock.json', 'targets.json', 'plan.json']:
        include(batch / name)
    rows = []
    for target in targets:
        op = target['operator']
        current = state.get('operators', {}).get(op, {})
        record = current.get('best') or {}
        row = dict(order=target['order'], operator=op, category='not_qualified',
                   status=current.get('status', target.get('status')), source=None)
        stability = reference_stability(batch, op)
        if stability['status'] not in {'not_applicable', 'not_run'}:
            row['reference_stability'] = stability
            for path in sorted((batch / 'analysis/cumulative_trapezoid/reference-stability-replay-v1').iterdir()):
                if path.is_file():
                    include(path)
            if stability['status'] == 'default_reference_inconsistent':
                row['status'] = '默认参考重复不自洽；确定性条件需单列'
        rows.append(row)
        if not record:
            continue
        if stability['status'] in {'default_reference_inconsistent', 'evidence_requires_review'}:
            continue
        local = dict(record)
        for field in ['source', 'result', 'extra_checks']:
            if local.get(field):
                local[field] = str(io_path(resolve(project, local[field])))
        check = verify(local)
        row['verification'] = check['status']
        row['supplementary_status'] = check.get('extra_status')
        if check['status'] != 'verified' or not check.get('correctness_passed') or check.get('extra_status') != 'passed':
            continue
        extra = read(Path(local['extra_checks']))
        if extra.get('operator') != op or not extra.get('total') or extra.get('passed') != extra['total']:
            row['supplementary_status'] = 'invalid_counts_or_operator'
            continue
        score = check.get('score')
        if not isinstance(score, (float, int)) or not math.isfinite(score) or score <= 0:
            continue
        source = include(local['source'], check['source_sha256'])
        result = include(local['result'], check['result_sha256'])
        request = include(check['request'])
        gates = include(local['extra_checks'])
        checker = extra.get('checker_source') or {}
        if checker.get('path'):
            include(checker['path'], checker['sha256'])
        if extra.get('random_details'):
            details = extra['random_details']
            include(details['path'], details['sha256'])
            include(project / 'tools/profiling/hgkda/check-endurance.py', details['checker_sha256'])
        contract = contracts[op]
        leaf = batch / contract['catalog_leaf']
        contract_files = {name: include(leaf / name, sha)
                          for name, sha in contract.get('contract_sha256', {}).items()}
        if not contract_files:
            raise ValueError('缺少冻结合同SHA：' + op)
        reference = current.get('reference_path_status', 'not_run')
        audit = batch / 'reference-audits' / op / 'result.json'
        if audit.exists():
            include(audit)
            for child in sorted(audit.parent.glob('case-*/result.json')):
                include(child)
        baseline_sha = (current.get('baseline') or {}).get('source_sha256')
        historical_sha = (current.get('starting_point') or {}).get('source_sha256')
        if reference != 'npu_paths_observed':
            category = 'reference_requires_separate_review'
        elif score <= .8:
            category = 'correct_below_threshold'
        elif check['source_sha256'] == baseline_sha:
            category = 'original_retained'
        elif check['source_sha256'] == historical_sha:
            category = 'historical_threshold_passed'
        else:
            category = 'new_threshold_passed'
        row.update(category=category, source=source, source_sha256=check['source_sha256'],
                   speedup=score, complete_tests=[check['passed'], check['total']],
                   supplementary_tests=[extra['passed'], extra['total']],
                   formal_result=result, formal_request=request, supplementary_result=gates,
                   contract=contract_files, reference_path_status=reference,
                   attribution=record.get('attribution', 'HG-KDA与Codex参与；具体来源见原批次记录'))
    categories = sorted({r['category'] for r in rows})
    return dict(schema='hgkda-source-index-v1',
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                timing_scope='KGS19659 walltime完整调用，相对同环境PyTorch', threshold=.8,
                batch=batch.relative_to(project).as_posix(), queue_status=state.get('status'),
                source_state_sha256=digest(state_bytes), current_job=state.get('current_job'),
                counts={c: sum(r['category'] == c for r in rows) for c in categories},
                operators=rows, files=files)


def verify_index(project, index):
    for name, expected in index['files'].items():
        data = io_path(resolve(project, name)).read_bytes()
        if digest(data) != expected['sha256'] or len(data) != expected['bytes']:
            raise ValueError('索引文件已变化：' + name)
    for row in index['operators']:
        if not row.get('source'):
            continue
        source = io_path(resolve(project, row['source'])).read_text(encoding='utf8')
        request = read(resolve(project, row['formal_request']))
        sources = request['implementation']['sources']
        if len(sources) != 1 or sources[0]['content'] != source:
            raise ValueError('请求与交付源码不一致：' + row['operator'])
        if digest(io_path(resolve(project, row['source'])).read_bytes()) != row['source_sha256']:
            raise ValueError('源码身份不一致：' + row['operator'])
    return dict(status='verified', files=len(index['files']), operators=len(index['operators']), counts=index['counts'])


def markdown(index):
    labels = dict(new_threshold_passed='本批新达标', historical_threshold_passed='历史源码复验达标',
                  original_retained='保留原版', correct_below_threshold='正确，性能未达标',
                  reference_requires_separate_review='参考条件单列', not_qualified='暂无完整合格实现')
    lines = ['# 本批最佳源码索引', '', index['timing_scope'] + '。保持导师目标顺序；代码链接相对项目根目录。', '',
             '|算子|加速比|完整评测|补测|结论|源码|', '|---|---:|---|---|---|---|']
    for row in index['operators']:
        code = '`' + row['source'] + '`' if row.get('source') else '—'
        score = f"{row['speedup']:.3f}×" if row.get('source') else '—'
        formal = '/'.join(map(str, row.get('complete_tests', []))) or '—'
        extra = '/'.join(map(str, row.get('supplementary_tests', []))) or '—'
        lines.append('|'+ '|'.join([row['operator'], score, formal, extra, labels[row['category']], code]) + '|')
    lines += ['', 'JSON索引保存相对路径、逐文件SHA、固定合同、正式请求及补测原件位置。',
              '索引是当前时间点快照；后续队列仍可产生更佳版本。验证命令不占用NPU：', '',
              '```bash', 'python tools/profiling/hgkda/next_phase/export_best.py --verify <本索引.json>', '```', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=PROJECT)
    p.add_argument('--batch', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--verify', type=Path)
    args = p.parse_args()
    if args.verify:
        print(json.dumps(verify_index(args.project, read(args.verify)), ensure_ascii=False))
        return
    if not args.batch or not args.output:
        p.error('生成时需要--batch及--output')
    outputs = [args.output, args.output.with_suffix('.md')]
    if outputs[0] == outputs[1] or any(path.exists() for path in outputs):
        raise FileExistsError('输出必须使用新的JSON与Markdown路径，保留既有索引')
    index = build(args.project, args.batch.resolve())
    summary = verify_index(args.project, index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    args.output.with_suffix('.md').write_text(markdown(index), encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
