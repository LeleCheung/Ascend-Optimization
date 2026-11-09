"""复用已审查交付与原始 Git 对象，建立三类版本的可核验索引，不运行设备实验。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PREFIX = 'project/Ascend-Optimization-v1/'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


class Archive:
    def __init__(self, repository, ref):
        self.repo = repository
        self.ref = ref
        self.commit = subprocess.check_output(['git', 'rev-parse', ref], cwd=repository).decode().strip()
        raw = subprocess.check_output(['git', 'ls-tree', '-r', '-z', ref, '--', PREFIX], cwd=repository)
        self.tree = {}
        for item in raw.split(b'\0'):
            if not item:
                continue
            header, path = item.split(b'\t', 1)
            self.tree[path.decode()] = header.split()[2].decode()
        self.process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=repository,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def path(self, value):
        if value.startswith('/'):
            return 'project/'+value.split('/project/', 1)[1]
        return value if value.startswith('project/') else PREFIX+value

    def get(self, value):
        path = self.path(value)
        oid = self.tree[path]
        self.process.stdin.write((oid+'\n').encode())
        self.process.stdin.flush()
        header = self.process.stdout.readline().split()
        raw = self.process.stdout.read(int(header[2]))
        assert self.process.stdout.read(1) == b'\n'
        return raw, path, oid

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    parser.add_argument('--archive-ref', default='archive/pre-release-20261103')
    parser.add_argument('--previous-table', type=Path, required=True,
                        help='语雀原表解析得到的三组二维数组，只读历史报告字段')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='恢复中断整理，已保存原件必须逐字节相同')
    parser.add_argument('--skill-validation', type=Path, help='已完成的代表算子实际生成试验原件目录')
    args = parser.parse_args()
    repo = args.repository.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    archive = Archive(repo, args.archive_ref)
    artifacts = []

    def original(value, destination, expected=None):
        raw, path, oid = archive.get(value)
        if expected:
            assert sha(raw) == expected, path
        target = output/destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == raw, target
        else:
            target.write_bytes(raw)
        record = dict(path=target.relative_to(repo).as_posix(), sha256=sha(raw),
                      original_path=path, git_oid=oid)
        artifacts.append(record)
        return record, raw

    def local(record):
        assert sha((repo/record['path']).read_bytes()) == record['sha256'], record
        return dict(record)

    def measurement(spec, source_sha, directory):
        result_path = spec.get('result') or spec.get('path')
        if not result_path:
            return None
        result_id, raw = original(result_path, directory/'result.json')
        result = json.loads(raw)
        score = result.get('geo_mean')
        if spec.get('score') is not None:
            assert score is not None and abs(score-spec['score']) < 1e-10, result_path
        p = Path(archive.path(result_path))
        request_path = p.with_name('request.json' if p.name == 'result.json'
                                   else p.name.replace('.result.json', '.request.json')).as_posix()
        binding = None
        if request_path in archive.tree:
            request_id, request_raw = original(request_path, directory/'request.json')
            request = json.loads(request_raw)
            code = request['implementation']['sources'][0]['content'].encode()
            # 源文件原件的字节摘要必须与正式请求直接相同。
            frozen_source = (output/directory.parent/'candidate.py').read_bytes()
            assert sha(frozen_source) == source_sha
            assert code.replace(b'\r\n', b'\n') == frozen_source.replace(b'\r\n', b'\n'), result_path
            binding = dict(request=request_id, source_sha256=sha(code),
                           source_match='exact_bytes' if sha(code) == source_sha else 'newline_equivalent',
                           frozen_source_sha256=source_sha,
                           bundle_id=request['binding']['bundle_id'], settings=request.get('settings'))
        return dict(reference='CANN', status=result.get('status'), score=score,
                    passed=result.get('num_passed'), total=result.get('num_workloads'),
                    is_hack=result.get('is_hack'), result=result_id, binding=binding,
                    verified_source_binding=binding is not None)

    try:
        inputs = {}
        for key, path in {
            'isolation': 'operators/campaigns/strict-triton-20261102/historical-isolation.json',
            'priority': 'operators/campaigns/cann-priority-20261102/delivery.json',
            'reviews': 'operators/campaigns/strict-triton-20261102/candidate-index.json',
        }.items():
            # 旧批次文件名由实际 Git 树确认；不存在时不猜测同名文件。
            if archive.path(path) not in archive.tree:
                matches = [p for p in archive.tree if '/strict-triton-20261102/' in p
                           and p.endswith('/'+Path(path).name)]
                assert len(matches) == 1, (key, matches)
                path = matches[0]
            record, raw = original(path, Path('inputs')/(key+'.json'))
            inputs[key] = json.loads(raw)
        frozen = read(repo/'benchmarks/index.json')
        current = read(repo/'benchmarks/current.json')
        previous = json.loads(args.previous_table.read_bytes())
        write(output/'inputs/previous-report-tables.json', previous)
        isolation = {r['operator']: r for r in inputs['isolation']['rows']}
        priority = {r['operator']: r for r in inputs['priority']['rows']}
        current_rows = {r['operator']: r for r in current['rows']}
        rows = []
        for fixed in frozen['rows']:
            op = fixed['operator']
            old = isolation[op]
            report = priority[op]
            now = current_rows[op]
            versions = {}
            if old['historical_disallowed']:
                selected = report.get('selected_new')
                source = selected['source'] if selected else old['historical_source']
                source_sha = selected['source_sha256'] if selected else old['historical_source_sha256']
                source_id, _ = original(source, Path('historical_nonconforming')/op/'candidate.py', source_sha)
                spec = selected.get('cann') if selected else report.get('starting_vs_cann')
                measured = measurement(spec, source_sha, Path('historical_nonconforming')/op/'cann') if spec else None
                review = inputs['reviews'].get(op, {})
                versions['historical_nonconforming'] = dict(
                    status='archived_nonconforming', source=source_id,
                    reason=('显式Ascend C设备路径' if selected else review.get('method', old['status'])),
                    isolation=old, measurements={'cann': measured},
                    selection='既有CANN优先批次的正式选中版本；未遍历全历史候选比较',
                    included_in_strict_delivery=False)
            else:
                versions['historical_nonconforming'] = dict(status='no_verified_record',
                    reason='该算子在本次核验的历史选中版本中未发现不合规版本；不人为补齐三份源码')
            source = local(fixed['source'])
            evidence = {name: dict(value, artifact=local(value['artifact']))
                        for name, value in fixed['evidence'].items() if 'artifact' in value}
            versions['historical_strict_triton'] = dict(status='reviewed_historical',
                source=source, review=local(fixed['review']), evidence=evidence,
                accepted=fixed['accepted'], reference_kind=fixed['reference_kind'],
                selection='拆分前最终交付中固定的历史最佳；不重新挑选同源码的最高复测分数')
            if now.get('original_contract_accepted'):
                versions['independent_entry_verified'] = dict(status='verified',
                    origin='existing_candidate_regression_or_codex_repair',
                    source=local(now['source']), cann_score=now['cann_score'],
                    formal_cases=now['formal_cases'], checks=now['checks'],
                    evidence=local(now['latest_evidence']),
                    generated_by_independent_skill=False)
            else:
                versions['independent_entry_verified'] = dict(status='not_verified_original_contract',
                    reason=now['status'], current_record=now)
            rows.append(dict(operator=op, group=fixed['group'], order=fixed['order'], versions=versions,
                             current_original_contract_accepted=now['original_contract_accepted']))
        for i, values in enumerate(previous[2][1:], 1):
            op = values[0]
            versions = {name: dict(status='not_reaudited', reason='早期报告未纳入本轮30项合同及逐路径复核')
                        for name in ['historical_nonconforming', 'historical_strict_triton', 'independent_entry_verified']}
            if op == 'narrow_copy':
                p = 'operators/narrow_copy/candidates/contiguous-dma-hybrid-20261010.py'
                record, raw = original(p, Path('historical_nonconforming')/op/'candidate.py')
                assert b'aclrtMemcpyAsync' in raw
                versions['historical_nonconforming'] = dict(status='archived_nonconforming',
                    source=record, reason='小型连续输入调用CANN aclrtMemcpyAsync，其他路径使用Triton',
                    included_in_strict_delivery=False, reported_pytorch_score=values[3],
                    performance_binding='原语雀报告值；本轮未重新绑定该成绩的完整请求')
            rows.append(dict(operator=op, group='early', order=i, versions=versions,
                historical_report=dict(zip(previous[2][0], values)),
                current_original_contract_accepted=None))
        result = dict(schema_version='hgkda.version-lineage/v1', date='2026-11-09',
            archive_commit=archive.commit, rows=rows, count=len(rows),
            current_original_contract_accepted=current['accepted'], target_count=current['total'],
            scope='历史最佳指已有正式交付所选版本；保留原口径，不宣称已穷举所有历史实验。早期9项只保留报告，缺少现行审查时不划入严格Triton。',
            artifacts=artifacts)
        if args.skill_validation:
            experiment_root=args.skill_validation.resolve()
            summary=read(experiment_root/'summary.json')
            op=summary['operator']
            target=next(row for row in rows if row['operator']==op)
            formal=experiment_root/summary['formal_directory']
            formal_row=read(formal/'audit-final.json')['rows'][0]
            assert formal_row['evidence_verified'] and formal_row['source_sha256']==summary['source_sha256']
            def binding(path):
                return dict(path=path.relative_to(repo).as_posix(),sha256=sha(path.read_bytes()))
            target['versions']['independent_entry_verified']['generated_experiment']=dict(
                status='full_passed_performance_pending' if not formal_row['accepted'] else 'accepted',
                generated_by_independent_skill=True,source=binding(formal/'operators'/op/'candidate.py'),
                review=binding(formal/'operators'/op/'review.json'),
                formal_result=binding(formal/'operators'/op/'evaluation/result.json'),
                cann_score=formal_row['geo_mean'],formal_cases=formal_row['formal_cases'],
                accepted=formal_row['accepted'],checks=formal_row['checks'],
                summary=binding(experiment_root/'summary.json'),audit=binding(formal/'audit-final.json'),
                retained_delivery=True,prior_best_source_provided=summary['prior_best_source_provided'])
        assert len(rows) == 39 and len({r['operator'] for r in rows}) == 39
        write(output/'index.json', result)
        print(json.dumps(dict(count=len(rows), archived_artifacts=len(artifacts),
                              accepted=current['accepted']), ensure_ascii=False))
    finally:
        archive.close()


if __name__ == '__main__':
    main()
