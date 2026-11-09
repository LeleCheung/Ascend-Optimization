"""异机离线核验单算子skill试验原件及正式结果；冻结依赖映射到既有回归归档。"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    repo = args.repository.resolve()
    if os.name=='nt':
        # 原始厂商采集目录可能超过MAX_PATH；保留原路径和字节身份。
        def extended(path):
            return path if str(path).startswith('\\\\?\\') else Path('\\\\?\\'+str(path))
        root,repo=extended(root),extended(repo)
    assert not args.output.exists()
    read = lambda p: json.loads(p.read_bytes())
    manifest = read(root/'export-manifest.json')
    for record in manifest['files']:
        relative = PurePosixPath(record['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root.joinpath(*relative.parts)
        assert path.resolve().is_relative_to(root)
        raw = path.read_bytes()
        assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'], relative
    config = read(root/'config.json')
    prior_root = PurePosixPath(config['job']['request']).parents[2]
    original_root = PurePosixPath(manifest['source_run'])

    def local(value):
        path = PurePosixPath(value)
        if path.is_relative_to(original_root):
            return str(root.joinpath(*path.relative_to(original_root).parts))
        assert path.is_relative_to(prior_root), '未声明的外部依赖：'+value
        return str((repo/'benchmarks/regressions/20261109/fixed-25').joinpath(*path.relative_to(prior_root).parts))

    audit = runpy.run_path(str(repo/'tools/regression/audit.py'))
    formal = []
    for path in sorted(root.glob('formal-*/plan.json')):
        plan = read(path)
        for job in plan['jobs']:
            mapped = copy.deepcopy(job)
            for name in ['source', 'review', 'archive', 'request']:
                mapped[name] = local(mapped[name])
            for spec in mapped['checks'].values():
                for name in ['leaf', 'checker']:
                    spec[name] = local(spec[name])
            formal.append(audit['inspect_job'](mapped, path.parent))
    recovery=config.get('goal_kind')=='performance_recovery_with_formal_feedback'
    if recovery:
        summary=read(root/'summary.json')
        assert len(formal)==summary['formal_requests']
        selected=next(row for row in formal if row['source_sha256']==summary['source_sha256'])
        assert selected['full_passed'] and selected['accepted']
        for name in ['offset','runtime']:
            extra=read(root/'attempts'/selected['source_sha256']/(name+'-check.json'))
            assert extra['source_sha256']==selected['source_sha256']
            assert extra['passed']==extra['total']>0
        assert summary['cann_score']==selected['geo_mean']
    else:
        assert len(formal) == 2, '本轮应保留一次入口失败与一次新源码正式验证'
        assert sum(row['full_passed'] for row in formal) == 1
        assert read(root/'offset-check-before.json')['passed'] == 3
        assert read(root/'offset-check-final.json')['passed'] == 9
        assert read(root/'entry-equivalence.json')['device_kernels_unchanged']
    value = dict(status='passed', files_verified=len(manifest['files']), formal=formal,
                 scope=('启动路径恢复实验的原件、完整结果与专项补测' if recovery else
                        '原件身份及两次不同源码的完整请求；不重复设备评测，不将入口失败作为性能样本'))
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(status=value['status'], files_verified=value['files_verified'],
                          formal=[{k:r[k] for k in ['source_sha256','full_passed','geo_mean','accepted']} for r in formal])))


if __name__ == '__main__':
    main()
