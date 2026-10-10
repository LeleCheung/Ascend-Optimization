#!/usr/bin/env python3
"""归档 KG 的结构化 profiling 分析和原始附件，不收集 Claude 配置或凭据。"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    work = workspace / 'stages/optimize/work'
    progress = json.loads((workspace / '.kernelgen/run-progress.json').read_text(encoding='utf-8'))
    assert progress['state'] in {'SUCCEEDED', 'FAILED', 'CANCELLED'}, '只归档已经结束的工作区'
    selected = []
    for name in ('profile-analysis', 'profiles'):
        root = work / '.kernelgen' / name
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            assert not path.is_symlink()
            path.resolve().relative_to(root.resolve())
            selected.append((path, Path(name) / path.relative_to(root)))
    # 遇到异常大的采集先停下检查；不悄悄丢弃附件。
    assert sum(path.stat().st_size for path, _ in selected) <= 512 * 1024 * 1024, '采集超过 512MiB，须单独检查归档范围'
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, (source, relative) in enumerate(selected, 1):
        data = source.read_bytes()
        # KG 原路径含多个完整 SHA 和 profile ID，在 Windows 超过路径上限。
        # 附件按编号平铺，原路径和哈希保留在清单中，内容不改写。
        archived = Path('files') / (f'{index:05d}-' + source.name)
        target = args.output / archived
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({'source_path': source.relative_to(work).as_posix(),
                        'original_archive_path': relative.as_posix(),
                        'archived_path': archived.as_posix(), 'size_bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()})
    (args.output / 'archive-manifest.json').write_text(json.dumps({
        'workspace': str(workspace), 'state': progress['state'], 'files': records,
        'format_version': 2,
        'note': '原始分析和 manifest 字节保持不变；附件使用短路径平铺，旧绝对路径可通过 source_path 映射到 archived_path。',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ARCHIVED_KG_PROFILE_EVIDENCE', len(records), sum(x['size_bytes'] for x in records), flush=True)


if __name__ == '__main__':
    main()
