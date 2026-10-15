#!/usr/bin/env python3
"""生成独立 KG 角色/skill 覆盖层，不修改基线 KG、KGS 或编译器。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def build(output: Path, *, repo=REPO):
    source = repo / 'sources/kernelgen'
    skill = ROOT / 'skills/hgkda'
    if output.exists():
        raise FileExistsError('覆盖层目录已存在，请使用新批次：' + str(output))
    output = output.resolve()
    if output == source.resolve() or source.resolve() in output.parents:
        raise ValueError('覆盖层不能写到基线 KG 源码内')
    output.mkdir(parents=True)
    target = output / 'kernelgen'
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        '.git', '__pycache__', '*.pyc', '.pytest_cache', 'kb', 'tests'))
    # 固定知识库与 KGS 源码只读复用，不另复制大目录或更新共享运行时。
    for name, original in [('kernelgen/kb', source / 'kb'),
                           ('kernelgen_server', repo / 'sources/kernelgen_server')]:
        if original.exists():
            (output / name).symlink_to(original.resolve(), target_is_directory=True)
    dest_skill = target / '.kernelgen/skills/hgkda'
    shutil.copytree(skill, dest_skill, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    body = (skill / 'SKILL.md').read_text(encoding='utf-8').split('---', 2)[2]
    references = '\n\n'.join(
        f'### {p.relative_to(skill).as_posix()}\n\n{p.read_text(encoding="utf-8")}'
        for p in sorted((skill / 'references').rglob('*.md')))
    appended = '\n\n# HG-KDA 角色扩展\n\n' + body + '\n\n' + references
    changed = {}
    for name in ['kernel-profile-analyzer', 'kernel-knowledge-profile-analyzer']:
        src = source / '.kernelgen/agents' / (name + '.md')
        dest = target / '.kernelgen/agents' / src.name
        # 保留原角色工具、证据登记要求及接口，只扩展诊断方法。
        dest.write_text(src.read_text(encoding='utf-8') + appended, encoding='utf-8')
        changed[name] = {'base_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                         'overlay_sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    data = {'schema_version': 1, 'base_source': str(source.resolve()),
            'overlay_source': str(target), 'roles': changed,
            'skill_files': {p.relative_to(skill).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(skill.rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
            'integration': '原生 KG 分析角色调用 KGS；额外分阶段方法与分类知识随角色加载',
            'compiler_changed': False, 'evaluator_changed': False}
    (output / 'hgkda-overlay.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
