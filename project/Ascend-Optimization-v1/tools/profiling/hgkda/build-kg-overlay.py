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


def build(output: Path, *, repo=REPO, native_context=False, paired_evidence=None, roofline_evidence=None):
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
    paired_extension = {}
    if paired_evidence:
        raw = Path(paired_evidence).read_bytes()
        evidence = json.loads(raw)
        if evidence.get('schema_version') != 'hgkda-paired-v1' or not evidence.get('cases'):
            raise ValueError('成对证据缺少schema或case')
        if len(raw) > 128 * 1024:
            raise ValueError('角色内成对摘要过大，请保留紧凑摘要和原始证据路径')
        dest = dest_skill / 'paired-evidence.json'
        dest.write_bytes(raw)
        paired_extension = {'source': str(Path(paired_evidence).resolve()),
                            'sha256': hashlib.sha256(raw).hexdigest(), 'cases': len(evidence['cases'])}
        appended += ('\n\n## 本批起点的成对硬件证据\n\n'
            '以下是冻结seed与PyTorch的独立对照，不是本轮新候选的测量。核对operator、'
            '源码SHA和case参数；它用于提出修改，不能替代本轮KGS profiling或正式评测。'
            '如新候选改变执行路径，需要重新采集以检验机制。\n\n'
            + json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    roofline_extension = {}
    if roofline_evidence:
        raw = Path(roofline_evidence).read_bytes()
        evidence = json.loads(raw)
        if evidence.get('schema_version') != 'hgkda-roofline-v1' or not evidence.get('cases'):
            raise ValueError('Roofline摘要缺少schema或case')
        if len(raw) > 128 * 1024 or not evidence.get('source', {}).get('sha256'):
            raise ValueError('Roofline摘要过大或缺少完整分析SHA')
        for case in evidence['cases']:
            if not case.get('spec', {}).get('source_sha256') or not case.get('paths'):
                raise ValueError('Roofline case缺少源码或有效路径')
        (dest_skill / 'roofline-evidence.json').write_bytes(raw)
        roofline_extension = {'source': str(Path(roofline_evidence).resolve()),
            'sha256': hashlib.sha256(raw).hexdigest(), 'cases': len(evidence['cases'])}
        appended += ('\n\n## 本批冻结的Roofline数值证据\n\n'
            '先核对源码SHA、shape、dtype和采集范围。它是起点证据，不是新候选测量。'
            '有效算法FLOPs、硬件内部操作、官方模型和经验参考分别解释；'
            'PyTorch GEMM组件不代表完整融合算子。结合IR提出可证伪修改，'
            '新候选须重新采集并完成KGS正式评测。\n\n'
            + json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    context_extension = {}
    if native_context:
        helper = ROOT / 'native_context.py'
        helper_target = target / 'tools/hgkda_native_context.py'
        shutil.copyfile(helper, helper_target)
        profile_target = target / 'tools/profile_round.py'
        patch = '''

# HG-KDA 可选扩展：只补充冻结 Native 输入，不改变测量、合同或采集参数。
_hgkda_base_get_profile_context = get_profile_context

def get_profile_context(workspace, context, round_num):
    from kernelgen.tools.hgkda_native_context import enrich_native_context
    authoritative = _hgkda_base_get_profile_context(workspace, context, round_num)
    return enrich_native_context(authoritative, _load_snapshot(Path(workspace).resolve(), round_num))
'''
        original = profile_target.read_bytes()
        profile_target.write_bytes(original + patch.encode('utf-8'))
        context_extension = {
            'helper_sha256': hashlib.sha256(helper_target.read_bytes()).hexdigest(),
            'base_profile_round_sha256': hashlib.sha256(original).hexdigest(),
            'overlay_profile_round_sha256': hashlib.sha256(profile_target.read_bytes()).hexdigest(),
        }
        appended += ('\n\n## Native 用例的准确输入\n\n'
                     '`get_profile_context` 额外返回 `hgkda_workloads`，其中按用例身份保存原始'
                     '`inputs`、seed、tolerance及phase。先从这里读取形状、dtype、stride相关规格'
                     '与标量参数；不要根据case序号或其他算子的历史形状猜测。'
                     '它不改变正式延迟、指标、候选源码或计分合同。\n')
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
    if native_context:
        data['native_context_extension'] = context_extension
    if paired_extension:
        data['paired_evidence_extension'] = paired_extension
    if roofline_extension:
        data['roofline_evidence_extension'] = roofline_extension
    (output / 'hgkda-overlay.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-context', action='store_true',
                   help='仅新覆盖层：从冻结Native快照补充准确输入规格；不改变既有批次')
    p.add_argument('--paired-evidence', type=Path,
                   help='本批seed成对采集的结构化摘要；只注入新覆盖层，保留来源身份')
    p.add_argument('--roofline-evidence', type=Path,
                   help='本批已校验的Roofline紧凑摘要；保留算法量、计数和参考来源')
    a = p.parse_args()
    print(json.dumps(build(a.output, native_context=a.native_context,
                           paired_evidence=a.paired_evidence, roofline_evidence=a.roofline_evidence), ensure_ascii=False))


if __name__ == '__main__':
    main()
