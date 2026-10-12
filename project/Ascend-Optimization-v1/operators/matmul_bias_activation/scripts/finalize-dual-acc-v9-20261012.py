#!/usr/bin/env python3
"""核验完整合同、两次复验与三组原生源码，生成中文交付材料。"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    operator = Path(__file__).resolve().parents[1]
    project = operator.parents[1]
    tool = project / 'tools/evaluation/compare-operator-versions.py'
    spec = importlib.util.spec_from_file_location('compare', tool)
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    reports = operator / 'reports'
    final = reports / 'master-closure-20261012'
    v9 = reports / 'profiling-dual-acc-v9-20261012'
    repeated = [compare.load_result(v9 / f'optimized-v9-{i}.result.json') for i in (1, 2)]
    repeated_summary = compare.compare([('v9-1', repeated[0]), ('v9-2', repeated[1])])
    candidate = operator / 'candidates/profiling-dual-acc-v9-20261012.py'
    source = candidate.read_text(encoding='utf-8')
    for result, row in zip(repeated, repeated_summary):
        if len(result['rows']) != 42 or result['correctness_count'] != 27 or len(result['timing']) != 15:
            raise ValueError('未完整覆盖 27 项正确性和 15 项性能')
        if row['relative_pytorch'] <= 0.8:
            raise ValueError('两次复验必须分别超过 0.8 倍')
        request = json.loads(Path(result['path']).with_name(
            Path(result['path']).name.replace('.result.json', '.request.json')).read_text(encoding='utf-8'))
        if request['implementation']['sources'][0]['content'] != source:
            raise ValueError('最终源码与复验源码不一致')
    selected = min(repeated_summary, key=lambda row: row['relative_pytorch'])
    paths = [
        ('master', final / 'baselines/master/master-1.result.json'),
        ('kg-no-profile', final / 'baselines/kg-no-profile/kg-no-profile-1.result.json'),
        ('kg-profile', final / 'baselines/kg-profile/kg-profile-1.result.json'),
        ('optimized-v9', Path(selected['result_path'])),
    ]
    historical = [
        reports / 'master-20261010/flaggems-master-1.result.json',
        reports / 'kg-no-profile-master-20261010/independent/kg-no-profile-1.result.json',
        reports / 'kg-profile-master-20261010/independent/kg-profile-1.result.json',
    ]
    for (_, current_path), old_path in zip(paths, historical):
        current, old = compare.load_result(current_path), compare.load_result(old_path)
        if current['source_sha256'] != old['source_sha256']:
            raise ValueError('对照源码发生变化')
        if current['rows'].keys() != old['rows'].keys():
            raise ValueError('历史与当前完整用例集合不同')
        for case, row in current['rows'].items():
            if row['axes'] != old['rows'][case]['axes'] or row['phase'] != old['rows'][case]['phase']:
                raise ValueError('历史用例语义发生变化')
    command = [sys.executable, str(tool)]
    for label, path in paths:
        command += ['--version', label, str(path)]
    command += ['--output', str(final / '版本对比.md')]
    subprocess.run(command, check=True)
    for suffix in ('.md', '.json'):
        generated = final / ('版本对比' + suffix)
        generated.write_bytes(generated.read_bytes().replace(b'\r\n', b'\n'))
    summary = json.loads((final / '版本对比.json').read_text(encoding='utf-8'))
    summary_tool = project / 'tools/evaluation/summarize-master-goal.py'
    summary_spec = importlib.util.spec_from_file_location('summary', summary_tool)
    summary_module = importlib.util.module_from_spec(summary_spec)
    summary_spec.loader.exec_module(summary_module)
    all_operators = summary_module.audit(project)
    for item in all_operators['operators']:
        if item['operator'] == 'matmul_bias_activation':
            item['versions'] = [dict(row) for row in summary]
            for row in item['versions']:
                row['result_path'] = Path(row['result_path']).resolve().relative_to(project.resolve()).as_posix()
    combined = project / 'operators/低于0.8算子四版本闭环-20261012.md'
    combined.write_text(summary_module.render(all_operators), encoding='utf-8', newline='\n')
    combined.with_suffix('.json').write_text(json.dumps(all_operators, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    optimized = summary[-1]
    evidence = {
        'schema_version': 1,
        'operator': 'matmul_bias_activation',
        'upstream_commit': 'd6a8eec473517a3d68157b208eb9c057eb1d4c50',
        'threshold': 0.8,
        'candidate': candidate.relative_to(project).as_posix(),
        'source_sha256': optimized['source_sha256'],
        'repeats': repeated_summary,
        'selected_result': Path(selected['result_path']).relative_to(project).as_posix(),
        'selected_relative_pytorch': optimized['relative_pytorch'],
        'selected_relative_master': optimized['relative_master'],
        'benchmark_fingerprint': repeated[0]['fingerprint'],
        'historical_baseline_source_sha_verified': True,
        'correctness_count': 27,
        'timing_count': 15,
        'timing_scope': 'device_kernel',
    }
    (final / '最终候选.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    ratios = [row['relative_pytorch'] for row in summary]
    lines = [
        '# matmul_bias_activation 精简汇报', '',
        f'910B 上完成新的 v9 优化：两次完整复验分别为 **{repeated_summary[0]["relative_pytorch"]:.3f}×、{repeated_summary[1]["relative_pytorch"]:.3f}×**（相对 PyTorch），均 **42/42 全通过**，达到大于 0.8× 的目标。', '',
        f'同合同四版本：固定 FlagGems master **{ratios[0]:.3f}×**，KG 无 profiler **{ratios[1]:.3f}×**，KG 原生 profiler **{ratios[2]:.3f}×**，我们的 v9 **{ratios[3]:.3f}×**；最终相对 master **{optimized["relative_master"]:.2f}×**。', '',
        '关键改进：在 v8 双 N tile 复用 A 数据的基础上，使用点积累加器，并启用 Ascend 多缓冲和流水调度，让数据搬入与计算重叠。FP32 和不规则输入保留原来的稳定路径。', '',
        '诊断探针中，半精度 2048/4096 方阵的新流水配置比无流水配置快约 27%～32%，输出与 v8 完全一致。完整评测以 15 项性能用例的加速比几何平均计分，另有 27 项正确性门禁；整体仍低于 PyTorch，但已经超过验收线。', '',
        '评测环境：Ascend 910B4-1 物理卡 7，CANN 9.0.0，Triton 3.5.1/Flagtree 0.6.0，KGS 6.5.0/19655，设备 kernel 计时。三组对照沿用原有 KG/FlagGems 源码，在同一仓库 HEAD 与 fingerprint 下重新评测；原历史结果保留。', '',
        '证据：[逐 case 四版本对比](版本对比.csv)、[两次复验与源码哈希](最终候选.json)、[v9 源码](../../candidates/profiling-dual-acc-v9-20261012.py)、[配置探针](../dual-acc-probe-20261012/artifacts/dual-acc.json)。', '',
        '复现：在 910B 宿主机的 tmux 中执行 `bash project/Ascend-Optimization-v1/operators/matmul_bias_activation/scripts/run-dual-acc-v9-20261012.sh`。默认目录用于恢复现有结果；重新测量时将脚本中的输出目录换成新批次，再运行本汇总脚本。', '',
    ]
    (final / '精简汇报.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    main()
