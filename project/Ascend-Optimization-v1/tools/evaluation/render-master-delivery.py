#!/usr/bin/env python3
"""从已审计的四版本证据生成精简中文汇报，不手填性能数值。"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    spec = importlib.util.spec_from_file_location('summary', Path(__file__).with_name('summarize-master-goal.py'))
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    data = summary.audit(project)
    selected_narrow = json.loads((project / 'operators/narrow_copy/reports/master-closure-20261010/最终候选.json').read_text(encoding='utf-8'))
    selected_matmul = json.loads((project / 'operators/matmul_bias_activation/reports/master-closure-20261010/最终候选.json').read_text(encoding='utf-8'))
    narrow_result = selected_narrow['representative_result']
    if 'profile-small-dma' in narrow_result:
        narrow_method = '小连续片段走 CANN DMA，大输入采用少量 Triton program 循环复制，消除逐元素动态除法。'
    elif 'contiguous-dma-grid' in narrow_result:
        narrow_method = '单段连续 dim=0 切片直接使用 CANN DMA，其余由简化索引后的 Triton 内核复制。'
    elif 'small-dma-grid' in narrow_result:
        narrow_method = '小连续片段走 CANN DMA，大输入使用连续索引与更大分块的 Triton 复制。'
    else:
        narrow_method = '简化连续切片的索引并扩大分块，使用纯 Triton 复制。'
    methods = {
        'amin': '直接沿原布局归约，按实测调整分块；float16 分支采用原精度 min，bf16 保留 float32 累积，减少转置和多余类型扩展。',
        'matmul_bias_activation': 'K 分块从 32 扩大到 256，调整多缓冲与分组调度。' +
            ('规则半精度矩阵复用同一 A 面板计算两个相邻输出 tile。' if selected_matmul['version'] == 'v8' else ''),
        'narrow_copy': narrow_method,
    }
    lines = ['# 本阶段精简汇报', '',
             '在 Ascend 910B 上完成三个 Excel 华为列低于 0.8× 算子的四版本对照：固定 FlagGems master、KG 无 profiler、KG 原生 profiler、分析指导优化版。', '',
             '| 算子 | 固定 master | KG 无 profiler | KG 原生 profiler | 我们的优化版 | 相对 master | 完整测试 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    passed = 0
    for op in data['operators']:
        rows = op['versions']
        final = rows[-1]
        passed += final['relative_pytorch'] > 0.8
        numbers = ' | '.join(f"{row['relative_pytorch']:.3f}×" for row in rows)
        lines.append(f"| {op['operator']} | {numbers} | {final['relative_master']:.2f}× | {op['total_cases']}/{op['total_cases']} |")
        status = '超过 0.8× 目标' if final['relative_pytorch'] > 0.8 else '仍低于 0.8× 目标'
        report = [f"# {op['operator']} 精简汇报", '',
                  f"四版本完整通过 **{op['total_cases']}/{op['total_cases']}**。相对 PyTorch 依次为 " +
                  '、'.join(f"**{row['relative_pytorch']:.3f}×**" for row in rows) + '。', '',
                  methods[op['operator']] + f"相对固定 master 提高 **{final['relative_master']:.2f}×**，{status}。", '',
                  '性能是同合同设备计时下逐 case 加速比的几何平均。完整源码、逐 case 数据与 SHA 见 [版本对比](版本对比.md)。', '']
        if op['operator'] == 'narrow_copy':
            report.insert(-1, '最终实现类型：' + selected_narrow['implementation_kind'] + '；设备任务计时包括 DMA 与 AIV kernel，输出重新分配并实际复制。')
        (project / 'operators' / op['operator'] / 'reports/master-closure-20261010/精简汇报.md').write_text('\n'.join(report), encoding='utf-8')
    lines += ['', f'三个优化版均完整通过测试；其中 {passed} 个超过 0.8×。超过 1× 才表示快于 PyTorch。', '']
    lines += [f"- **{name}**：{method}" for name, method in methods.items()]
    lines += ['', 'narrow_copy 最终实现类型：' + selected_narrow['implementation_kind'] + '。amin 和矩阵乘采用设备 kernel 计时；narrow_copy 包括 DMA 与 AIV kernel，均不含 host 发射间隙。', '',
              '独立 profiling workflow 已实现源码、case、合同与原始附件校验，并按归约、矩阵乘和复制语义输出诊断。实际优化使用了真机指标、设备耗时和编译配置分析；数值 Roofline 尚未建立。', '',
              'amin 与 narrow_copy 的 master 列包含启动兼容修复，保留原计算内核；Excel 历史数据不与本轮结果混算。', '',
              '复现见 [交接步骤](../tools/evaluation/复现与交接-20261010.md)，逐 case 与源码 SHA 见 [完整四版本表](低于0.8算子四版本闭环-20261010.md)。', '']
    output = project / 'operators/本阶段精简汇报-20261010.md'
    output.write_text('\n'.join(lines), encoding='utf-8')
    print('DELIVERY_REPORT_GENERATED', output)


if __name__ == '__main__':
    main()
