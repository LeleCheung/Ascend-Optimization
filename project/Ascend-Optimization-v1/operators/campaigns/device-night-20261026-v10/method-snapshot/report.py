"""从终态核验与原始KGS结果生成中文报告；不改候选、合同或选版。"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('device_report_audit', HERE / 'audit.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
comparison = module.comparison


def read(path): return json.loads(path.read_text(encoding='utf8'))
def fmt(value): return f'{value:.3f}×'


def report(batch, audit_path, artifact_path, output):
    audit = read(audit_path); artifacts = read(artifact_path)
    if audit['status'] != 'passed' or artifacts['status'] != 'passed':
        raise ValueError('两份证据核验必须均通过')
    state = read(batch / 'state.json'); plan = read(batch / 'plan.json')
    if state['status'] != 'completed': raise ValueError('系统批次未完成')
    output.mkdir(parents=True, exist_ok=True)
    if (output / '报告.md').exists(): raise ValueError('不覆盖已交付报告')
    names = dict(raw_evidence='原始证据模型组', mechanism='机制诊断模型组')
    native = []; original = []; per_case = []
    for row in audit['results']:
        native.append(row)
    for record in plan['records']:
        op = record['operator']; root = batch / 'results' / op
        for split in ['development', 'heldout']:
            parent = comparison.load_result(root / split / 'parent/parent-1.result.json')
            for arm in plan['arms']:
                chosen = state['selected'][op + '/' + arm]
                path = (batch / chosen['source']).parent / 'formal/candidate-1.result.json' if split == 'development' else root / split / arm / (arm + '-1.result.json')
                result = comparison.load_result(path)
                for case, row in result['timing'].items():
                    baseline = parent['timing'][case]
                    per_case.append(dict(operator=op, split=split, arm=arm, case_id=case, axes=json.dumps(row.get('axes', {}), ensure_ascii=False),
                        parent_us=baseline['latency_ms'] * 1000, candidate_us=row['latency_ms'] * 1000,
                        pytorch_us=row['reference_latency_ms'] * 1000,
                        relative_parent=baseline['latency_ms'] / row['latency_ms'],
                        relative_pytorch=row['reference_latency_ms'] / row['latency_ms'], scope=result['timing_scope']))
        base_path = root / 'original-contract/parent/original-1.result.json'
        base = None
        if base_path.exists():
            try: base = comparison.load_result(base_path)
            except ValueError: pass
        for item in state.get('legacy_contracts', []):
            if item['operator'] != op: continue
            row = dict(item)
            if item['status'] == 'passed':
                result = comparison.load_result(root / 'original-contract' / item['version'] / 'original-1.result.json')
                row.update(relative_pytorch=statistics.geometric_mean(r['reference_latency_ms'] / r['latency_ms'] for r in result['timing'].values()),
                           timing_scope=result['timing_scope'])
                if base is not None: row['relative_parent'] = comparison.compare([('parent', base), (item['version'], result)])[1]['relative_master']
            original.append(row)
    lines = ['# HG-KDA设备机制系统闭环', '',
        '本轮把设备干预、竞争机制假设、分层补测和受约束模型配置接入同一流程，检验矩阵乘与归约两类设备机制。'
        '父版是历史KG/HG-KDA、CC/DeepSeek与Codex共同优化源码；矩阵乘新增两组共享的布局及小K兼容修复。以下相对父版收益不等于相对原版FlagGems。', '',
        '## 一、结果', '',
        '| 算子 | 组别 | 开发集完整调用：相对父版 | 独立集完整调用：相对父版 | 开发集设备kernel：相对父版 |',
        '|---|---|---:|---:|---:|']
    for rec in plan['records']:
        op = rec['operator']
        for arm in plan['arms']:
            development = next(r for r in native if r['operator'] == op and r['arm'] == arm and r['split'] == 'development')
            heldout = next(r for r in native if r['operator'] == op and r['arm'] == arm and r['split'] == 'heldout')
            device = state['selected'][op + '/' + arm]['device_geomean']
            lines.append(f'| `{op}` | {names[arm]} | {fmt(development["relative_parent"])} | {fmt(heldout["relative_parent"])} | {fmt(device)} |')
    lines += ['', '每个Native合同包含18项正确性与6项性能。相对父版由同case实际延迟相除后取几何平均；完整调用包含host与device。'
        '设备列来自4个开发输入、各5轮同输入配对，排除host发射间隙；独立集只检验完整调用，没有将开发设备收益冒充独立设备收益。', '',
        '### 设备确认逐形状', '', '| 算子 | 组别 | 形状 | 类型 | 相对父版设备kernel |', '|---|---|---|---|---:|']
    for row in audit['predictions']:
        lines.append(f'| `{row["operator"]}` | {names[row["arm"]]} | {row["shape"]} | {row["dtype"]} | {fmt(row["device_speedup"])} |')
    lines += ['', '### 原项目FlagGems合同', '', '| 算子 | 版本 | 状态 | 完整项数 | 相对PyTorch完整调用 | 相对本批父版 |', '|---|---|---|---:|---:|---:|']
    for row in original:
        lines.append(f'| `{row["operator"]}` | {names.get(row["version"], "历史优化父版＋共享适配")} | {row["status"]} | {row.get("workloads", "—")} | '
                     f'{fmt(row["relative_pytorch"]) if "relative_pytorch" in row else "—"} | {fmt(row["relative_parent"]) if "relative_parent" in row else "—"} |')
    lines += ['', '原合同与本轮Native合同的形状、参考和计时方法不同，分表报告。原合同失败保留错误，不用局部通过性能代替完整结果。', '',
        '## 二、方法和证据', '',
        '```mermaid', 'flowchart LR',
        '    A[源码与输入快照] --> B[同输入设备干预]',
        '    B --> C[竞争机制假设]',
        '    C --> D{最小必要补测}',
        '    D --> E[真机：时间与资源现象]',
        '    D --> F[IR：类型、缓冲、搬运与同步]',
        '    D --> G[模拟器：小范围指令与等待]',
        '    E --> H[同预算模型生成配置]', '    F --> H', '    G --> H',
        '    H --> I[预测确认＋完整KGS]', '    I --> J[冻结选版]', '    J --> K[独立形状检验]',
        '    L[RTL证据接口：预留] -.-> D', '```', '',
        'Codex实现干预空间、测量器、竞争假设、补测规则、模板与兼容修复；CC＋DeepSeek选择配置并输出预测。'
        '两组均看到相同原始干预、实际IR摘要和新增采集；机制组额外得到结构化诊断。每组每算子两次候选机会，最多一次格式修正，失败占用机会。'
        '它是有限空间配置生成实验，不是模型自由合成任意kernel的对照，也不是纯原生KG对照。', '',
        '| 算子 | 采集 | 真机/KGS终态 | OpProf kernel数 | 模拟器core数 | 时间（秒） |', '|---|---|---|---:|---:|---:|']
    for row in audit['profiles']:
        lines.append(f'| `{row["operator"]}` | {row["name"]} | {row["status"]} | {row["opprof_kernels"]} | {row["simulator_cores"]} | {row["seconds"] or 0:.1f} |')
    for rec in plan['records']:
        op = rec['operator']; development = batch / 'results' / op / 'development'
        decision = read(development / 'acquisition-decision.json')
        diagnosis = read(development / 'diagnosis.json')
        lines += ['', f'### `{op}`', '', f'补测选择：`{decision["next_probe"]}`。{decision["reason"]}。', '',
                  '干预方向按相同调度重表达对照比较；变化超过测量范围且最坏收益>5%才标为待确认方向。最终候选再次与父版配对。']
        supported = [(c['id'], o['variant'], o['reason']) for c in diagnosis['cases'] for o in c.get('observations', []) if o['sample_ranges_separated']]
        lines += [''] + ([f'- case {case}／`{variant}`：{reason}。' for case, variant, reason in supported] or ['没有干预达到预设的明确方向门槛；仍保留原始测量与失败。'])
        chosen = state['selected'][op + '/mechanism']
        proposal = read((batch / chosen['source']).parent / 'proposal.json')
        lines += ['', '机制组胜出配置：`' + json.dumps(chosen['config'], ensure_ascii=False) + '`。', '',
                  '模型假设：' + str(proposal.get('hypothesis', '未提供')) + '。', '',
                  '模型预测原文：`' + json.dumps(proposal.get('predictions', []), ensure_ascii=False) + '`。']
    lines += ['', 'OpProf流水活跃比例可以重叠，不能加和成时间分解。静态IR匹配不是动态指令计数，逻辑请求量不等于实测HBM流量。'
        '编译metadata中的shared/workspace字段不直接当作UB独占峰值。模拟器只有成功返回匹配目标kernel的指令附件才算采集成功；失败和超时单列。'
        '本轮尚无定量Roofline、完整指令依赖图或RTL执行器。', '', '## 三、成本与失败', '',
        '| 算子 | 组别 | 候选 | 状态 | 总时间（秒） |', '|---|---|---:|---|---:|']
    for row in state['records']:
        lines.append(f'| `{row["operator"]}` | {names[row["arm"]]} | {row["index"]} | {row["status"]} | {row["seconds"]:.1f} |')
        if row['status'] != 'passed': lines += ['', '失败：`' + row.get('error', '见原始回执').replace('\n', ' ') + '`。', '']
    lines += ['', '前置失败批次device-night-20261026至v7完整保留：ABI、历史小K流水超时、采集文件遮蔽stdlib、精度合同、带步长UB溢出与Definition可选参数问题。'
        'v8起点修复属于系统建立成本，不从总投入中删除。超时kernel日志只能定位失败路径，尚不能证明精确的硬件死锁原因。', '',
        '## 四、核验与复现', '',
        f'方法/源码/模型/评测核验{audit["assertions"]}项；附件及CSV核验{artifacts["assertions"]}项，覆盖'
        f'{artifacts["unique_artifacts"]}份唯一附件和{artifacts["device_measurements"]}次设备配对采集。完整原件在批次raw-evidence归档。', '',
        '方法在system-snapshot/冻结；selection-frozen.json先于独立集评测写入。源码、配置、提示词、模型回执和失败均随批次保留。'
        '独立集固定开发外形状，未根据其结果继续调参；这仍是两个算子的形状迁移检验，尚不构成跨算子或跨芯片泛化结论。', '',
        '在910B宿主运行以下命令，可在新目录完整复现冻结交付，不重新调用模型：', '', '```bash',
        'cd /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1',
        '/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python3 \\',
        f'  tools/profiling/hgkda/device_mechanism/reproduce.py --batch {batch.name} \\',
        '  --operator amin --arm mechanism --split heldout \\',
        '  --output /data/hanle/ascend-optimization/runs/device-mechanism-reproduction-新目录', '```', '',
        '使用项目物理卡7互斥锁及既有KGS19659；替换operator可复现矩阵乘。复现脚本做18项语义检查、开发设备确认及指定合同的一次完整评测。'
        '本轮未执行额外重复的正式复测，也未提交或推送GitHub。', '']
    (output / '报告.md').write_text('\n'.join(lines), encoding='utf8')
    result = dict(batch=plan['batch'], native=native, original_contracts=original, device=audit['predictions'], generation=audit['generation'], profiles=audit['profiles'],
                  audit_assertions=audit['assertions'], artifact_assertions=artifacts['assertions'])
    (output / '结果.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    with (output / '逐case.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_case[0])); writer.writeheader(); writer.writerows(per_case)
    print(json.dumps(dict(report=str(output / '报告.md'), cases=len(per_case)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--artifacts-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); report(args.batch_dir, args.audit, args.artifacts_audit, args.output)
