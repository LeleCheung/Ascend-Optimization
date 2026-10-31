"""从终态核验与原始KGS结果生成中文报告；不改候选、合同或选版。"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
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
    stage = read(batch / 'stage-state.json')
    scope_root=batch/'timing-scope-evidence';scope=read(scope_root/'scope.json')
    if scope['actual_scope']!='device_kernel' or scope['native_scope']!='walltime':raise ValueError('计时解释不符合本批源码核对')
    for item in scope['files'].values():
        if hashlib.sha256((scope_root/item['snapshot']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('计时依据源码快照身份不同')
    if hashlib.sha256((scope_root/'backend-testing.py').read_bytes()).hexdigest()!=scope['backend']['sha256']:raise ValueError('实际计时后端快照身份不同')
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
            try:
                base = comparison.load_result(base_path)
                base['timing_scope']='device_kernel'
            except ValueError: pass
        for item in state.get('legacy_contracts', []):
            if item['operator'] != op: continue
            row = dict(item)
            if item['status'] == 'passed':
                result = comparison.load_result(root / 'original-contract' / item['version'] / 'original-1.result.json')
                result['timing_scope']='device_kernel'
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
    lines += ['', '### 原项目FlagGems合同：设备计时', '', '| 算子 | 版本 | 状态 | 完整项数 | 相对PyTorch设备计时 | 相对本批父版设备计时 |', '|---|---|---|---:|---:|---:|']
    for row in original:
        lines.append(f'| `{row["operator"]}` | {names.get(row["version"], "历史优化父版＋共享适配")} | {row["status"]} | {row.get("workloads", "—")} | '
                     f'{fmt(row["relative_pytorch"]) if "relative_pytorch" in row else "—"} | {fmt(row["relative_parent"]) if "relative_parent" in row else "—"} |')
    lines += ['', '原合同使用FlagGems默认KERNEL模式，其do_bench_npu通过MSPTI或torch_npu.profiler记录设备任务。'
        'evaluate-operator.py的--timing-scope只记录标签，不改变benchmark；本批原合同provenance误记walltime，原字节保留，'
        '这里根据timing-scope-evidence更正为device_kernel。Native完整调用与该表不混用。原合同失败保留错误，不用局部通过性能代替完整结果。', '',
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
    def duration(record):
        start=datetime.datetime.fromisoformat(record['started_at'].replace('Z','+00:00'))
        end=datetime.datetime.fromisoformat(record['finished_at'].replace('Z','+00:00'))
        return (end-start).total_seconds()
    lines += ['## 五、共享成本与版本修复', '',
        f'本批共享准备、父版语义、完整起点和干预阶段耗时{duration(stage):.1f}秒；'
        f'冻结方法后的生成、评测、采集与独立检验阶段耗时{duration(state):.1f}秒。'
        '这些阶段成本不能只用模型推理时间代替；模型usage与时长保留于结果JSON的generation字段。', '',
        'v8两个归约候选均在NaN补充检查中为9/18，未交付。固定Triton的minimum与min默认忽略NaN；'
        '本批两组共享显式NaN传播minimum和reduce组合器。显式reduce同时去掉标准tl.min对低精度输入的隐式扩展，'
        '这些共享修复不计为模型配置搜索收益。v9/v10新增预检暴露工具导入冲突，均在算子提交前停止，原回执保留。', '',
        '本批父版先通过18项补充语义再进入完整评测；候选语义不通过则不继续设备采集。'
        '中间轴FP16原生实际BR为配置middle_br两倍，FP32/BF16为配置值；默认组织与父版实际参数对齐。'
        '模型输入包含最终wrapper模板、编译绑定常量、历史失败的具体语义用例及设备确认，避免只根据父版代码猜测候选路径。', '',
        '统一层级证据由evidence_layers.py在采集后只读导出，不参与本批冻结的诊断或选版。'
        'RTL条目固定为not_run并列出适配器所需设计版本、时钟、存储模型、激励和波形身份；没有RTL运行成果。', '']
    lines += ['## 六、代码机制的直接核验', '',
        '以下按实际捕获的编译常量和grid核对配置，程序数是Triton逻辑program数；两份捕获路径均只有一次kernel提交。'
        'program数减少不等于kernel launch次数减少。', '',
        '| amin类型 | BO | BR | grid中的program数 | 每个program归约循环次数 | 设备确认 |',
        '|---|---|---|---|---|---:|']
    chosen=state['selected']['amin/mechanism']
    gate=read((batch/chosen['source']).parent/'supplementary/check.json')
    for case in gate['confirmations']:
        if len(case['shape'])!=2: continue
        parent=case['parent_ir'][0];candidate=case['candidate_ir'][0]
        a=parent['constants'];b=candidate['constants']
        lines.append(f'| {case["dtype"]} | {a["BO"]}→{b["BO"]} | {a["BR"]}→{b["BR"]} | '
                     f'{math.prod(parent["grid"])}→{math.prod(candidate["grid"])} | '
                     f'{math.ceil(a["R"]/a["BR"])}→{math.ceil(b["R"]/b["BR"])} | {fmt(case["paired_speedup"])} |')
    lines += ['', '输入元素总数不变；该配置同时改变每个program处理的行数、循环次数和末次横向归约跨度，'
        '因此已确认的是整体归约组织的设备收益。仅凭这一组联动变换，不能把全部收益唯一归因于program调度或MTE2。', '',
        '### 成对资源计数的输入核对', '']
    for rec in plan['records']:
        op=rec['operator'];root=batch/'results'/op/'development'
        a=read(root/'parent-metrics/facts.json');b=read(root/'winner-metrics/facts.json')
        match=a['case_id']==b['case_id']
        lines += [f'- `{op}`：父版`{a["case_id"]}`，候选`{b["case_id"]}`；'+('同case。' if match else 'case不同，不相除或推断计数变化。')]
    lines += ['', '模型关于MTE2、带宽或等待的原文是待验证假设。设备资源指标属于所绑定的case；'
        '3D最慢case的计数不能代替2D归约的资源证据，逐block计数也不能直接当作全卡HBM峰值利用率。'
        '本轮用实际grid、IR、设备配对和完整评测核验了代码变更与性能效应；未建立设备完整因果依赖图。', '']
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
