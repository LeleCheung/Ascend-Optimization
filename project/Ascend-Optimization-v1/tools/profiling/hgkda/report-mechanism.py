"""从已核验的开发与独立批次生成中文机制报告及逐case证据，不调用设备或模型。"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mechanism_report_compare',HERE.parents[1]/'evaluation/compare-operator-versions.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


def read(path):return json.loads(path.read_text(encoding='utf8'))


def report(development,heldout,output):
    if output.exists():raise ValueError('保留已有报告，请使用新输出')
    plans=[read(p/'plan.json') for p in [development,heldout]]
    for name in ['core.py','probe.py','check.py','rewrite-dispatch.py']:
        if plans[0]['method_sha256'][name]!=plans[1]['method_sha256'][name]:
            raise ValueError('开发与独立检验的核心方法不同：'+name)
    rows=[];paths=[];costs=[];csv_rows=[];audit_count=0
    for batch,role in [(development,'开发'),(heldout,'冻结后独立检验')]:
        audit=read(batch/'evidence-audit.json')
        if audit['status']!='passed':raise ValueError('证据未核验')
        audit_count+=audit['assertions']
        for rec in audit['records']:
            if rec.get('status')!='passed':continue
            op=rec['operator'];arm=rec['arm'];root=batch/'results'/op
            baseline=comparison.load_result(root/'FlagGems/baseline-1.result.json')
            current=comparison.load_result(root/arm/'formal'/(arm+'-1.result.json'))
            values=comparison.compare([('FlagGems＋共享适配',baseline),(arm,current)])[1]
            label='原始证据自由生成' if arm=='raw_evidence' else '机制诊断＋受约束变换'
            rows.append(f'| `{op}` | {role} | {label} | {values["relative_master"]:.3f}× | '
                        f'{values["relative_pytorch"]:.3f}× | 51/51＋21/21 |')
            usage=rec.get('model_usage') or {}
            costs.append(f'| `{op}` | {label} | {usage.get("input_tokens","未提供")} | '
                         f'{usage.get("output_tokens","未提供")} | {rec["model_seconds"]:.2f} | {rec["total_arm_seconds"]:.2f} |')
            if arm=='mechanism_rewrite':
                semantics=read(root/arm/'semantics.json')
                for item in semantics['path_interventions']:
                    times=item['timings'];paired=statistics.median([a/b for a,b in zip(times['seed_us'],times['candidate_us'])])
                    paths.append(f'| `{op}` | {item["shape"][0]} | {item["seed_warm_jit_calls"]}→{item["candidate_warm_jit_calls"]} | '
                                 f'{statistics.median(times["seed_us"]):.2f}→{statistics.median(times["candidate_us"]):.2f} | {paired:.3f}× |')
            for key,value in current['timing'].items():
                parent=baseline['timing'][key]
                csv_rows.append([op,role,arm,key,json.dumps(value.get('axes'),ensure_ascii=False),
                    parent['latency_ms']*1000,value['latency_ms']*1000,value['reference_latency_ms']*1000,
                    parent['latency_ms']/value['latency_ms'],value['reference_latency_ms']/value['latency_ms'],
                    value['reference_latency_ms']/parent['reference_latency_ms'],current['source_sha256']])
    if len(rows)!=4:raise ValueError('开发与独立检验的两组结果尚不完整')
    text='''# HG-KDA 机制闭环原型：从观察瓶颈到验证改法

本轮交付了一种可复现的单机制优化流程：**捕获真实调用 → 定向干预区分原因 → 选择受约束变换 → 完整评测 → 核验预测变化**。同一份通用变换器在开发算子 `selu_` 和冻结后的独立算子 `asinh_` 上生效，数学内核保持原样。

## 本轮结果

以下从固定 FlagGems 数学源码＋两组共享正确性适配出发。相对 FlagGems 的列逐case计算“本批起点实际延迟÷候选实际延迟”后取几何平均；不是两组相对PyTorch分数相除。采用 KGS 批量调用前后同步的完整调用 walltime，一次正式评测。

| 算子 | 用途 | 方法 | 相对本批FlagGems＋共享适配 | 相对PyTorch | 完整KGS＋补充检查 |
|---|---|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

这些成绩验证调用机制改写，不替换历史优化版，也不表示新算子已超过0.8×验收线。共享适配保留原数学计算，对非连续布局执行连续临时量后回写，独立批次另显式处理空输入；两组起点完全相同。逐case延迟、参考漂移和源码身份见同名CSV。

## 已实现的机制闭环

```mermaid
flowchart LR
    A[源码与输入绑定] --> B[捕获 JIT、grid、编译产物]
    B --> C[同产物普通调用 vs 直接提交]
    C --> D[区分分派成本与设备组织假设]
    D --> E[模型选择有证据支持的动作]
    E --> F[通用变换器保留数学内核]
    F --> G[完整 KGS 与语义门禁]
    G --> H[核验 JIT 次数和延迟预测]
```

1. **绑定证据。** 保存输入形状、dtype、设备、源文件SHA、运行时SHA、编译产物摘要和实际grid。调用图只表示wrapper→JIT→runtime提交→设备→完成的语义链。
2. **区分原因。** 在同输入、同编译对象和grid上交错比较普通调用与直接提交。所有配对比例大于1.15时支持分派假设；具备明确长度与分块接口时，再用相同JIT入口改变block/grid。1.15与1.10是预登记经验门槛，非统计置信度。
3. **编译动作。** CC＋DeepSeek选择缓存分派或保留；通用Python模板缓存编译对象和参数模板，逐次重新绑定当前输入、指针和stream。模板没有算子数学公式，首调用执行一次后直接返回，未知布局回原实现。
4. **验证预测。** 不仅检查数值和速度，还核对暖态JIT分派是否减少。当前交付只应用分派变换，未同时改变数学内核和设备分块。

| 算子 | FP32元素数 | 暖态JIT调用数 | 同输入中位延迟，微秒 | 配对加速比中位数 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(paths)+'''

上述延迟为9组交错观测的同步walltime，包含调用与设备执行；不是单个流水的独占时间。补充检查覆盖三种dtype、特殊值、空输入、偏移、stride、新地址与两个stream，正式合同另有42项正确性和9项性能。原始trace保存于各批次归档；采集出现level0不含aic metrics及schedule警告，不能据此声称本轮已得到PipeUtilization或指令级因果图。

## 冻结后独立检验与对照

开发阶段使用SELU；先冻结规则、探针、检查器和变换器，再执行独立算子及不同形状。`asinh_` 按明确的一维JIT接口选择，选择时查看了接口源码，没有加入其专用公式改写或专用优化建议；不是按测得加速比挑选。两批上述四个核心文件SHA一致，变换器为 `ad9406ce8e7b14f523d10719d4920c87a0ba007cb6277968245e8697a0b592da`。

原始证据组与机制组共享起点、接口知识、模型、原始探针和正式合同，各一次候选机会，最多一次格式修正；开发算子先自由生成，独立算子先受约束变换。前者自由输出完整源码，后者输出动作JSON再由模板改写。因此本轮比较的是“机制诊断＋受约束变换”的组合效果，不能单独归因执行图，也不是纯原生KG组。一次候选检验用于证明可运行和迁移，不给出统计显著结论。

Codex实现并审查工具、制定合同与接口知识、编排实验和核验证据；CC＋DeepSeek完成自由源码生成或动作选择。最终受约束候选可从起点和冻结模板逐字节重建，没有手改已生成候选。

## 失败与适用范围

| 对象或批次 | 真实结果 | 处理 |
|---|---|---|
| 首次pilot | 公共别名冲突，0/51，未调用模型 | 独立新版本修正ABI，原证据保留 |
| pilot-v2机制自由生成 | 首次正常调用后又提交一次，补充仅9/21 | 拒绝正式计分；新增通用变换器，未手修失败候选 |
| sigmoid_独立尝试 | 原起点48/51，二维非连续布局错误 | 停在起点评测；后续共享布局适配单独登记 |
| tanh_独立尝试 | 起点51/51，暖态探针0/4捕获JIT | 保留unsupported，不生成候选；另设asinh_批次 |
| compiler初稿 | 预审发现拟修改全局JIT类 | 未执行；最终只临时代理候选自身JIT实例并恢复 |

本轮最重要的系统进展是：**机制假设带有可执行探针，代码改写带有前提和不变量，结果同时验证性能与预期路径变化。** 当前调用图尚没有设备指令依赖、资源占用时间轴或定量Roofline；也未实现硅前模拟器／RTL与真机的统一重放。向论文系统推进应复用同一证据接口，增加设备机制、未见算子集和受控比较，而不是仅增加算子分数。

## 成本与证据

| 算子 | 方法 | 输入token | 输出token | 模型秒数 | 生成至完整评测秒数 |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(costs)+f'''

两组同候选上限，不等同实际墙钟相等；探针成本另存 `seed-probe/cost.json`，起点评测时间和进程启动记录在host日志中。未获得可信计费金额，不以token估造费用。两批证据核验共 **{audit_count}项通过**；早期pilot-v2另有134项身份核验通过，不将其失败候选计为成功。

## 文件与复现

- [开发批次](campaigns/{development.name}/README.md)：完整对照、tanh_接口失败与158项核验。
- [独立检验](campaigns/{heldout.name}/README.md)：独立算子两组、冻结身份和原始证据。
- [机制模块](../tools/profiling/hgkda/mechanism/README.md)：探针、规则、调用关系与命令。
- `tools/profiling/hgkda/reproduce-mechanism.py`：无模型重跑交付源码，使用新目录、卡锁、21项门禁和一次完整KGS；该入口做过语法检查，未为检查入口再重复正式测试。
- `tools/profiling/hgkda/audit-mechanism.py`：离线核验来源、合同、冻结模板重建、预测及原始归档。
- `tools/profiling/hgkda/report-mechanism.py`：从已核验批次重新生成本报告与CSV；须指定新输出，避免覆盖旧报告。

环境固定为910B物理卡7、容器 `tle_yy`、KGS19659；FlagGems提交 `d6a8eec473517a3d68157b208eb9c057eb1d4c50`，FlagTree `c286cba6d5c30f07988c0414beb21fd2be1abb3e`，锁定记录见仓库 `manifests/hgkda-stage2.lock.json`。快路径使用固定后端私有启动ABI，升级须重新验证。Windows长路径profiler保留在 `raw-evidence/remote-files.tar.gz`，不是Git LFS指针；索引记录每个成员SHA。未修改共享镜像、其他服务或他人进程。本轮文件待用户后续授权提交GitHub。
'''
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text,encoding='utf8')
    with output.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['算子','用途','方法','case_id','输入','FlagGems_us','候选_us','PyTorch_us','相对FlagGems','相对PyTorch','PyTorch参考变化','源码SHA256']);writer.writerows(csv_rows)
    print(json.dumps(dict(output=str(output),formal_rows=len(csv_rows),audit_checks=audit_count),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True);p.add_argument('--heldout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.development,a.heldout,a.output)
