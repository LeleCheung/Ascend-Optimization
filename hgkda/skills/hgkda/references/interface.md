# 调用接口

先在项目根目录安装 python -m pip install .。skill 脚本与 python -m hgkda 使用同一入口。

    python -m hgkda doctor
    python -m hgkda compare --directory examples/paired-expm1/paired --output /tmp/comparison.json
    python -m hgkda context --operator expm1_ --source examples/paired-expm1/candidate.py --comparison /tmp/comparison.json --output /tmp/context.json
    python -m hgkda hotspots --listing examples/matmul-instructions/listing.json --source examples/matmul-instructions/candidate.py
    python -m hgkda regions --listing examples/matmul-instructions/listing.json --source-map examples/matmul-instructions/source-map.json
    python -m hgkda knowledge --operator expm1_

Windows 将 /tmp/ 换成项目外的新输出目录。每次输出使用新文件名。

hgkda.result/v1 封装含 operation、status、inputs 和 result。每个输入记录 SHA-256；结果可直接由模型或网页读取。compare 的输入是同一 spec.json 下两侧 collection.json 及 CSV。hotspots 要求指令清单声明 source_sha256；历史清单需由采集请求和源码校验后显式导入，不能仅凭文件名绑定。

context 的 facts 保存对照、源码/IR、Roofline、热点和区域；hypotheses 单独保存经验规则结果；knowledge 给出版本化参考。它不自动调用任何模型，也不宣称模型已按建议完成优化。

`regions --listing <listing.json> --trace <bound-trace.json> --kernel <symbol> --core <core> --trace-unit us`接受单kernel、单硬件核的逐指令时间轴。时间轴须携带`hgkda_binding`，声明`source_sha256`、`case`、`kernel`、`core`、`time_unit`，并与指令清单匹配；采集适配层负责核对原始文件和范围。单位须由采集格式确认，不能从`displayTimeUnit`猜测。逐流水使用区间并集计算`productive_union`、`wait_only_union`与`idle_union`，三者相加等于相同观察窗口；流水之间可重叠，不可相加为总耗时。

将结果通过`context --listing <listing.json> --regions <regions.json>`与相同源码、case的成对对照及实际IR一起交给模型。`idle`表示窗口内未观察到该流水执行，不等同已证实的阻塞原因或关键路径；模型须提出具体机制和验证预期。

roofline --input <json> 接受已核验的 GEMM 工作量、同次采集指标和同环境经验参考：facts、shape（M,N,K）、dtype、empirical，并保留 source_sha256 与 case。缺少工作量合同时使用成对 profiling，不用估计值填补未知字段。

## 知识整理

curate --records <records.json> --root <evidence-root> 按算子类型、动作和作用域去重。只有源码、完整请求/结果、候选后端审查及所需补测身份均匹配，才保留 validated 状态；缺失证据降为 draft，旧条目保留在 superseded。

`observed_failure`可保存补测失败，或正确性通过但性能未达标的反例。后者须绑定同源码的完整成绩、性能反馈文件及`scope.performance_threshold`；不能仅凭文字判断失败。同步指令的CSV累计等待与trace区间口径分开；时间单位仅用非同步指令交叉核验，等待指令仍参与区域及时间轴统计。

retrieve --index <index.json> --operator-type <type> --scope <scope.json> --max-chars 6000 重新校验证据后按作用域检索，不截断单条记录。它是确定性整理规则；模型提出的新经验不会未经验证自动成为成功规律。

gate --source <candidate.py> 只执行已知路径静态筛查；--review 可绑定已有审查记录。KGS evaluate 还需冻结请求、审查与新的输出目录。后端详细用法见仓库 docs/backends.md。

## 性能反馈

`feedback --operator <name> --source <candidate.py> --result <result.json> --audit <audit.json> --timing-cases <timing.jsonl> --output <new.json>` 校验源码、正式审计及计时case身份，重建几何平均，按性能损失选择代表case，并覆盖各dtype的最大输入。

返回 `continue_optimization` 时把逐case反馈交回模型继续修改；只有超过阈值（默认0.8×）且正式审计所需补测通过才返回 `target_met`。偏移、stream等任务专项检查另行绑定，不由这个分数决策替代。该命令不调用模型、不执行设备测试，也不代替宿主的调度和审查。

## 公共设备与模型接口

执行python -m hgkda.runtime --help，按需选择prepare、context、preflight、generate、pair、simulate或evaluate。它们随Python包安装，调用明确配置的设备与外部KGS服务。运行不依赖内部开发目录。配置示例和边界见[公共运行接口](../../../docs/runtime.md)。宿主负责审核、多轮编排和停止条件。
