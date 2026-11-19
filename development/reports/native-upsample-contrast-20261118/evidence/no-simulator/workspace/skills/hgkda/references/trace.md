# 连续 trace 与修改前决策

用于新实验的过程记录。每轮指一次明确命名的模型调用；模型调用、文本修改、设备预检和正式评测分别计数。记录由公共runtime执行，不依赖development目录，也不改变候选计算和KGS测试合同。

## 自动保存什么

`generate`记录调用前后源码、模型日志、初始上下文、方法清单和诊断引用；`preflight`、`evaluate`记录动作开始／结束、当前源码及已有检查／正式结果。`context`、`pair`、`simulate`分析动作也记录事件，其他专项探针仍保留各自输出，由决策记录显式引用。事件状态`returned`只表示函数返回，不能代替正确性、完整性或性能判定。

源码快照按SHA去重，事件独立编号；同源码重复评测、无修改调用和异常均不合并。可变workspace文件按内容保存快照。模型中间编辑与工具读取保留在原始`agent-events.jsonl`，本轮没有把模型每次Read都强制升级为“决策引用”。

```text
<run>/
├── trace/
│   ├── events/<event-id>.json          动作开始／结束及决策事件，创建后不覆盖
│   ├── sources/<SHA>.py               输入／输出／决策前源码快照
│   ├── artifacts/<SHA>.json           可变上下文与反馈的内容快照
│   ├── evidence/<SHA>.<ext>           决策引用证据的不可覆盖字节快照
│   ├── decisions/<decision-id>.json   修改前显式声明的证据、假设与预期
│   └── rounds/<invocation>/round.json 模型调用总结，引用上面各项和原始日志
├── agent-calls/<invocation>/          原始模型工具日志、输入／输出及方法身份
├── attempts/<SHA>/                    设备预检原件
└── formal-<SHA前缀>/                  KGS与补充检查原件
```

单个动作的开始与结束通过事件ID连接。若宿主明确选择了父轮次，可在config中提供`trace_parent_round_id`；未提供时保留未知，不仅凭SHA相同或时间相邻猜测父轮次。进程被强制杀死可能只留下开始事件，作为未闭合动作保留。

嵌套工具返回时，模型仍可能向`agent-events.jsonl`追加。引用该日志时记录`prefix_bytes`及当时前缀SHA，后续按指定字节前缀核验；可变`source-provenance.json`保存内容快照。早期无前缀长度的日志只能在原件中找到完全匹配SHA时恢复，不能重新计算一个不同SHA覆盖旧记录。

## 修改前记录什么

先填写workspace中的`diagnosis.json`，再执行：

```bash
python -m hgkda.runtime decision --config /path/run/workspace/agent-config.json --record diagnosis.json
```

CC工具桥使用同样的`decision --record diagnosis.json`动作。生成入口会将当前调用ID写入该次配置。记录提交时必须仍是修改前的源码；一个机制对应多次小编辑时，不要求为每次文本替换重复长分析。

跨版本对照使用`role: background`并保留该证据实际绑定的源码SHA。当前候选的失败日志才能作为该候选的trigger／mechanism；旧版本的性能反馈不是当前源码的实测证据。若提交提示`evidence artifact belongs to another source`，核对具体引用并修正role／来源，不更改证据文件或反复提交同一记录。每轮先登记一个简短且有效的决策，再及时修改候选；接口修复不重复复盘全部历史。

```json
{
  "source_sha256": "当前候选源码SHA",
  "evidence": [
    {
      "path": "simulator-before/core0.veccore0-regions.json",
      "category": "simulator",
      "role": "disambiguation",
      "finding": "在指定区域看到了可区分两个解释的等待现象"
    }
  ],
  "hypothesis": "当前机制解释",
  "alternatives": ["待区分的另一种解释"],
  "discriminating_observation": "哪条观察区分了两个解释；不明确时留空",
  "change": "准备修改的机制",
  "expected": "预期改变的可观察量及方向",
  "validation": "如何重新采集及完成正式评测"
}
```

`category`取`paired_profiling/simulator/source_ir/roofline/knowledge/evaluation/probe/other`；`role`取`trigger/disambiguation/mechanism/background`。完整评测的逐case耗时与加速比属于`evaluation`，不属于`paired_profiling`；只有实际成对采集并比较硬件指标的材料才标后者。没有使用某工具就不填写它，没有实际区分两个解释时不写`disambiguation`。路径相对run解释（工作区文件需写`workspace/<file>`），或使用明确绝对路径；文件SHA自动计算。登记时将引用字节复制到`trace/evidence`，保留原路径与SHA，后续覆盖工作区不会改变当时证据。已知的结构化源码绑定不匹配时拒绝登记，历史材料须显式标为`background`。

脚本保存“作者声明依据”，不会替模型判定决定性证据或证明独立因果。修改后诊断、实际指标与正式结果另存；根据结果核验预期是否成立。没有修改前记录的历史轮次标`not_recorded`，不能事后伪造记录。

## 导出与验证边界

```bash
python -m hgkda.runtime trace --config /path/run/config.json --output /path/run/trace-export-01.jsonl
```

输出必须是新文件。导出按记录时间排序，保留事件ID、源码身份和原件路径；模型内部工具顺序仍以原始日志为准。时钟顺序不证明跨并行动作的因果关系。

这套能力经过离线身份、失败、重复源码与CLI检查；真实910B优化闭环还需在下一次新实验中验证。不重新登记冻结历史原件，也不将自动日志能力说成完整自主优化调度器。
