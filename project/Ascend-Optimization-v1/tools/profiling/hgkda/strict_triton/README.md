# 严格Triton候选的10＋20复核

用户要求：性能baseline优先同语义CANN原生库或来源明确的Ascend C参考；我们的候选严格Triton。Triton编译器生成DMA允许，宿主直接调用原生DMA、CANN算子或Ascend C设备核代替目标计算不允许。Python／C++只可作分配、元数据、生成器状态和Triton启动胶水。

本批`operators/campaigns/strict-triton-20261102/`独立保存源码和证据，不沿用历史混合后端25/30、24项等总数。固定原30项输入、容差、FlagGems、FlagTree、CANN及KGS完整调用计时，按1015原10项和1029新增20项分别统计。

## 检查顺序

1. `prepare.py`恢复Ascend C包装追加前的Triton代码；恢复累计积分的纯Triton扫描；将历史回退中的PyTorch复制、连续化、填充及转换改为内嵌Triton核。全部原始来源与SHA保留。
2. `gate.py`按AST查找已知禁止的外部库、厂商接口及宿主Tensor计算；`review.py`记录实际人工逐文件、逐分支审查与固定依赖身份。静态扫描不是完整程序证明，不能仅凭“导入Triton”或backend标签准入。
3. `runtime_audit.py`在候选调用范围内使用TorchDispatchMode拒绝非许可ATen计算，输入和参考生成在范围外。全部原正确性输入与最大性能输入检查；未触发分支由源码审查补齐。该检查不替代数值测试，也不作性能计时。
4. 使用原PyTorch语义做独立补测和一次完整KGS；使用本机已完成资格验证的CANN参考作性能及语义对照。候选源码完全相同且合同、参数和补测一致时，复用已有CANN原始请求，不重新测高分。
5. 数值通过的候选采集一项代表输入trace。采到文件不等于捕获设备任务：本轮发现厂商解析器未输出设备任务，原空trace如实保留。`verify_compiled.py`在完整队列结束后持同一卡锁，提取代表输入实际调用返回的Triton编译对象，核验TTIR、npubin及逐文件SHA；与源码全路径、依赖和运行时审查共同作为后端证据。实际IR不能伪称为成功的硬件trace。
6. `report.py`核验源码、请求、计时参数和逐case几何平均，重建两张中文表。原语义／补测失败、相对CANN未过0.8×、参考资格不足分别列示。

## 无人值守运行

910B仅使用物理卡7和`/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock`，三个KGS服务必须健康且空闲。运行于宿主tmux，Windows断开不影响队列：

```bash
cd /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python tools/profiling/hgkda/strict_triton/run.py
```

既有tmux会话为`hanle-strict-triton-20261102`。先检查会话与本批`state.json`，不要重复启动；恢复时已完成任务复用原件，未知正式请求禁止重发。旧`cann_priority`、`remaining`等队列不具备本轮准入约束，不能直接续跑。

编译产物复核排在主队列之后，tmux为`hanle-strict-compiled-20261102`，使用相同项目锁；它等待主队列结束，不会并发占用卡：

```bash
/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python tools/profiling/hgkda/strict_triton/verify_compiled.py
```

QKV历史C++宿主桥经逐函数审查后允许恢复：只分配、取元数据与当前stream、打包参数并调用Triton生成启动器。`restore_host_glue.py`将候选、C++源码和二进制绑定SHA；未审查外部库仍被拒绝。旧纯Python宿主复核结果单独归档，不覆写。对同字节、同合同、同计时的既有完整结果保留原始请求，复用时明确标记，不重测挑分。

`delivery.json`中的当前达标数要求原合同与补测通过、后端审查通过、相对合格CANN参考超过0.8×，以及实际Triton编译路径核验通过。单接口与兼容／组合参考在表内分开标注；参考不合格或缺失不能算作CANN对照达标。

`report.py`同时生成`strict-implementation-context.json`：`implementations`只包含后端、原合同、补测和实际编译产物已核验的Triton源码，并单独标记是否超过性能验收线；`references`只存合格CANN对照。候选与参考不能互换，新算子不能继承其他实现的成绩。重新取用时须核验源码与证据身份。

新建`build-kg-overlay.py`覆盖层会把[角色硬约束](角色硬约束.md)插入全部KG角色的正文开头，保留原frontmatter，完整诊断知识仍只追加给两个分析角色。现有已冻结覆盖层不会自动改变；旧夜间控制器不能因此视为已经具备严格Triton准入。

运行时首遍曾把`aten.new_empty`误判为计算操作；该接口只分配未初始化存储，符合宿主胶水边界。`repair_runtime_allocations.py`仅对因此失败的两项重做运行时检查，并执行此前尚未提交的完整评测，旧失败检查与状态保留在`revisions/`。填零、复制和实际计算未加入许可列表。

`repo_identity.py`可由指定Git提交生成实体SHA清单，并比较工作区；LFS使用实体OID与字节数，指针文件不能通过实体核验。普通文本的Git规范换行与工作区换行可能不同，完整仓库审计会如实显示这些差异，不自动改写历史文件。本轮的原始合同、请求、结果、源码快照和编译附件保持采集字节，另行逐文件核验。

## 一次新复测

完整队列结束、卡锁空闲后，在项目目录执行以下一条命令。输出目录每次取新名字；已存在目录会拒绝重发。脚本先核验冻结源码、审查记录、依赖和服务版本，再持同一卡锁执行一次完整KGS：

```bash
/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python tools/profiling/hgkda/strict_triton/replay.py --operator selu_ --baseline CANN --output /data/hanle/ascend-optimization/runs/strict-replay-selu-01
```

`--baseline PyTorch`用于原语义与历史计时复测；当前性能验收仍使用合格CANN／Ascend C参考。脚本不会为缺少合格CANN参考的算子自动切换参考，也不会覆盖冻结批次的源文件、请求或状态。

本批是源码恢复、路径修复和重新验收。修改来自Codex，历史KG／CC／DeepSeek源码归属保留；不能表述为新一轮无人辅助模型生成或将不同后端差异直接归为算法收益。
