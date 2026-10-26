# 会议剩余三项最终验收

当前交付见[精简汇报](精简汇报.md)、[机器可读清单](final-delivery.json)及[证据审计](evidence-audit.json)。7项默认达标、1项确定性条件达标、2项补充NPU分解对照通过；不是10项默认全部达标。

## 复现前提

1. 固定FlagGems d6a8eec、FlagTree c286cba6、CANN9.0、torch_npu 2.9.0.post2；不要更新共享环境。
2. 910B仓库位于`/data/hanle/ascend-optimization/Ascend-Optimization`，运行容器`tle_yy`，KGS19659只使用物理卡7。安装Git LFS并执行`git lfs pull`。
3. 在宿主进入`project/Ascend-Optimization-v1`。下面复现入口自动持串行卡锁，检查三服务空闲并提交隔离KGS debug作业；输出标签必须未使用。

```bash
python3 tools/profiling/hgkda/reproduce-acceptance.py   --operator fractional_max_pool3d --reference default --label pool-default-new
python3 tools/profiling/hgkda/reproduce-acceptance.py   --operator fractional_max_pool3d --reference npu-decomposition --label pool-npu-new
python3 tools/profiling/hgkda/reproduce-acceptance.py   --operator _conj_physical --reference npu-decomposition --label conj-npu-new
```

原参考含CPU回退；`npu-decomposition`是自建多原语补充参考，均不能冒充已有官方NPU单算子。两组结果不平均、不择高替换。源码已通过完整补充门禁，复现入口重新运行准入和正式完整评测；新输出位于`reproductions/<标签>/`。

默认扫描反例及执行方法见[反例报告](默认扫描参考反例.md)。已完成的官方确定性0.812×源码和33/33成绩继承上批`two-closure-20261024`，复现仍使用`reproduce-two-delivery.py --operator cumulative_trapezoid --deterministic --label 新标签`。本批不声称重新优化了扫描性能。

## 文件索引

- `deliverables/`：三项交付源码；共轭和确定性积分复用已验证版本。
- `catalog/`：原合同完整副本，身份见plan；`npu-decomposition-catalog-v2/`：明确命名的独立补充对照。
- `validation-v2/`：有效两类参考正式结果、条件、完整门禁、成对路径和IR；池化父版对照在`parent-comparison/`。
- `validation/`：首次验证在缺catalog manifest处停止；数值门禁原件保留，不伪装成完整性能交付。
- `research/`：后端注册、CANN接口、源码与反汇编、默认/确定性反例、布局与分块候选及选版。上游master源码仅作参考，安装torch_npu提交的精确op-plugin版本另有证据。
- `stages/`、`jobs/`：调试请求与回执，失败和超时参数被服务拒绝的记录保留。
- `raw-evidence/`：不可变LFS原始归档和逐文件SHA；Windows超长路径仅在归档内，910B保留原展开文件。

## 验收边界

CPU回退高分表示避开CPU往返，不说明设备端比原生官方内核快相同倍数。自建分解仍有多次kernel、临时张量和host包装开销，存在`aten::item`等事件；第二版修正零维表达不等于已消除所有host开销。实测与源码、IR解释分开，未建立定量Roofline。

