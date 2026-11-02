# CANN优先对照与剩余算子优化

2026-11-02新目标：尽量使用同语义的CANN原生算子作为baseline，继续让更多算子通过正确性及0.8×验收。起点为24项原PyTorch参考常规达标、4项命名NPU对照通过、1项确定性条件通过、1项求逆失败。

先冻结30项原合同、最近交付源码和软件版本，保留原PyTorch正确性与历史成绩。直接aclnn单算子、带dtype／状态适配的CANN组合、NPU分解分别记录。接口存在不代表已支持全部输入；参考通过完整KGS和补测后，才进入正式对照。

优先攻关已有CANN对照低于验收线的`hardswish_out`，扩大其他目标的CANN覆盖，并继续核验累计积分和求逆。新候选来源写清KG／CC／DeepSeek／Codex贡献，直接调用厂商目标API只记参考。原参考本身不稳定或无同语义厂商实现时保留证据，不能靠换参考、删除输入或放宽容差凑分。

所有设备任务由910B宿主tmux控制，持物理卡7项目串行锁；单CPU构建、独立目录，不重启服务、不改共享镜像。新源码默认一次完整KGS和独立补测，不用重复测量挑高分。结果写入`operators/campaigns/cann-priority-20261102/`，最终按源码及证据SHA核验、同步与推送。

## 执行顺序

1. `prepare.py`固定30项原合同、种子和已交付源码，不覆盖前批。
2. `build.py`、`build_extra.py`及`build_storage_rng.py`构建三个独立参考桥；`make_references.py`和`repair_index.py`登记直接接口及兼容组合。
3. `run.py --phase all`在项目锁内核验参考、复测既有候选，并验证新`hardswish_out`；相对CANN及相对PyTorch分别评测。
4. `followup.py`等待主队列正常完成，再顺序执行存储／随机参考补充和低分项搜索。不会与主队列同时占卡。
5. `search_missing.py`按本批实测CANN差距固定最多3个未过0.8×的对象，使用独立子合同运行KG/HG-KDA＋CC/DeepSeek，各3轮、2400秒。最终源码再通过原PyTorch语义、CANN对照及补测；模型上下文与真实角色加载记录保留。
6. `report.py`离线核验源码、请求、二进制和结果，生成`delivery.json`及中文30项表。它不会启动设备任务。`capture.py`为通过的新增候选保存同输入设备路径，独立profiling时间不替代KGS成绩。

在910B宿主项目目录，准备后的主队列入口为：

```bash
/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python tools/profiling/hgkda/cann_priority/launch.py --phase all
```

后续链由独立tmux会话`hanle-cann-priority-20261102-followup`运行`followup.py`。已有请求只读取结果，不重复提交；控制器会核对完整bundle ID、源码和计时参数。上传摘要和bundle ID带有不同前缀，不能直接将两者当作同一字符串。

## 来源与验收

本轮`hardswish_out`由Codex复用此前自写Ascend C设备计算并改接out，复杂布局沿用历史KG生成Triton路径，不宣称本轮全自动生成。既有候选重新与CANN比较属于参考覆盖，不能记成新增代码收益。新增KG搜索的来源与这类直接改写分开记录。

当前补充Uniform桥采用公开NPU生成器和厂商随机接口，旧模式计数增量参考上游[Uniform实现](https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/UniformKernelNpuOpApi.cpp)及[RandomUtil](https://github.com/Ascend/op-plugin/blob/master/op_plugin/utils/RandomUtil.h)。上游master不是本机固定版本的证明；本机实际范围、状态推进和随机机制仍由完整合同与补测核验，通过前不列合格baseline。

共轭参考明确采用CANN Copy／Neg及运行时DMA的组合：复制物理实数存储后，按共轭位决定是否对虚部取负；这不是厂商专用共轭单kernel。累计积分的CANN Add＋Muls＋Cumsum也是组合，默认参考不稳定与确定性条件继续分别保留。
