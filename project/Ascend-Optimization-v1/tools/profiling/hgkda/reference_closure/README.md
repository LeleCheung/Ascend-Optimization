# 最后六项与 CANN 原生对照夜间攻关

起点为 2026-10-31 已交付的24/30项常规稳定达标。目标是推进其余六项：`linalg_inv_ex`、`cumulative_trapezoid`、`fractional_max_pool3d`、`_conj_physical`、`fake_quantize_per_tensor_affine_cachemask_out`、`batch_norm_with_update_functional`。另外补齐最近两项`fix`、`expm1_`的CANN直接参考。

## CANN 与候选的角色

- PyTorch保留为原正确性和历史性能参考；原合同不覆盖。
- CANN原生库是直接调用`aclnn`的独立参考，不是自写Ascend C。先核验设备、dtype、参数、别名和输出语义，再测同一KGS合同。
- HG-KDA候选可以是KG生成Triton、Codex改写Triton，或自写Ascend C＋布局回退。厂商目标API只进入明确命名的参考列，不冒充自主优化。
- 没有对应单算子时采用已审计NPU分解作为补充对照，并独立命名。存在API头文件不等于910B和全部dtype/目标语义已经支持。

不是所有实现都由人工从头手写：历史候选包含KG＋CC/DeepSeek生成、HG-KDA证据辅助生成、Codex改写Triton，以及近期自写Ascend C。归属以每份源码和运行记录为准。CANN是可直接调用的厂商实现，Ascend C是编写自定义设备算子的语言／接口；包装CANN参考不算新设计了该算子。

`scan_cann_support.py`只读本机9.0.0头文件及`libopapi.so`导出符号，生成[30项目标接口清单](../../../../operators/campaigns/reference-closure-20261031/research/cann-api-inventory.json)。接口存在只作为下一步资格检查的依据；例如`aclnnInverse`没有`info`输出，`aclnnCumsum`只是累计积分的扫描部分，不能直接声称整个目标已有合格单算子参考。

## 夜间顺序

1. 冻结六项源码、原合同、固定软件版本和前批成果；不重跑已有24项搜索。
2. 为`fix`、`expm1_`、伪量化、BatchNorm及求逆构建直接CANN C++包装。保留输入/输出分配与stream正确性；不缓存输入数据或评测结果。
3. 在物理卡7项目锁下串行运行资格检查、完整KGS、补测与设备路径。CANN不支持的规格明确登记，不偷偷混入候选回退。
4. 对通过资格的CANN/NPU补充参考，保持原输入与容差，新建明确命名的对照合同，评测现有候选；保存原PyTorch成绩和新增参考成绩。
5. 求逆保留奇异输入，分别核对inverse、info和残差。不能用给奇异矩阵补一个info=0、删除用例或放宽容差凑过线。
6. 累计积分先复现默认参考不稳定问题，再核验确定性条件；存在参考矛盾时明确列为待专家确认，不能称30项原合同全部通过。
7. 输出30项验收状态、原PyTorch/CANN/NPU分解分列成绩、机制经验和复现命令。知识只吸收有源码、请求、结果及补测身份的记录。

## 执行边界

宿主tmux独立运行，控制端关机不影响已提交任务。设备任务仅物理卡7串行，不重启共享服务、修改镜像或升级固定FlagGems/FlagTree/CANN。使用独立构建目录，单CPU编译；每个新源码默认一次完整KGS，临界确认须预先登记，不反复挑高分。未知请求保留并等待核验，不自动重发。

方法重点是将“参考支持核验→设备路径→差异诊断→完整验证”接入现有HG-KDA，而非仅增加算子分数。是否全部达标由最后完整结果决定。

## 本批进度与复现

最新状态见批次`next-state.json`与`heartbeat.json`；`30项验收总表.md`区分原参考常规达标和命名NPU参考。服务器tmux会话为`hanle-reference-closure-20261031`及`hanle-reference-finish-20261031`，使用同一物理卡7项目锁。后续队列等待锁时没有设备子进程。

`check_cann_semantics.py`额外检查伪量化动态scale／zero_point和BatchNorm动态eps／momentum、缺省仿射参数，覆盖三种dtype、非连续布局、独立out及调用者stream。`probe_inverse_isolated.py`对每个求逆输入启动独立子进程，避免一次设备异常污染后续规格；只核对逆矩阵、残差，不伪造`info`。

`report_v2.py`核验请求、冻结合同、源码／二进制SHA、额外语义检查和逐case低分；`acceptance_table.py`核对既有24项交付后汇总30项。它们是离线报告入口，不启动设备评测。

本轮队列已完成。`finish_layout.py`保存纯Triton通用布局修复的失败反例；`finish_hybrid.py`改用非末轴Triton增量＋直接CANN扫描，`conditional_scan.py`在独立进程同时对候选与参考启用官方确定性条件。默认、确定性结果及补测分开归档，不缓存计算结果、不更改原容差。混合版的厂商扫描桥由`build_cumsum.py`单线程构建，原源码、头文件接口及二进制SHA保留。

复现已交付混合版时，在服务器项目根目录使用以下命令；应先取得物理卡7项目锁，并确认本项目KGS空闲。选择新输出目录，不能覆盖未知请求。

```bash
# 在现有tle_yy容器内，保持CANN 9.0.0、Python 3.11.15和软件锁版本。
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export PYTHONPATH=/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src:/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/client:/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server:/data/hanle/ascend-optimization/Ascend-Optimization/sources
/usr/local/python3.11.15/bin/python3.11 tools/profiling/hgkda/reference_closure/conditional_scan.py \
  --source operators/campaigns/reference-closure-20261031/proposals/cumulative_trapezoid/hybrid-cann-v3/main.py \
  --leaf operators/campaigns/reference-closure-20261031/catalog/ops/cumulative_trapezoid \
  --output /data/hanle/ascend-optimization/runs/scan-hybrid-deterministic-new
```

脚本同时完成原33项KGS和36项新数据／stream补测；确定性4.476×的参考条件不同，不能写成默认PyTorch下的加速比。服务器原Git HEAD和工作文件同步分别记录，不能只因文件已复制就声称三端Git HEAD相同。
