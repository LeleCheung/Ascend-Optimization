# 多kernel调用中后续阶段的模拟器采集

整调用在前置阶段耗尽单次模拟时限时，可先保存真机产生的中间输入，再单独回放目标launch。该入口用于热点与流水分析；整算子的正确性、完整性能及候选保留仍使用固定原合同。

1. 完成原IR或已审查候选的完整KGS及必要补测，保存源码、IR、实际二进制和成对profiling。
2. 使用实际paired `spec.json`，在共享设备锁内运行 `python -m hgkda.runtime.launch_replay --spec <spec> --kernel-name <symbol> --output <snapshot>`。运行端须使用真实设备运行库。前置kernel在真机执行，目标launch前同步并保存张量和参数。
3. 使用独立观察方法包运行 `hgkda.runtime.collect_simulator`，在原参数基础上增加 `--kernel-replay <snapshot>/manifest.json`。`--kernel-name`须与快照目标一致；原paired发现、合同、case和源码身份继续核验。
4. 核对回放的源码、张量SHA、其他阶段IR、目标编译收据和二进制。设备目标名称分析别名继续要求与真机二进制逐字节一致。
5. 同时保存指令热点与流水等待／可观测空闲。超时的部分窗口保存真实退出与边界；逐PC时钟校验、源码映射和应用完成状态分别记录。

当前快照入口支持完整连续张量，相同存储的相同视图保持共享；其他别名视图显式拒绝。输入与输出参数均按原launch保存，回放启动原Triton函数，ABI、参数和网格保持一致。快照的张量文件仅作诊断输入，候选及正式评测入口继续执行完整原计算。

GQA源码候选可通过 `attention_profile_driver.py --snapshot-kernel _gqa_fwd_kernel` 保存前置阶段的真实结果，随后用 `--kernel-replay <output>/launch-snapshot/manifest.json` 单独重放 attention。该入口复用原合同的输入生成器和公共 `capture_call`，记录合同、源码、原参数、真实前置launch及张量身份；源码模式和IR覆盖模式分别核验，IR模式继续固定所有前置阶段的IR。`gqa_continue_observe.py` 在发现多个编译kernel后分别采集辅助阶段profiling，再创建真机快照与模拟器重放。正式完整调用计时始终覆盖全部阶段及分配。

直接以脚本路径启动paired driver时，辅助模块通过同目录加载。观察方法包按版本冻结，采集失败日志、旧包和旧快照均保留。IR续接协调器为`development/tools/research/ir_isolate_collection.py`，重试协调器为`ir_retry_simulator.py`。

续接评测统一使用`development/tools/research/ir_collect.py`；scatter传入`--kernel-name _scatter_add_sort_out_kernel --isolate-target`，先完成成对profiling，再自动保存真实中间输入并隔离采集。当前scatter成功方法包为`runs/ir-performance-20261119/observation-method/hgkda-v5`。入口在创建采集目录前检查方法包及IR采集支持，保留旧版`ir_collect_v3.py`作为历史工具身份。

已验证实例为[scatter_add写回采集恢复](../reports/ir-performance-20261119/scatter-collection-recovery.md)：真机前置排序结果保存后，sort_out部分窗口取得目标二进制一致性及152/152个逐PC时钟匹配。该采集验证与后续候选性能验收分别记录。
