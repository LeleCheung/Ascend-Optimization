# 新25算子：成对机制诊断与夜间队列

计划见[新25项计划](../../../../operators/下一阶段计划-20261029.md)。本目录是独立Python工具与KG覆盖层控制器，复用既有CC/DeepSeek和KGS，不升级共享镜像或修改上游源码。

## 文件

|文件|职责|
|---|---|
|prepare.py、oracle.py|登记25项、冻结FlagGems源码、建立准确签名和代表Native合同；无同名入口的目标保留待映射状态|
|run.py|宿主tmux串行队列、物理卡7项目锁、失败留档、KG/CC搜索、一次完整KGS独立复验|
|native_search.py|独立搜索控制；预算到期时给本worker已提交的同服务正式评测最多180秒收尾，保留操作身份与记录，不重发请求|
|catalog_snapshot.py|只排除派生__pycache__/pyc并逐项核验合同SHA，生成KG搜索专用快照；不删除原目录或隐藏其他未知文件|
|probe_inverse.py|搜索求逆前分别保存冻结输入的inverse、info及NPU/CPU参考差异，供Agent区分奇异标记和返回值问题；不修改合同|
|validate.py|独立检查原地/out别名、NaN/Inf、非连续输入；随机算子使用合同统计校验|
|validation_status.py|补测准入；negative必须包含目标NPU的三种dtype正负零检查，旧较小测试集需补验|
|cann_reference.py|直接调用aclnn两段接口，记录头文件和库SHA；不调用PyTorch计算目标；库内源码未核验时明确标CANN原生库|
|ascendc/、ascendc_reference.py、check_references.py|来源可见的Ascend C负号补充参考与资格核验；仅声明连续FP32、8对齐，不冒称厂商最优|
|driver.py、collect.py|相同冻结输入的候选/PyTorch/CANN语义检查、真实路径、OpProf指标与实际编译IR；多kernel逐项保存|
|assemble.py、evidence.py|对照聚合、绝对差和比率、缺失/零分母标记、分类限额摘要；绑定源码、IR、热点区域，进入KG角色|
|regions.py|源码SHA及调试文件名核验后按静态循环/条件/函数范围聚合；缺源码时退回源码/PC邻域。有逐指令时间轴时求流水区间并集及空闲补集，等待另列|
|knowledge.py|结构化记录去重、旧记录替代、源码/请求/评测身份核验、作用域过滤及上下文预算|
|context.py|将完整采集压缩为限额证据索引；保留对照数值、差值、比率、完整原件路径和SHA，源码/IR按需读取|
|baseline.py|核验KGS入口改名与固定原版AST一致；入口冲突和计算失败分别保留，报告允许已核验的入口适配|
|selection.py|按同合同的相对PyTorch加速比选择搜索轮次；成绩相同时才用候选绝对延迟择优|
|host_profile.py|对已验证候选的代表输入采集宿主cProfile和编译缓存状态；诊断计时不替代KGS成绩|
|capture_record.py|对当前完整评测及补测通过的指定记录补采；选择最大冻结FP32输入，不重跑正式评测，旧版证据不替换最佳上下文|
|compare_records.py|核对输入/合同及证据SHA后比较新旧候选，保留搬运量、指令数、时间和粒度各自变化，不将指标相关性自动写成性能因果|
|host_bridge.cpp、build_host_bridge.py、load_host_bridge.py|实验性C++宿主启动包装、单CPU构建与源码/二进制/ABI核验；计算仍由Triton完成，完整KGS与补测通过前不计性能成果|
|monitor.py、report.py|只读查看工作区终态和正式轮次；核对请求与源码身份，自动生成可汇报进展|
|export_best.py|生成25项目标的跨机器最佳源码索引，绑定源码、合同、版本、正式请求/结果与补测SHA；区分新达标、历史复验、未达标及参考条件单列，不复制实验代码|
|followup.py、resume.py|在持卡锁的任务边界接续控制器，不重复未知请求；加载新证据和修正候选后继续队列|
|proposals.py|在任务边界验证带源码SHA的排队提案；完整KGS及补测通过后才能择优和入知识库|
|prepare_launch_reuse.py|将历史SELU启动方式迁移到negative候选，附来源SHA；生成不代表正确或更快|
|test_analysis.py、test_controller.py|区间重叠、捕获窗口、等待与空闲、零分母、缺证据知识拒绝等测试|

## 启动

Windows准备新批次；输出目录必须不存在：

```bash
python tools/profiling/hgkda/next_phase/prepare.py --output operators/campaigns/next25-20261029-v2
python -m unittest discover -s tools/profiling/hgkda/next_phase -p test_analysis.py -v
```

服务器项目目录下，用宿主控制器Python启动；tmux不依赖Windows保持连接：

```bash
tmux new-session -d -s hanle-next25-v2 '/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda/next_phase/run.py --batch next25-20261029-v2 >> /data/hanle/ascend-optimization/runs/next25-v2-controller.log 2>&1'
```

查看operators/campaigns/批次/进展.md、next-state.json和宿主runs/批次/*.log。恢复时执行相同入口，已完成结果按原源码核对，不重发未知请求。只想在工作边界停下时创建宿主runs/批次/STOP-AFTER-JOB；它不会杀掉其他进程。首次默认14小时预算，到边界保留状态并报告待执行项，不把预算结束等同25项交付完成。

## 本轮修复与解释

首个next25-20261029合同有NaN输入却漏设equal_nan，导致PyTorch参考自检也失败；v2显式开启相同位置NaN相等，不删除特殊值或放宽数值容差，旧原件保留。

新版结构化上下文不再将所有历史Markdown全文追加到角色。它提供当前成对数值、实际路径、当前源码/IR附件、区域和作用域兼容的有限知识；当前指标缺失时写明状态，交由原生KG继续采集。

模拟器trace的ts/dur默认按Chrome微秒解释，displayTimeUnit仅为显示偏好；每次使用需核对厂商导出格式。空闲是指定核/流水在捕获窗口内没有可见事件的区间，不能直接称依赖空泡或关键路径。静态指令清单没有时间轴时，不提供空闲比例。

CANN参考中生命周期同步只用于正确性/设备任务采集，不能将该Python封装的完整调用耗时和PyTorch作公平速度比较。正式速度仍由KGS同一合同测量。原生库具体实现尚未核验，不能称已获取厂商Ascend C源码参考；后续可在同一注册结构增加来源明确的源代码实现。

已加入并在固定CANN9.0.0编译来源可见的negative Ascend C补充参考。构建只使用2个CPU任务，不占用NPU；正式资格检查由持卡锁的队列执行，覆盖数值、正负零、原地别名与不支持规格拒绝。构建命令如下，产物放在仓库外runtime，不改共享安装：

```bash
cmake -S tools/profiling/hgkda/next_phase/ascendc -B /data/hanle/ascend-optimization/runtime/next25-ascendc-build -DCMAKE_BUILD_TYPE=Release
cmake --build /data/hanle/ascend-optimization/runtime/next25-ascendc-build -j2
```

## 2026-10-29真机核验

- 来源可见的Ascend C negative参考：4项数值检查和3项不支持规格拒绝检查通过。CANN expm1_/native_group_norm的8项FP16/FP32、布局/affine检查通过。详见主批次reference-qualification-v1.json；这是参考资格，不是优化成绩。
- probe_inputs.py对同一输入检查复制入口、固定FlagGems模块及公共入口。negative_/expm1_的7×9非连续FP32用例三种FlagGems入口均失败，PyTorch参考通过；连续对照通过。不再将该失败归因于单文件适配。
- regions.py已在历史矩阵乘的6份真实模拟器trace重放，输出保存在tool-replay-v2/。默认窗口使用硬件流水首末可见事件，辅助CACHEMISS泳道不扩大窗口。非等待事件、等待与空闲分列，不写成已证明的关键路径空泡。
- maintenance.py等待同一卡锁后执行短验证。run.py修复缺父目录、本地准备失败恢复和历史起点误标FlagGems的问题，保留旧结果原件。
- debug_budget.py只安装到新的KG独立覆盖层：首次正式评测前最多2次debug，单次最多120秒，获得新正式结果后再补充预算。这是任务调度约束，不改变正确性、计时或得分。

当前正式新成果以批次完整KGS结果和补充检查为准，不以工具实现或参考接口成功代替算子达标。

## 目录缓存与独立输入布局

host_profile.py导入oracle曾生成两份__pycache__，导致fix的KG目录校验在模型启动前失败。catalog_snapshot.py保留原目录，在新快照内仅排除派生pyc，逐项核验25个合同；真实KGS客户端已加载全部25项、712条用例。后续worker禁写字节码。对于已确认结束、未进入设备评测且明确属于此目录错误的旧工作区，控制器只另建一次attempt-2，原错误和请求记录继续保留。

补测按两个输入分别检查布局。subtract_补齐A连续、B非连续以及广播和alpha切换；QKV补齐输入/bias独立stride和输出互不覆盖；special_xlogy_、fmod_tensor_补齐双侧stride、广播、NaN/Inf及原地语义。检查器原字节、SHA和复现命令随每份结果保存，不修改冻结KGS合同或以旧小测试集替新要求背书。

## 后续接续与上下文整理

2026-10-29夜间接续中，expm1_和native_group_norm的候选/PyTorch/CANN三侧指标已采集；初版参考符号漏选的原件保留，修正采集位于symbol-repair-1。agent-context-v3.json保留完整指标比较附件；context.py将实际注入的采集摘要从约11万字符压至约2.6万字符，再追加有限历史知识。每类最多3个显著差异指标，保留完整数值原件。这里15%的筛选阈值是关注度规则。

续跑从已验证最佳源码开始；若历史成对采集绑定的是另一份源码，上下文显式标记不匹配。KG实际物化角色、加载上下文快照和SHA分别归档。已运行中的角色保持原样，新控制器在任务边界加载修改。

negative_的首个启动复用候选完整KGS通过，但正负零补测失败；后续符号位翻转与缩块的验证结果见下文。知识库将这类可核验失败记为observed_failure，与成功实现分开，不自动归因为某条编译优化。

两份simulator-cache共约915MB已迁至仓库外runtime/next25-compiler-cache/；位置和文件SHA见本批reproducible-cache-migration.json。原始CSV、trace、实际IR及评测结果继续保留。重采时编译缓存可重新生成。

服务器只读查看命令：

```bash
/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python tools/profiling/hgkda/next_phase/monitor.py --batch operators/campaigns/next25-20261029-v2 --runtime /data/hanle/ascend-optimization/runs/next25-20261029-v2
```

进展以本批已核验进展.md和next-state.json为准。Windows同步是运行中的时间点快照，不能据此声称持续变化的三端完全一致；本轮尚未提交或推送GitHub。

## 22:30后续队列

negative的原始取负写法已通过NPU补测和实际TTIR定位为`0-x`；两个安装版semantic.py也有同一写法。旧候选移出合格最佳列表，原始成绩和失败仍保留。符号位翻转的大分块版本触发UB溢出；缩至8192后通过正式30/30、补测60/60，完整调用加速比0.512×，性能仍未过线。后续成对采集绑定新源码SHA，避免沿用旧候选指标。

新控制器加载调整后的计划：后续算子每项搜索最多20分钟，保留原三轮设置和完整独立复验，为25项目标留出覆盖时间。旧计划按SHA归档；已经启动的worker不被热修改。原版若已超过0.8×，先完成补充检查，满足后直接保留原版。

调度先补齐未知目标的原版完整评测，再开始后续长搜索；每完成一项即保存状态与报告，参考路径审计和补充检查按需接续。

nan_to_num增加同一存储连续切换默认/自定义/非有限替换参数、三种dtype和连续/非连续布局的18项检查。发现的潜在语义修订先存为提案；当前最佳若已通过补测，则跳过该修订。需修订时仍先完整评测和补测，保留CC生成与Codex修改的源码归属。

知识检索继续按作用域选择已验证事实；另外最多提供两个同计算类型、同编译器的源码示例，允许借鉴启动/布局组织，但不继承其他算子的公式、分块或成绩。源码示例再次核验原件与补测。归一化、随机、布局融合等目标各自检索相应分类文档。

23:13：nan_to_num的六项自定义参数失败已修复，提案正式30/30、补测54/54，完整调用0.758×；仍未过性能线。当前控制器已开始补齐剩余21项原版，首项log2_正在评测。排队提案、补测准入和基线优先调度等20项控制器/分析检查通过。

23:31：selu_最初0/36来自KGS的同名公开入口冲突。新入口文件仅将原实现改为私有名称，保留run(A)包装及原有计算/回退；AST核验拒绝任何超出改名的改动。重测36/36、0.135×，旧失败保存在baseline_history和原结果路径。固定FlagGems、FlagTree和评测合同保持原样；当前21项控制器/分析检查通过。

uniform补测复用已有check-endurance.py，增加非连续布局、输入不变、分配、范围、stream，以及默认生成器seed/offset高位、连续调用推进和状态恢复重放。详细结果单独保存并绑定源码及检查器SHA，统计分布通过不能替代状态检查。原版或历史实现若在本批复测已过0.8×，优先补充检查，通过后直接保留并标明来源。

候选择优先满足当前完整评测与补测，再在同合同、同环境下按相对PyTorch几何平均加速比排序，与0.8×验收线一致；绝对延迟和参考延迟均保留。尚未补测的原版/历史起点只作占位，不能替换已通过当前补测的实现。旧控制器按候选绝对延迟排序，已在状态中记录重选事件，原评测不改写。新增提案在正常任务边界自动处理，未进入最佳的实现保留原件而不替换成功知识。

## 2026-10-30接续

25项固定FlagGems原版扫描已完成：11项完整通过、11项部分通过、3项运行/编译失败。完整通过还需补测和参考路径核查，原版成绩不算新优化成果。`fix`遇到CUDA设备限制，`special_round_out`使用当前后端不支持的FP64转换，`fmod_tensor_`在BiShengIR转换中失败；三项原始请求和错误均保留。

`nan_to_num_`的stage-nomask-v3提案完整30/30、补测54/54，但仅0.727×，未替换0.758×的已验证版本。重选记录绑定源码SHA和原始结果。搜索内部的轮次选择也改用相同的相对加速比指标，避免先按绝对延迟选错轮次，再对其独立复测。

新增两项排队提案，提交时尚无成绩：`fix/ieee-trunc-v1`通过IEEE尾数处理实现截断并支持stride；`native_group_norm/warmup-launch-v1`明确取得编译产物，改善启动路径，并独立处理可选weight和bias。`validate.py`为前者增加三种dtype的整数边界、特殊值与目标NPU正负零检查，为后者增加六项单独缺少weight/bias的检查；旧的较小补测不能自动满足新门禁。

交接到v14后优先核查fake_quantize与batch_norm两个异常高分原版的参考路径，再继续其余优化。v14通过同一卡锁等待当前log2_任务结束，交接标记不取消正在运行的任务。29项控制器与分析检查通过；算子是否交付仍以实际完整KGS和补测为准。

热点区域增加可选的源码AST范围：只有冻结源码SHA与调试文件名精确匹配，才按静态循环、条件或函数范围聚合；不猜测同名文件。已在本批negative_与native_group_norm真实清单重放，后者识别出四个循环范围，原始模拟器采集仍保留partial状态。源码范围不是编译后基本块，也不用于声称重建了关键路径。`--source-map`接受“调试文件名到冻结源码路径及预期SHA”的JSON映射，独立复现如下：

```bash
python tools/profiling/hgkda/next_phase/regions.py --listing candidate/listing.json --source-map source-map.json --kernel kernel_name --core core0.veccore0 --output source-regions.json
```

`assemble.py`在源身份核验后自动使用相同聚合，输出继续携带候选模拟器状态及完整来源。知识检索每次重新核验请求、结果、源码和当前补测门禁；新门禁出现后，旧的较小测试集不再自动作为已验证经验。

00:52更新：log2_独立复验30/30、0.593×；fix/vector-trunc-v3完整30/30、补测39/39、0.316×；native_group_norm/separate-output-v2完整27/27、补测42/42、0.728×。此前两个高分原版的参考审计均发现PyTorch明确CPU回退，单列该比较条件。当前33项控制器/分析检查通过。v15接续时可用`--host-profile native_group_norm --host-profile fix`执行短宿主调用诊断，再恢复原25项队列。

01:47更新：negative_达到0.643×（30/30、60/60），fix达到0.476×（30/30、39/39），两项round接口完整30/30与48/48补测通过但仍未到0.8×。selu_历史源码本批0.905×、36/36及42/42通过。hardshrink_backward的1.042×候选补测失败，不能计为交付，修订继续排队。

`proposals.py`支持可选`extra_check_revision`：补测文件及job key增加版本后缀，旧失败原件保留，完整KGS请求/源码不变时不重复正式评测。expm1_原新增的小值检查使用零绝对误差对照NPU，无法区分NPU参考的消去误差与候选错误；新版沿用正式NPU容差，再独立对照CPU小值相对精度。新旧检查分别落盘，KGS冻结合同和容差不变。当前40项检查通过。

`resume.py --skip-negative-refresh`用于只更新控制器的交接，保留旧采集的源码身份，避免重采未改变的小输入路径。未匹配当前种子的证据仍明确标为历史。`host_profile_context`最多提供一份来源核验的宿主诊断，保留完整原件SHA。`report.py`把常规NPU达标与参考条件需复核的过线数值分开计数。

后续接续：新控制器v20已接手；45项控制器/分析检查通过。`xlogy_scalar_tensor`已完整30/30、补测72/72、1.253×；`hardshrink_backward`已完整30/30、补测45/45、0.868×，两项是本批新达标。`cross_out`通用维度、广播和out处理修订已21/21及45/45通过，目前0.769×；`expm1_`循环分块版完整30/30及63/63补测通过，0.462×。动态后续结果仍以主批次核验报告为准。

补测进程正常完成并返回失败JSON时，搜索结果现在保留正式KGS通过及补测失败两项事实，附`extra_checks`与`supplementary_status`；缺失输出、源码身份不符等故障继续报错。检查器实际快照与复现命令已在真机补测中生成。cross_out补测增加默认轴、广播、非连续out、输入与out别名；原地逐元素检查增加相同shape下的指针对齐变化。
