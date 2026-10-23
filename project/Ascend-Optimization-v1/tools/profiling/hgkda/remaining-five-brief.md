# 会议剩余五算子连续达标任务

目标是正确性完整通过且相对PyTorch的完整调用walltime超过0.8×，不是只生成分析报告。只使用固定FlagGems/FlagTree环境、物理卡7和KGS19659；不要升级环境、重启服务、调用其他卡或绕过评测。无需阶段结束询问是否继续。

## 执行与证据

- 必须实现真实通用算子，不针对case序号、随机种子、输入数值或benchmark环境硬编码。不得调用被优化的PyTorch/FlagGems原算子代替候选；原始输入不得修改，返回结构、dtype、布局和别名符合接口。
- 一份源码必须在新形状/布局、当前stream及全部冻结timing输入上验数值。Native timing本身可能只计时不核对输出，不能把搜索高分当正确交付。
- 使用真实trace、IR和metrics形成假设；修改后用实测检验。Pipe活跃比例不是峰值利用率，不建立未经校准的Roofline。
- 首批每项4轮、最多50分钟，第二批未达标项3轮、最多40分钟。单次profiling最多240秒。每份源码一次独立完整KGS；按同case候选真实延迟选版，不用参考分数波动挑版。
- profiler失败可以先检查launcher元数据、当前Tensor与stream，不能切换计算kernel来伪装采集。记录不支持与超时，不把缺失报告解释成零瓶颈。

## upsample_bilinear2d

原版24项中6项编译失败，全部是输出1×1、三种dtype和两种align_corners。代码使用二维广播、runtime OW/OH取模和嵌套N/C循环；先尝试一维扁平输出tile映射N/C/H/W，将形状和坐标选项设为constexpr。align_corners=True且输出维度1时坐标0，False时为输入中心坐标，不能直接返回左上角。尊重scales_h/scales_w、非连续输入、边界和插值浮点舍入。先修编译再优化N*C并行和连续写回。

## _conj_physical

本批是aten._conj_physical，不是public conj_physical。实数也必须返回新张量，不能直接返回输入；保留dense输入的布局，非dense按参考规则物化。复数要按逻辑值共轭，处理lazy conjugate bit、负stride不可假设支持；使用实部/虚部读取的真实存储和符号，不能对已有conj bit重复取反。原版依赖FlagGems copy和launch_conj_physical，可评估融合为一次读写。当前环境complex参考可能CPU回退，必须据trace标记；不可宣称回退加速代表原生NPU kernel优势。补充实数和复数、转置、stride2、offset、空输入及输出独立性检查。

## scatter_add

从此前完整通过的0.419×候选开始。当前末轴owner方案对每个输出块遍历整行source，比较复杂度接近O(width²)；大输入仍用逐lane原子写，实际IR已证实性能差。禁止重复盲目增大tile。优先研究按目标分区的局部归约、分桶/排序的总成本，或用受支持的原子向量实现；检查TLE/IR方案是否真正在本机可用，不升级固定FlagTree。FP16/BF16须与参考一致使用正确累加语义，不能每次原子都舍入低精度；先用CPU/NPU小例证明算法。边界包含空index、非末轴、输入/index/src不同stride和重复索引。优化必须包括初始化、临时张量和所有launch。

## cumulative_trapezoid

先修低精度长序列正确性。CPU表达式cumsum((left+right)*(dx/2))匹配CPU；NPU参考任务为Slice/Add/Muls/Cumsum。第二批设备反例证明FP16增量直接cumsum与参考一致，但同增量先升FP32再cumsum再降精度不同。不能继续声称“保留增量舍入+FP32扫描就必然正确”。应拆开验证增量生成、输入dtype对NPU cumsum算法/结合顺序的影响，再选择匹配的扫描策略；必要时查看torch_npu/CANN调用路径和IR。原第二批候选扩展16/18，性能数值仅2/9、2/9、3/9，无交付。保留非二次幂dx、长度4097、非末轴和转置。不得改容差或替换参考来让候选通过。

## fractional_max_pool3d

先核实PyTorch参考的CPU回退范围、同步和拷贝，完整计时须如实标注，不能把超过0.8×直接解读为原生NPU优势。输出同时包含最大值和argmax索引；输入是5D/4D，采样参数和窗口起点规则必须吻合，支持return_indices=False。低精度_random_samples不能假设FP32，保留tie、NaN/Inf和out_size=1边界。优化方向包括D/H/W tile组织、消除重复窗口地址计算与采样读取，但每项先验证完整数值。若原生参考不可用，保留原PyTorch含回退计时，另行注明有效NPU对照缺失；不得静默改成自己构造的更慢reference。

## 知识与贡献

CC/DeepSeek完成KG实际生成及工具调用，Codex提供先验、额外机制研究、门禁、控制及必要修复；记录实际加载角色、源码SHA、完整结果和失败，不把共同结果写成纯原生KG或规则脚本独立优化。有效经验和失败约束服务后续轮次，不能把未经验证的建议当成成功知识。
