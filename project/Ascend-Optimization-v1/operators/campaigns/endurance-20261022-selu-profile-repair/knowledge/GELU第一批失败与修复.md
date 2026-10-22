# gelu_and_mul：第一批高分尚未通过FP32补充门禁

第一批四轮KG内测的完整19项均通过。第三轮内测1.8102×，但四份源码独立均12/18，失败集中全部六项FP32补充输入；15项原生BF16性能输入数值全通过。第四轮另在FP32宽8194输入编译UB overflow，要求1835008 bits，设备只有1572864 bits。不能把BF16合同通过扩大为三dtype的通用实现正确；也不能把当前高分当作有效交付。

优先复用第三轮连续按行/块读写、exact-GELU表达式融合及预绑定启动架构，不从原版persistent stride动态寻址重新测四轮：

results/gelu_and_mul/attempt-1/candidates/c479b35793c26d80c14c902785cc2b3f54d83ed989456ffa3c4feb52d5db43f7/main.py

原始semantics.json保存FP32偏差：随机输入最大约0.0013，大值探针约0.0044，超出1e-4容差。所有候选都用tl.erf，因此需要实际比较本机tl.erf、FP32参考F.gelu、准确FP32表达式的逐值误差。先确定误差来自erf降低、表达式收缩，还是NPU参考路径；不能只凭源码写了erf就宣称精度合格。严禁改容差或调用目标PyTorch gelu作为候选实现。

可补测一个FP32 erf有界误差近似作为明确候选（尚未在本机验证）：z=x/sqrt(2)，u=abs(z)，t=1/(1+0.3275911*u)，poly=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t，erf(z)=sign(z)*(1-poly*exp(-u*u))。标准该组系数对数学erf的最大绝对误差约1.5e-7，但本机编译、乘法收缩与GELU参考仍须实际逐值验证；正负大值和零附近要检查。可按dtype保留已通过的低精度快路径，并为FP32使用更准确表达式和更小分块；正式完整调用必须包含该分流。

第三轮launcher_key只有(dtype,hidden_size,device)，缺少输入地址对齐身份；第四轮已加入data_ptr()&15等完整标量身份，但不能因此照搬其8192分块。若保留第三轮启动缓存，应补齐对齐属性并只缓存编译产物，读取本次Tensor/stream，profiling用同kernel的标准元数据路径。独立check-gelu-launch-cache.py已加入先对齐预热再存储偏移的9项检查，可将候选放在本批目录后执行。检查不能与主卡锁队列并行发设备工作。

优先通过check-endurance.py的18项及全部15项性能输入数值，再做新的完整KGS评测；额外地址检查也必须通过。该笔记由Codex在第一批结束后形成，供第二批新overlay读取，不改写第一批模型贡献或原始成绩。

## 新的强线索：先核对NPU参考的实际公式

CPU同种子同输入的exact与tanh GELU差异，六项最大误差的位置全部与NPU失败回执吻合：case0 (0,0)、case1 (1,16)、case2 (4,426)、case3 (0,492)、case4 (1,13)、case5 (0,25)。最大值也接近：CPU约0.00011659、0.00023793、0.00124709、0.00124649、0.00136075、0.00442140；NPU失败约0.00011656、0.00023797、0.00124749、0.00124563、0.00136111、0.00441873。CPU显式erf表达式与CPU exact的差异只有约1e-7，不能据此将tl.erf归咎为低精度。

原始CPU见证在research/gelu_and_mul/exact-tanh-cpu-witness.json。它强烈指向本机NPU F.gelu(approximate='none')实际采用tanh型近似的可能，而非候选erf本身不准；仍需NPU逐值对照确认。第二批先以同一输入比较：CPU exact、CPU tanh、NPU F.gelu none、NPU F.gelu tanh、当前Triton候选，并保存真实任务和最大误差。这个探针应先于上文FP32 erf多项式替换，避免修错方向。

若证实固定NPU参考none实际等同tanh，候选可按该已证实语义对齐，保留明确公式和精度说明；不能把tanh声称为数学exact，也不能改oracle或容差。若不成立，再定位erf降低、浮点收缩和舍入。先前“tanh不能默认替代exact”的约束仍成立：需要这些实际对照证据才能替换。补充门禁与原生合同继续全部执行。

## 复用第一批KG分析角色已经提出的启动修复

第一批第3轮的结构化分析已定位：BLOCK=4096、EVEN=True的不同hidden size可共用同一JIT编译cache。当前源码只在fresh=set(cache)-known恰好出现一个新条目时登记_LAUNCHER，因此已经有编译产物的hidden size可能一直不登记，重复走慢入口。这个方法来自第一批实际KG分析角色，不是Codex重新发明。下一项实验建议如下：

正常调用_gelu_and_mul_rows_kernel[grid](...)后直接取得返回的CompiledKernel，按完整启动身份登记_LAUNCHER；删除known/fresh差集门槛。缓存键包含分块、EVEN、输入地址对齐及全部实际专门化属性，缓存只保存编译产物和元数据，保留profiling hook与正常路径回退。

版本和首次执行次数约束：先确认固定FlagTree的调用返回可用编译对象；不可用时保留正常计算，不因登记失败再执行一遍。完整数值、真实缓存命中和每次一个设备任务均需验证。原分析预期宽8192的四项由约95–100微秒降至约51–60微秒，这是待验证预测，不是已取得收益。

原始结构化分析：/data/hanle/ascend-optimization/runs/endurance-20261022/kg-runs/gelu_and_mul-endurance-20261022-attempt-1/stages/optimize/work/.kernelgen/profile-analysis/round-0003.json；SHA256：25f799e6f4c31bdb5f3eac3ead7ac7dc6e333e056fbd24df066532b684c998af。第二批优先实际验证该缓存登记修复，与FP32参考公式核对共同完成；不需要重新做一遍全部启动探针。
