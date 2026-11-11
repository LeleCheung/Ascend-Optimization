固定版本：FlagGems d6a8eec473517a3d68157b208eb9c057eb1d4c50；FlagTree 0.6.0+ascend.gitc286cba6；Triton 3.5.1；CANN 9.0.0；KGS 6.5.0。三类分别保留，倍数在单元格内注明参考；原30项合同达标26/30，独立数学合同单列。历史最佳为既有正式交付选中版本；FlagGems及早期设备计时与当前完整调用计时分开。

<h2 id="yfm-h2-3-1">关键算子（1015）（8/10 原合同达标）</h2>

<table>
<colgroup>
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
</colgroup>
<tr>
<td>
<p>算子</p>
</td>
<td>
<p>Excel原记录</p>
</td>
<td>
<p>FlagGems实测（相对PyTorch）</p>
</td>
<td>
<p>历史最佳（不符合当前约束）</p>
</td>
<td>
<p>历史最佳（严格Triton）</p>
</td>
<td>
<p>独立入口验证版（严格Triton）</p>
</td>
<td>
<p>完整评测与状态</p>
</td>
<td>
<p>关键方法</p>
</td>
<td>
<p>更新日期</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/selu_/candidate.py">selu_</a></p>
</td>
<td>
<p>精度失败</p>
</td>
<td>
<p>历史固定版0.119×</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>CANN 0.982×<br />PyTorch 0.922×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/selu_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.039×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/selu_/candidate.py">源码</a><br />Skill新生成：CANN 0.887×（达标）<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/skill-validation/20261109-selu-recovery/formal-5e9e4d482620/operators/selu_/candidate.py">新生成源码</a></p>
</td>
<td>
<p>36/36；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>缓存Triton启动器与宿主参数；对齐异步任务队列，按类型选择分块</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/rad2deg_/candidate.py">rad2deg_</a></p>
</td>
<td>
<p>精度失败</p>
</td>
<td>
<p>正确性失败，33/36</p>
</td>
<td>
<p>CANN 1.297×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/rad2deg_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.922×<br />PyTorch 1.119×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/rad2deg_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.834×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/rad2deg_/candidate.py">源码</a></p>
</td>
<td>
<p>36/36；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>分离编译与执行，避免冷启动重复计算，按真实 stride 原地写回</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/log10_/candidate.py">log10_</a></p>
</td>
<td>
<p>精度失败</p>
</td>
<td>
<p>正确性失败，33/36</p>
</td>
<td>
<p>CANN 1.306×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/log10_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.843×<br />PyTorch 0.806×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/log10_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.941×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/log10_/candidate.py">源码</a></p>
</td>
<td>
<p>36/36；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>处理特殊值与非连续布局，优化分块和启动路径</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/cumulative_trapezoid/candidate.py">cumulative_trapezoid</a></p>
</td>
<td>
<p>精度失败</p>
</td>
<td>
<p>历史0.363×；默认扩展正确性未通过</p>
</td>
<td>
<p>源码已归档；成绩待绑定<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/cumulative_trapezoid/candidate.py">源码</a></p>
</td>
<td>
<p>PyTorch 0.845×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/cumulative_trapezoid/candidate.py">源码</a></p>
</td>
<td>
<p>原合同未闭环<br />数学合同 CANN 1.012×（单列）<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/cumulative_trapezoid/candidate.py">候选源码（未验收）</a></p>
</td>
<td>
<p>原合同语义／正确性未闭环；独立数学合同通过</p>
</td>
<td>
<p>Triton 融合前处理与倍增扫描，调整舍入顺序；仍有 FP16 补测失败</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/regressions/20261109/uniform-native-reference/operators/uniform/candidate.py">uniform</a></p>
</td>
<td>
<p>精度通过，无有效加速比</p>
</td>
<td>
<p>0.141×；扩展正确性未通过</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>PyTorch 1.628×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/uniform/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 2.399×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/regressions/20261109/uniform-native-reference/operators/uniform/candidate.py">源码</a></p>
</td>
<td>
<p>18/18；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>修复 Philox 整数运算与状态更新，保证冷启动只执行一次</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/value_selecting_reduction_backward/candidate.py">value_selecting_reduction_backward</a></p>
</td>
<td>
<p>0.4935×</p>
</td>
<td>
<p>0.759×</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>CANN 1.470×<br />PyTorch 1.676×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/value_selecting_reduction_backward/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.349×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/value_selecting_reduction_backward/candidate.py">源码</a></p>
</td>
<td>
<p>96/96；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>融合补零与梯度回填，修复非连续布局</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/scatter_add/candidate.py">scatter_add</a></p>
</td>
<td>
<p>精度失败</p>
</td>
<td>
<p>0.258×；扩展正确性未通过</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>CANN 1.886×<br />PyTorch 1.880×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/scatter_add/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.892×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/scatter_add/candidate.py">源码</a></p>
</td>
<td>
<p>18/18；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>排序与分段累加，避免不同目标间的数值干扰</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/upsample_bilinear2d/candidate.py">upsample_bilinear2d</a></p>
</td>
<td>
<p>精度失败</p>
</td>
<td>
<p>编译／布局失败</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>CANN 1.147×<br />PyTorch 0.945×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/upsample_bilinear2d/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.142×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/upsample_bilinear2d/candidate.py">源码</a></p>
</td>
<td>
<p>24/24；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>连续读取相邻输入行，一次计算两行输出，修复重复写入</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fractional_max_pool3d/candidate.py">fractional_max_pool3d</a></p>
</td>
<td>
<p>0.0266×</p>
</td>
<td>
<p>正式9/9，扩展16/27；参考CPU回退</p>
</td>
<td>
<p>源码已归档；成绩待绑定<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/fractional_max_pool3d/candidate.py">源码</a></p>
</td>
<td>
<p>PyTorch 33.002×（参考含CPU回退）<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fractional_max_pool3d/candidate.py">源码</a></p>
</td>
<td>
<p>原合同未闭环<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fractional_max_pool3d/candidate.py">候选源码（未验收）</a></p>
</td>
<td>
<p>缺合格的同语义CANN参考；不计达标</p>
</td>
<td>
<p>连续输出分块，复用索引和采样数据，保持低精度窗口语义</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/_conj_physical/candidate.py">_conj_physical</a></p>
</td>
<td>
<p>精度失败</p>
</td>
<td>
<p>新接口正式／扩展正确性未通过；参考CPU回退</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>CANN 1.153×<br />PyTorch 45.279×（参考含CPU回退）<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/_conj_physical/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.234×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/_conj_physical/candidate.py">源码</a></p>
</td>
<td>
<p>14/14；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>Triton 处理复数实虚部及非连续布局，保留 lazy conjugate 语义</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
</table>

<h2 id="yfm-h2-7-1">关键算子（1029新增）（18/20 原合同达标）</h2>

<table>
<colgroup>
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
<col width="83" />
</colgroup>
<tr>
<td>
<p>算子</p>
</td>
<td>
<p>Excel原记录</p>
</td>
<td>
<p>FlagGems实测（相对PyTorch）</p>
</td>
<td>
<p>历史最佳（不符合当前约束）</p>
</td>
<td>
<p>历史最佳（严格Triton）</p>
</td>
<td>
<p>独立入口验证版（严格Triton）</p>
</td>
<td>
<p>完整评测与状态</p>
</td>
<td>
<p>关键方法</p>
</td>
<td>
<p>更新日期</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/regressions/20261109/negative-complete-block/operators/negative_/candidate.py">negative_</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>正确性失败 27/30</p>
</td>
<td>
<p>CANN 1.251×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/negative_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.814×<br />PyTorch 0.560×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/negative_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.806×<br />本轮修复候选<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/regressions/20261109/negative-complete-block/operators/negative_/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>按规模选择分块，整块去掩码，复用已编译的启动入口</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/xlogy_scalar_tensor/candidate.py">xlogy_scalar_tensor</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>0.148×</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>CANN 1.579×<br />PyTorch 1.234×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/xlogy_scalar_tensor/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.536×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/xlogy_scalar_tensor/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>融合乘法与对数，按标量零值特化，简化连续输入与启动</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/nan_to_num_/candidate.py">nan_to_num_</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>正确性失败 27/30</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>CANN 1.100×<br />PyTorch 0.876×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/nan_to_num_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.863×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/nan_to_num_/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>提前确定默认替换常量，合并饱和处理，多块处理减少分派</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/cross_out/candidate.py">cross_out</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>0.136×</p>
</td>
<td>
<p>CANN 4.057×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/cross_out/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 4.090×<br />PyTorch 3.757×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/cross_out/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 3.868×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/cross_out/candidate.py">源码</a></p>
</td>
<td>
<p>21/21；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>连续移位读取替代取模与 gather，修复动态维度及共享存储</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/expm1_/candidate.py">expm1_</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>正确性失败 27/30</p>
</td>
<td>
<p>源码已归档；成绩待绑定<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/expm1_/candidate.py">源码</a></p>
</td>
<td>
<p>PyTorch 0.647×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/expm1_/candidate.py">源码</a></p>
</td>
<td>
<p>原合同未闭环<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/expm1_/candidate.py">候选源码（未验收）</a></p>
</td>
<td>
<p>缺合格的同语义CANN参考；不计达标</p>
</td>
<td>
<p>小值采用级数，按类型选择计算精度与分块；性能仍不足</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/hardshrink_backward/candidate.py">hardshrink_backward</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>0.154×</p>
</td>
<td>
<p>CANN 0.894×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/hardshrink_backward/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.042×<br />PyTorch 1.037×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/hardshrink_backward/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.999×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/hardshrink_backward/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>流式分块，恢复梯度乘 0/1 的特殊值及负零语义</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/subtract_/candidate.py">subtract_</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>正确性失败 27/30</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>CANN 1.556×<br />PyTorch 1.131×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/subtract_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.659×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/subtract_/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>融合缩放与减法，简化连续输入启动，保留广播与原地语义</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/native_group_norm/candidate.py">native_group_norm</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>正确性失败 21/27</p>
</td>
<td>
<p>CANN 1.024×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/native_group_norm/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.987×<br />PyTorch 0.783×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/native_group_norm/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.004×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/native_group_norm/candidate.py">源码</a></p>
</td>
<td>
<p>27/27；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>单核融合分组统计与归一化，分块累加，减少分派与分配</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/log2_/candidate.py">log2_</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>正确性失败 27/30</p>
</td>
<td>
<p>CANN 1.055×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/log2_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.846×<br />PyTorch 0.634×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/log2_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.880×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/log2_/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>双块先读后写，按类型、规模及布局选择分块</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/transform_bias_rescale_qkv/candidate.py">transform_bias_rescale_qkv</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>0.004×</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>CANN 0.931×<br />PyTorch 1.043×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/transform_bias_rescale_qkv/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.941×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/transform_bias_rescale_qkv/candidate.py">源码</a></p>
</td>
<td>
<p>21/21；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>融合偏置、缩放与 QKV 重排，分块读写并简化宿主启动</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fix/candidate.py">fix</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>运行失败 0/30</p>
</td>
<td>
<p>CANN 1.197×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/fix/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.014×<br />PyTorch 0.735×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fix/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.969×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fix/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>绝对值取 floor 后恢复符号，多块流水处理与启动复用</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/round_out/candidate.py">round_out</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>正确性失败 24/30</p>
</td>
<td>
<p>CANN 1.430×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/round_out/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.829×<br />PyTorch 0.942×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/round_out/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.916×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/round_out/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>流式处理整块与尾块，修复可选 out 的分配和尺寸检查</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/regressions/20261109/hardswish-native-low-precision/operators/hardswish_out/candidate.py">hardswish_out</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>0.177×</p>
</td>
<td>
<p>CANN 1.395×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/hardswish_out/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.862×<br />PyTorch 1.050×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/hardswish_out/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.867×<br />本轮修复候选<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/regressions/20261109/hardswish-native-low-precision/operators/hardswish_out/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>流式融合计算，减少元数据查询与参数打包，保持 out 语义</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fake_quantize_per_tensor_affine_cachemask_out/candidate.py">fake_quantize_per_tensor_affine_cachemask_out</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>8.651×</p>
</td>
<td>
<p>CANN 6.975×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/fake_quantize_per_tensor_affine_cachemask_out/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 7.060×<br />PyTorch 9.623×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fake_quantize_per_tensor_affine_cachemask_out/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 8.155×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fake_quantize_per_tensor_affine_cachemask_out/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>保留量化与 mask 融合核，布局整理和回写改为 Triton</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/special_round_out/candidate.py">special_round_out</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>运行失败 0/30</p>
</td>
<td>
<p>CANN 1.261×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/special_round_out/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.946×<br />PyTorch 0.967×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/special_round_out/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.000×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/special_round_out/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>复用流式舍入核，恢复必需的 out 参数及输出检查</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/special_xlogy_/candidate.py">special_xlogy_</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>正确性失败 27/30</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>CANN 1.004×<br />PyTorch 0.846×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/special_xlogy_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.071×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/special_xlogy_/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>融合对数与乘法，扩大分块，按两侧真实 stride 原地写回</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/hardswish/candidate.py">hardswish</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>0.148×</p>
</td>
<td>
<p>CANN 1.061×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/hardswish/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.882×<br />PyTorch 0.731×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/hardswish/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.849×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/hardswish/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>每个 program 循环处理多个块，区分整块与尾块，精简启动</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fmod_tensor_/candidate.py">fmod_tensor_</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>运行失败 0/30</p>
</td>
<td>
<p>CANN 1.422×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/fmod_tensor_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.818×<br />PyTorch 0.729×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fmod_tensor_/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 0.841×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/fmod_tensor_/candidate.py">源码</a></p>
</td>
<td>
<p>30/30；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>按 program 数循环处理整块，单独保护尾块，保留余数符号</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/linalg_inv_ex/candidate.py">linalg_inv_ex</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>正确性失败 5/7</p>
</td>
<td>
<p>源码已归档；成绩待绑定<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/linalg_inv_ex/candidate.py">源码</a></p>
</td>
<td>
<p>PyTorch 77.743×（参考含CPU回退）<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/linalg_inv_ex/candidate.py">源码</a></p>
</td>
<td>
<p>原合同未闭环<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/linalg_inv_ex/candidate.py">候选源码（未验收）</a></p>
</td>
<td>
<p>缺合格的同语义CANN参考；不计达标</p>
</td>
<td>
<p>按列 LU 与三角求解，补偿乘加并修正掩码及倒数表达</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/batch_norm_with_update_functional/candidate.py">batch_norm_with_update_functional</a></p>
</td>
<td>
<p>未录入</p>
</td>
<td>
<p>100.912×</p>
</td>
<td>
<p>CANN 1.783×<br />不计严格 Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/batch_norm_with_update_functional/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.748×<br />PyTorch 23.563×<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/batch_norm_with_update_functional/candidate.py">源码</a></p>
</td>
<td>
<p>CANN 1.774×<br />既有源码回归<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/operators/batch_norm_with_update_functional/candidate.py">源码</a></p>
</td>
<td>
<p>27/27；双方补测通过<br />原合同达标</p>
</td>
<td>
<p>保留归一化与统计更新融合核，布局及状态复制改为 Triton</p>
</td>
<td>
<p>2026-11-09</p>
</td>
</tr>
</table>

<h2 id="yfm-h2-11-1">早期优化与方法验证</h2>

<table>
<colgroup>
<col width="68" />
<col width="68" />
<col width="68" />
<col width="68" />
<col width="68" />
<col width="68" />
<col width="68" />
<col width="68" />
<col width="68" />
<col width="68" />
<col width="68" />
</colgroup>
<tr>
<td>
<p>算子</p>
</td>
<td>
<p>Excel原记录</p>
</td>
<td>
<p>FlagGems实测（相对PyTorch）</p>
</td>
<td>
<p>历史报告（相对PyTorch）</p>
</td>
<td>
<p>历史报告（相对FlagGems）</p>
</td>
<td>
<p>历史最佳（不符合当前约束）</p>
</td>
<td>
<p>历史最佳（严格Triton）</p>
</td>
<td>
<p>独立入口验证版（严格Triton）</p>
</td>
<td>
<p>完整评测与历史状态</p>
</td>
<td>
<p>关键方法</p>
</td>
<td>
<p>记录日期</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/f63e6c30a0d8583ba092844e57ab90c0f6d0349f/project/Ascend-Optimization-v1/archive/comparisons/flaggems-comparison-20261009/measurements/mse_loss_backward/candidate.py">mse_loss_backward</a></p>
</td>
<td>
<p>1.1688×</p>
</td>
<td>
<p>1.11×，27/27</p>
</td>
<td>
<p>1.34×</p>
</td>
<td>
<p>1.20×，原版</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>未按当前约束复核</p>
</td>
<td>
<p>未执行本轮回归</p>
</td>
<td>
<p>正确性27/27；计时15/15<br />已交付；原版达标后进一步优化</p>
</td>
<td>
<p>扩大分块，连续输入采用简单寻址</p>
</td>
<td>
<p>截至2026-10-15</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/f63e6c30a0d8583ba092844e57ab90c0f6d0349f/project/Ascend-Optimization-v1/archive/comparisons/flaggems-comparison-20261009/measurements/t_copy/candidate.py">t_copy</a></p>
</td>
<td>
<p>3.0894×</p>
</td>
<td>
<p>1.96×，9/9</p>
</td>
<td>
<p>3.43×</p>
</td>
<td>
<p>1.73×，原版</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>未按当前约束复核</p>
</td>
<td>
<p>未执行本轮回归</p>
</td>
<td>
<p>正确性9/9；计时6/6<br />已交付；原版达标后进一步优化</p>
</td>
<td>
<p>按形状选择tile，一个program处理多个tile</p>
</td>
<td>
<p>截至2026-10-15</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/f63e6c30a0d8583ba092844e57ab90c0f6d0349f/project/Ascend-Optimization-v1/archive/comparisons/flaggems-repaired-comparison-20261009/validated-v5/measurements/_prelu_kernel_backward/candidate.py">prelu_kernel_backward</a></p>
</td>
<td>
<p>1.8152×</p>
</td>
<td>
<p>正确性失败，3/21；本机修复版0.49×</p>
</td>
<td>
<p>0.95×</p>
</td>
<td>
<p>1.97×，修复版</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>未按当前约束复核</p>
</td>
<td>
<p>未执行本轮回归</p>
</td>
<td>
<p>三轮均正确性21/21、计时9/9<br />已交付；原版正确性修复与优化</p>
</td>
<td>
<p>修复索引和梯度形状，融合梯度与归约，补偿累加误差</p>
</td>
<td>
<p>截至2026-10-15</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/f63e6c30a0d8583ba092844e57ab90c0f6d0349f/project/Ascend-Optimization-v1/archive/comparisons/flaggems-repaired-comparison-20261009/validated-v5/measurements/batch_norm_backward/candidate.py">batch_norm_backward</a></p>
</td>
<td>
<p>4.3885×</p>
</td>
<td>
<p>正确性失败，9/30；本机修复版2.32×</p>
</td>
<td>
<p>5.01×</p>
</td>
<td>
<p>2.19×，修复版</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>未按当前约束复核</p>
</td>
<td>
<p>未执行本轮回归</p>
</td>
<td>
<p>三轮均正确性30/30、计时15/15<br />已交付；原版正确性修复与优化</p>
</td>
<td>
<p>修复保存方差解释，按形状分流、融合及分阶段归约</p>
</td>
<td>
<p>截至2026-10-15</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/f63e6c30a0d8583ba092844e57ab90c0f6d0349f/project/Ascend-Optimization-v1/archive/comparisons/flaggems-repaired-comparison-20261009/validated-v5/measurements/smooth_l1_loss_backward/candidate.py">smooth_l1_loss_backward</a></p>
</td>
<td>
<p>2.2139×</p>
</td>
<td>
<p>正确性失败，290/291；本机修复版2.17×</p>
</td>
<td>
<p>3.48×</p>
</td>
<td>
<p>1.61×，修复版</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>未按当前约束复核</p>
</td>
<td>
<p>未执行本轮回归</p>
</td>
<td>
<p>三轮均正确性291/291、计时12/12<br />已交付；原版正确性修复与优化</p>
</td>
<td>
<p>修复beta=0边界，扩大分块，提前确定常量与分支</p>
</td>
<td>
<p>截至2026-10-15</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/tree/f63e6c30a0d8583ba092844e57ab90c0f6d0349f/project/Ascend-Optimization-v1/operators/logaddexp2">logaddexp2</a></p>
</td>
<td>
<p>0.3218×</p>
</td>
<td>
<p>0.231×</p>
</td>
<td>
<p>1.262×</p>
</td>
<td>
<p>—</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>未按当前约束复核</p>
</td>
<td>
<p>未执行本轮回归</p>
</td>
<td>
<p>36/36<br />已交付；历史最佳源码复现</p>
</td>
<td>
<p>使用数值稳定公式，简化连续输入寻址，一个program处理更多数据</p>
</td>
<td>
<p>2026-10-14</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/tree/f63e6c30a0d8583ba092844e57ab90c0f6d0349f/project/Ascend-Optimization-v1/operators/amin">amin</a></p>
</td>
<td>
<p>0.5536×</p>
</td>
<td>
<p>0.308×，启动兼容修复版</p>
</td>
<td>
<p>0.934×</p>
</td>
<td>
<p>—</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>未按当前约束复核</p>
</td>
<td>
<p>未执行本轮回归</p>
</td>
<td>
<p>27/27<br />已交付；历史交付</p>
</td>
<td>
<p>直接沿目标轴归约，减少重排列、转换和整除时的边界检查</p>
</td>
<td>
<p>2026-10-14</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/tree/f63e6c30a0d8583ba092844e57ab90c0f6d0349f/project/Ascend-Optimization-v1/operators/matmul_bias_activation">matmul_bias_activation</a></p>
</td>
<td>
<p>0.5408×</p>
</td>
<td>
<p>历史固定版0.439×</p>
</td>
<td>
<p>0.884×（2026-10-27演示）；历史：0.909×（2026-10-14）</p>
</td>
<td>
<p>—</p>
</td>
<td>
<p>无已核验记录</p>
</td>
<td>
<p>未按当前约束复核</p>
</td>
<td>
<p>未执行本轮回归</p>
</td>
<td>
<p>42/42；18＋24项补充<br />已交付；方法闭环复现；不同批次分别保留</p>
</td>
<td>
<p>相邻输出块复用输入，融合bias／激活；新演示验证共享B方案</p>
</td>
<td>
<p>2026-10-27</p>
</td>
</tr>
<tr>
<td>
<p><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/narrow_copy/candidate.py">narrow_copy</a></p>
</td>
<td>
<p>0.0138×</p>
</td>
<td>
<p>约0.010×，启动兼容修复版</p>
</td>
<td>
<p>1.118×</p>
</td>
<td>
<p>—</p>
</td>
<td>
<p>含CANN DMA；不计严格Triton<br /><a href="https://github.com/LeleCheung/Ascend-Optimization/blob/29a5d742539f44380a625f1d4518c760607ba054/benchmarks/lineage/20261109/historical_nonconforming/narrow_copy/candidate.py">源码</a></p>
</td>
<td>
<p>未按当前约束复核</p>
</td>
<td>
<p>未执行本轮回归</p>
</td>
<td>
<p>33/33<br />历史混合实现：含 CANN DMA，不计严格 Triton 成果</p>
</td>
<td>
<p>小型连续输入用CANN DMA，大输入／其他布局用Triton分块复制</p>
</td>
<td>
<p>2026-10-14</p>
</td>
</tr>
</table>
