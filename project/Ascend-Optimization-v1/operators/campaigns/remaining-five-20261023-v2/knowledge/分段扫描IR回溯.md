# 分段扫描：错误已缩小到后续编译执行路径

独立小探针在32及512元素下均复现：段首输入正确，tuple扫描的最后一项应为0.25，实际为134217728；OR前缀应全部为1，实际仅首项为1。显式分段扫描输出正确。job与准确原件见`research/scatter_add/manual-segmented-v2-probe/`。

同一job的真实缓存已保存于该目录的`compiler-cache/`，逐文件SHA见`compiler-cache-manifest.json`。不是从其他算子或历史缓存推测。

前端TTIR完整保留了我们要求的二元组合：

```mlir
%sum = arith.addf %left_sum, %right_sum : f32
%value = arith.select %right_head, %right_sum, %sum : f32
%head = arith.ori %left_head, %right_head : i1
tt.scan.return %value, %head : f32, i1
```

随后`components.ttadapter`和`components.bcmlir`仍保留正确的逐项扫描逻辑：当前元素的head控制是否重置，前一位置的累加值与当前值相加，head做OR传播。这两阶段中尚未发现组合函数丢失。生成的扫描临时缓冲分别是`memref<Nxf32>`和`memref<Nxi1>`。

因此，现在可以排除“Python前端没有接收自定义组合函数”这一解释；错误范围已收敛至这些中间产物之后的编译或执行路径。`i1`标志的标量存取、转存和后续转换是可进一步区分的方向，尚不能指定某个pass或指令已被证明有错。当前有效修复采用显式gather分段扫描，不修改固定后端或共享环境。

后续如需提交最小复现，使用该job的`probe.py`中`components`部分；完整源文件与32/512输入输出、IR、SHA均已保留。它不包含排序，因此不依赖scatter的排序键或二分查找来触发问题。
