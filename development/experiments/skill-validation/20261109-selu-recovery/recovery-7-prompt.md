加载skills/hgkda/SKILL.md，读当前candidate.py与host-profile-fast.json（按需前100行），只在workspace编辑，不读环境/历史源码。

最新0fed8d43398d正式36/36通过，相对CANN0.710539×，完整逐case在next-performance-feedback.json。大输入已恢复为FP16约45.5us、BF16约83.2us、FP32约87.2us；小输入仍约20.5–21.2us，参考约13.5–13.9us。因此本次只优化宿主缓存命中路径，不改任何设备kernel、分块选择或数值公式。

cProfile补测未插桩提交约19.60us；缓存命中每次仍经过run→_selu_inplace→_select_block→_launch_cached_flat→_debug_forced_fallback→_current_device_index→_do_launch。多次检查与计算不需要每次重复。cProfile数值包含插桩开销，只作函数定位，正式收益待KGS。

建议实现如下（你负责生成具体代码）：
1. 新增只存编译启动元数据的快速查找表，键为(device_index,dtype,numel)，值绑定已验证的bound launch、function、packed metadata、整数grid、masked和n等。不要保存Tensor/输入结果/旧stream。
2. 只在冷路径的编译器对象、debug/compile-only/msprof flags检查通过时填入快速表。编译metadata与launcher模式在该编译产物的生命周期固定；调试/profiling产物不放入快表。现有普通Triton回退继续保留。
3. run(A)直接处理缓存命中：核对连续性、当前输入dtype/numel/device、当前指针对齐、当前设备；读取当前raw stream，然后直接用已验证ABI调用bound launch一次。不再重新_select_block，不走多层辅助函数，不逐次对同一编译metadata做8个getattr。
4. 可在模块初始化缓存torch_npu._C._npu_getDevice等“查询函数对象”，每次调用读取当前设备，不缓存设备或stream的动态返回值。若C接口不存在用公共current_device函数回退。原始current stream查询仍使用与本机Triton后端一致的NoWait优先规则。
5. 新指针和stream必须每次填写；不要共享/修改可变参数数组造成并发污染。可用局部参数或不可变模板。快表未命中、未对齐、非连续、设备不符等进入原有可靠路径。输入数据的首次原地计算只能执行一次，不得异常后重新发射。
6. 仅run公开，其他Python函数单下划线。所有设备计算和回退严格Triton，无CANN/ATen目标算子。缓存元数据/调用Triton编译启动器合法。

不要再反复改写旧diagnosis.json。新写一个很短的iteration-diagnosis.json，仅包含evidence、mechanism、change、prediction、validation、unresolved六项。编辑完立即按task.json里的准确tool_actions.preflight命令执行，timeout=600000。42/42后返回，宿主继续源码审查、offset/stream专项及正式KGS；不在正确性通过处宣布goal完成。
