# 两组共享的固定后端接口知识

固定环境：910B4-1、CANN9.0、FlagTree0.6.0+ascend.gitc286cba6、Triton3.5.1。先导入 torch、torch_npu，再导入 Triton，避免当前安装版本的循环导入问题。

以下为本机已核对的 JITFunction.run 提交语义，供两组同等使用。它只描述编译和调用接口，不给任何算子的最佳实现。

1. 正常调用 `compiled = kernel[grid](*args, **kwargs)` 返回已编译对象。若是 future，使用 `.result()`。
2. 本机 `kernel.device_caches[device]` 是 `(kernel_cache, kernel_key_cache, target, backend, binder)`。
3. 绑定前，选项执行 `kwargs['debug'] = kwargs.get('debug', kernel.debug) or knobs.runtime.debug`，然后 `triton.runtime.jit._flagprism.apply_compile_options(kwargs)`；`binder(*args, **kwargs)` 返回 `(bound_args, specialization, options)`。直接查缓存必须与正常 JIT 采用同样的选项，否则可能命中不了真实编译产物。
4. `driver.active.get_current_device()`、`driver.active.get_current_stream(device)` 获取本次设备和当前 stream。
5. `grid(bound_args)` 或固定 tuple 得到实际 grid；不足三维用 1 补齐。
6. `metadata = compiled.launch_metadata(grid, stream, *bound_args.values())`。
7. 提交接口为 `compiled.run(gx, gy, gz, stream, compiled.function, compiled.packed_metadata, metadata, knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())`。
8. 所有输入指针和标量应来自本次调用。可以缓存编译对象与配置，不能缓存旧输入 Tensor、旧数据指针或固定 stream。新 dtype、shape、stride、设备、编译选项需要匹配正确的特化。
9. 直接调用是版本受限的优化途径，必须保留未知布局／版本的正常入口。禁止修改共享运行库、关闭正确性检查或缓存结果。

小输入、较大输入、连续与非连续输入都需要正确。grid 不超过 65535；容量受表达式和 dtype 影响，不假定 block 越大越好。不从活跃比例推导独占流水时间或定量 Roofline。
