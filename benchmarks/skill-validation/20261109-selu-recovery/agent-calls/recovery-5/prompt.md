继续遵循skills/hgkda/SKILL.md。读取当前candidate.py及next-performance-feedback.json，只在workspace读写，不探索环境。上个候选36/36和双方42/42、offset9/9、stream24/24通过，但正式CANN仅0.312005×，仍需优化。

新证据：独立cProfile定位到torch_npu.npu.current_stream(dev)每次构造Stream并经过通用device解析。未插桩单独测量约19.29us。宿主已验证torch_npu._C._npu_getCurrentRawStream(index)存在，每次查询当前raw stream，默认及非默认stream都与高层.npu_stream完全一致，约0.50us。它只查询运行时元数据，不执行目标计算，符合严格Triton。必须用这个带同步语义的版本，禁止NoWait版本，禁止缓存返回的stream。

直接Edit当前candidate.py：连续快路径每次获取`stream = torch_npu._C._npu_getCurrentRawStream(dev.index)`，index无效时走已有普通Triton回退。缓存键可用(device_index,dtype,n_elements,block,masked)，因为这个核只依赖连续元素个数，无需每次构造shape tuple。减少重复宿主元数据解析，保留当前device检查和调试回退。可缓存bound launch方法及不可变元数据，禁止缓存Tensor/结果/旧stream。

其他设备kernel和分块参数先保持不变，ABI仍然前缀9项+显式参数+BLOCK/F16MATH两个constexpr占位，grid是标量。不要再重新构造或猜测ABI。Python辅助全单下划线、run唯一公开。更新diagnosis.json，写清cProfile有插桩开销，19.29/0.50是独立函数测量，正式整体收益待KGS验证。

编辑后立即执行下述工具桥preflight --source 当前candidate.py绝对路径，timeout=600000。不执行其他Python或外部探索命令。预检42/42后返回，宿主完整审查、专项检查、正式KGS并反馈性能。不声明整个goal完成。

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/.queued/selu-skill-recovery-20261109-v2/package/tools/skill_validation/agent_tools.py --config /data/hanle/ascend-optimization/runs/selu-skill-recovery-20261109-v2/config.json