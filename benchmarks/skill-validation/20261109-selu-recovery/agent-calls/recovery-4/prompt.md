继续遵循skills/hgkda/SKILL.md。不要再读环境或IR，不要执行grep、Python探测。Read当前candidate.py，直接修复_compile_flat中的`grid = (triton.cdiv(n_elements, block),)`为`grid = triton.cdiv(n_elements, block)`。这个grid只缓存给直接C启动器；C接口前三项均为int，不能传tuple。warmup的grid=(1,)仍保留；普通Triton路径的grid lambda仍保留。后端源码已核验前缀为iiiKKOOOO，后面按当前签名保留两个constexpr槽位，不增加或猜测槽位。

这是宿主明确的ABI兼容修复建议，请执行Edit后立即调用下述工具桥preflight --source 当前candidate.py绝对路径，timeout=600000。只改这一行和诊断说明，不改设备kernel、缓存语义、其他宿主路径。不读取外部文件。预检通过后简短返回，宿主将审查并正式KGS。该返回仅本次模型阶段结束，不是goal结束。

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/.queued/selu-skill-recovery-20261109-v2/package/tools/skill_validation/agent_tools.py --config /data/hanle/ascend-optimization/runs/selu-skill-recovery-20261109-v2/config.json