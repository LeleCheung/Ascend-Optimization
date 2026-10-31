# negative_：C++宿主启动包装实验

计算部分沿用`streaming-tiles-v4`的四个Triton kernel，已核对AST相同。新增的C++扩展只负责读取张量元数据、查找启动计划、组装参数和调用编译产物；不调用PyTorch负号运算，也不缓存张量数据或结果。

计划按元素数量、dtype和设备索引缓存。每次调用重新读取当前张量指针与当前stream；空输入、非连续和非16字节对齐输入沿用Python路径。原Python编译入口的debug/profiling保护继续生效，未登记直接启动计划时继续回退。

## 复现

先在目标容器的固定Python/Torch环境中构建仓库内源码，产物位于仓库外runtime；不修改共享安装：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export MAX_JOBS=1
/usr/local/python3.11.15/bin/python3.11 tools/profiling/hgkda/next_phase/build_host_bridge.py --output /data/hanle/ascend-optimization/runtime/next25-host-bridge-build
```

`load_host_bridge.py`在导入时校验C++源码、二进制SHA、Torch版本、架构和ABI。迁移项目路径时设置`HGKDA_PROJECT`；迁移构建目录时设置`HGKDA_HOST_BRIDGE_DIR`。构建环境和二进制身份见`host-bridge-build.json`，新机器应重新构建，不能直接复用不同ABI的产物。

提案由持卡锁的夜间控制器执行完整KGS与补测。补测另存`host_bridge_stats`，用于确认实际进入了C++快速路径。构建成功不代表算子正确或性能达标；成绩以assessment目录的正式结果为准。
