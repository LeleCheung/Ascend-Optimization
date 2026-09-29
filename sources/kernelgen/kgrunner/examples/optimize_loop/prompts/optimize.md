# Kernel 单轮优化任务

## 算子信息

- **算子名称**: {{OPERATOR}}

## 工作流程

### 1. 查看性能历史

读取当前目录下的 `PERFORMANCE.md`（如果存在），了解已经尝试过的优化方向和对应的 speedup。

如果存在 `versions/` 目录，可以查看历史版本的 kernel 代码（`versions/vN/kernel.py`）了解之前的实现。

**重要**：不要重复已经尝试过的优化方向。

### 2. 分析并选择一个优化方向

根据性能历史和当前 kernel 代码，选择**一个**新的优化方向实施。

### 3. 修改 kernel 代码

编辑 kernel 文件实施优化。

### 4. 正确性验证

运行测试：

```bash
python -m pytest tests/ -m {{OPERATOR}} -vs 2>&1
```

**必须全部 PASS**。如果不通过，反复修复直到通过。

### 5. 性能验证

运行 benchmark：

```bash
python -m pytest benchmark/ -m {{OPERATOR}} -vs 2>&1
```

记录 speedup 数据。

### 6. 输出结果

完成后输出以下 JSON（用 ```json 代码块包裹）：

```json
{
  "operator": "{{OPERATOR}}",
  "status": "success 或 failed",
  "speedup": 0.0,
  "test_passed": true/false,
  "kernel_path": "相对 workspace 的 kernel 文件路径",
  "optimization_direction": "描述本轮优化方向",
  "error": null
}
```

## 规则

- **每轮只做一个优化方向**，不论性能是否提升都必须停止
- **正确性可以反复修复**直到通过，但通过后不要再为了性能修改代码
- **不要修改 tests/ 或 benchmark/**
- **公共 API 签名（函数名、参数列表）不可修改**
- **必须输出 JSON 结果**，即使失败也要输出
- **禁止 pip install**
