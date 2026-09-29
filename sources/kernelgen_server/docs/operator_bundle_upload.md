# 用户算子上传接口

KGS 支持将用户提供的 v6.2 Native 算子或 v6.0 Gems adapter Definition 上传到内容寻址存储，不修改官方 Catalog。KG 的本地 Catalog 输入复用此接口；`kg run --operator-dir` 快捷参数仍未实现。

## Bundle 格式

输入目录直接包含一个算子的资产：

```text
my_operator/
  definition.json
  oracle.py
  correctness.jsonl
  timing.jsonl
  assets/
```

`definition.json` 必须显式声明 `api_version=v6.2`。`correctness.jsonl` 和 `timing.jsonl` 至少有一个包含 workload；其余结构和资产约束与 v6.2 native Catalog 相同。

客户端将目录内容打包为确定性的未压缩 tar，以 tar 字节的 SHA-256 作为资源地址。上传请求使用 `application/vnd.kernelgen.operator-bundle.v1+tar`；KGS 也接受标准的 `application/x-tar`。

## Gems adapter Definition

Gems Bundle 只能包含 `definition.json`（纯 v6.0 ABI）和 `adapter.json`（来源记录），不能上传 oracle、workload 或执行代码。`adapter.json` 包含 `evaluator="flaggems"`、`benchmark_level="core"`、`source_revision`（40 位 Git commit）及 `source_files`（仓库相对路径到 SHA-256 的映射）。正确性和性能 pytest 全部来自目标 Gems checkout，不由上传者用 Native 测例替代。

安装后的布局为 `definitions/<name>.json` 与服务端生成的 flat manifest。执行时 Adapter 检查源 commit、原始 correctness/benchmark suite 覆盖及文件摘要，再沿用原来的 pytest override、preflight、benchmark、profile、取消和设备 slot 路径。来源记录只绑定这份上传的 Definition，不恢复对所有已安装 Gems adapter Catalog 的统一版本限制。源 pytest 更新后重新导出 Definition，而不是篡改旧 Bundle。

客户端必须在上传前确认 `/status.capabilities.operator_bundle_upload.evaluators` 包含 `flaggems`，并且 `enabled=true`、`evaluation_binding=true`；旧服务没有该 evaluator 能力时不能把 Definition 伪装成 Native Bundle。

默认 Gems 来源为官方 `flagos-ai/FlagGems` 的 `kernelgen-dev`，`compatibility.yaml` 声明 branch policy；KG 在新实验的服务准备阶段解析并记录实际 commit，不更新运行中的服务。分支可定期合并 master，更新实例后需重新导出输入并验证；已有 Bundle 仍校验其冻结来源，不能因分支名相同就接受不同 commit。详见 [Gems 来源策略](flaggems_source_policy.md)。

## HTTP 上传与查询

```http
HEAD /operator-bundles/{sha256}
GET  /operator-bundles/{sha256}
PUT  /operator-bundles/{sha256}
```

`HEAD` 用于判断缓存是否命中，`GET` 返回已安装 Bundle 的元数据，`PUT` 上传 tar。首次安装返回 `201`，相同摘要再次上传返回同一份元数据和 `200`。

```json
{
  "bundle_id": "sha256:0123...cdef",
  "sha256": "0123...cdef",
  "definition": "my_operator",
  "archive_format": "kernelgen.operator-bundle/v1",
  "size_bytes": 10240,
  "created_at": "2026-09-05T00:00:00+00:00"
}
```

KGS 在写入前校验请求摘要、上传大小、归档路径、文件类型和对应 Catalog 契约；软链接、硬链接、特殊文件、父目录穿越和不完整算子都会被拒绝。Native 校验通过后以原子 rename 安装为：

```text
<operator-bundle-root>/<sha256>/
  manifest.json
  .bundle.json
  ops/<Definition.name>/
```

Server 默认保存到 `/tmp/kernelgen_server_operator_bundles`，可通过 `--operator-bundle-root` 或 `KGS_OPERATOR_BUNDLE_ROOT` 修改。单个 Bundle 默认最多 256 MiB，可通过 `--operator-bundle-max-bytes` 或 `KGS_OPERATOR_BUNDLE_MAX_BYTES` 修改。

Python 客户端会先查询摘要，只有未命中时才上传：

```python
from kernelgen_server.client import upload_operator_bundle

bundle = upload_operator_bundle("./my_operator", "http://127.0.0.1:8000")
print(bundle.bundle_id)
```

`/status.capabilities.operator_bundle_upload` 描述归档格式、大小和条目数限制。支持执行绑定的 Server 返回 `evaluation_binding=true`；客户端必须检查该能力，不根据 release 推断。旧 Server 返回 false 时仅能上传和查询，不得开始评测。TTL/lease 清理仍未实现。

## 执行已上传的算子

`inspect/preflight/evaluate/profile` 共用 `EvaluatorBinding`，必须且只能选择 `catalog_name` 或 `bundle_id`，并指定 Bundle 内的准确 `definition`：

```json
{"binding": {"bundle_id": "sha256:<64位小写摘要>", "definition": "my_operator"}}
```

`catalog_name` 的原有请求保持兼容。Bundle 仅在当前 Server 配置的存储根内解析，不接受客户端文件路径；不存在返回 404，definition 不匹配返回 422，均在领取设备 slot 前拒绝。Preflight/Eval 的隔离 worker 和 Profile 按 evaluator 复用原 Native 或 Gems adapter、取消和 slot 恢复逻辑，不把 Bundle 注册为内置 Catalog，也不修改 Server 源码或环境变量。
