#!/usr/bin/env python3
"""在现有 KGS 上用同一评测合同对照 FlagGems 原实现与固定候选。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path


RUNS = {
    "mse_loss_backward": "mse-loss-backward-noprofile-63c4721c",
    "_prelu_kernel_backward": "prelu-underscore-phase7",
    "t_copy": "t-copy-profile-20261007",
    "batch_norm_backward": "batch-norm-adapter-phase6",
    "smooth_l1_loss_backward": "smooth-l1-profile-20261007c",
}
FILES = {
    "mse_loss_backward": "src/flag_gems/ops/mse_loss_backward.py",
    "_prelu_kernel_backward": "src/flag_gems/ops/_prelu_kernel_backward.py",
    "t_copy": "src/flag_gems/ops/t_copy.py",
    "batch_norm_backward": "src/flag_gems/ops/batch_norm.py",
    "smooth_l1_loss_backward": "src/flag_gems/ops/smooth_l1_loss.py",
}
UPSTREAM = "4772d816bc5d52d37c4718f36adf377d81261f83"


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def http(server: str, route: str, payload: dict | bytes | None = None,
         method: str = "POST", timeout: int = 1620) -> dict:
    data = payload if isinstance(payload, bytes) else (
        json.dumps(payload).encode() if payload is not None else None
    )
    headers = {"Content-Type": "application/x-tar" if isinstance(payload, bytes)
               else "application/json"}
    request = urllib.request.Request(server + route, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if isinstance(result, dict) and result.get("operation_id") and "per_workload" not in result:
        operation_id = result["operation_id"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = http(server, "/operations/" + operation_id, method="GET", timeout=60)
            if result.get("result") is not None:
                return result["result"]
            if result.get("status") in {"failed", "cancelled"}:
                return result
            time.sleep(5)
        raise TimeoutError(f"operation {operation_id}")
    return result


def upload_definition(server: str, catalog: Path, operator: str, output: Path,
                      flaggems: Path, head: str) -> dict:
    definition = (catalog / "definitions" / (operator + ".json")).read_bytes()
    manifest = json.loads((catalog / "manifest.json").read_text())
    source = manifest["definition_source"]
    save(output / "historical-definition-source.json", source)
    # 历史合同绑定旧 checkout；保留 ABI，明确绑定本轮实际使用的测试源码。
    source = {**source, "source_revision": head,
              "source_files": {name: digest((flaggems / name).read_bytes())
                               for name in source["source_files"]}}
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, data in [("definition.json", definition),
                           ("adapter.json", json.dumps(source, sort_keys=True).encode())]:
            item = tarfile.TarInfo(name)
            item.size = len(data)
            item.mode = 0o644
            tar.addfile(item, io.BytesIO(data))
    data = archive.getvalue()
    (output / "definition.json").write_bytes(definition)
    save(output / "definition-source.json", source)
    (output / "operator-bundle.tar").write_bytes(data)
    response = http(server, "/operator-bundles/" + digest(data), data, "PUT")
    save(output / "bundle-response.json", response)
    return {"bundle_id": response["bundle_id"], "definition": operator}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/hanle/ascend-optimization"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:19654")
    parser.add_argument("--operators", nargs="+", choices=list(RUNS), default=list(RUNS))
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出目录已存在，请指定新的目录")
    args.output.mkdir(parents=True)
    flaggems = args.root / "FlagGems"
    head = subprocess.check_output(["git", "-C", str(flaggems), "rev-parse", "HEAD"], text=True).strip()
    save(args.output / "environment.json", {
        "flaggems_head": head, "upstream_locked": UPSTREAM, "server": args.server,
        "server_status": http(args.server, "/status", method="GET", timeout=60),
        "source_resolution": "flaggems-active-functions-20261009.txt",
        "timing_comparison": "FlagGems 默认 kernel 模式；加速比为 PyTorch 设备侧延迟 / 被测实现设备侧延迟",
        "timing_scope": "Ascend do_bench_npu，默认 warmup=5、active=30；KGS timing=walltime 不改变 adapter 的 kernel 模式；请求的 warmup_ms/benchmark_ms 未用于该计时分支",
    })
    metadata = args.root / "runtime/kg-controller/flaggems-active-functions-20261009.txt"
    if metadata.exists():
        (args.output / metadata.name).write_bytes(metadata.read_bytes())
    summary = []
    for operator in args.operators:
        output = args.output / operator
        output.mkdir()
        run = args.root / "runtime/kg-controller/runs" / RUNS[operator]
        config = json.loads((run / ".kernelgen/run-request.json").read_text())["workflow_input"]
        optimization = config["optimization"]
        settings = {
            "warmup_ms": optimization["warmup_ms"],
            "benchmark_ms": optimization["benchmark_ms"],
            "num_trials": optimization["num_trials"],
            "timeout_seconds": optimization["eval_timeout_seconds"],
        }
        if config.get("catalog_path"):
            binding = upload_definition(args.server, Path(config["catalog_path"]), operator, output, flaggems, head)
        else:
            binding = {"catalog_name": optimization["catalog_name"], "definition": operator}
        save(output / "inspect.json", http(args.server, "/inspect", {
            "api_version": "v6.2", "binding": binding}, timeout=300))
        candidate_path = next(run.glob("stages/optimize/work/**/evals/round-0001/main.py"))
        candidate = candidate_path.read_text()
        original = (flaggems / FILES[operator]).read_text()
        # 仅增加 KGS 要求的入口；不修改原始内核或 host 实现。
        if operator == "_prelu_kernel_backward":
            original += (
                "\n\n_flaggems_original_prelu = _prelu_kernel_backward\n"
                "def run(grad_output, self, weight):\n"
                "    return _flaggems_original_prelu(grad_output, self, weight)\n"
                "_prelu_kernel_backward = run\n"
            )
        else:
            original += "\n\nrun = " + operator + "\n"
        (output / "flaggems-original.py").write_text(original, encoding="utf-8")
        (output / "candidate.py").write_text(candidate, encoding="utf-8")
        upstream = subprocess.run(["git", "-C", str(flaggems), "show", UPSTREAM + ":" + FILES[operator]],
                                  capture_output=True)
        if upstream.returncode == 0:
            (output / "flaggems-upstream.py").write_bytes(upstream.stdout)
        old_result = json.loads((candidate_path.parent / "result.json").read_text())
        save(output / "historical-result.json", old_result)
        save(output / "provenance.json", {
            "operator": operator, "historical_run": RUNS[operator], "candidate_path": str(candidate_path),
            "source_file": FILES[operator], "flaggems_head": head,
            "flaggems_file_sha256": digest((flaggems / FILES[operator]).read_bytes()),
            "upstream_file_sha256": digest(upstream.stdout) if upstream.returncode == 0 else None,
            "candidate_sha256": digest(candidate.encode()), "binding": binding, "settings": settings,
            "baseline_adapter": "只增加 run 入口；PReLU 原函数保留私有别名，公共符号指向同一具名签名包装，满足 KGS loader 要求",
        })
        for repeat in range(1, args.repeats + 1):
            for label, source in [("flaggems", original), ("candidate", candidate)]:
                name = f"{label}-{repeat}"
                request = {
                    "api_version": "v6.2", "binding": binding,
                    "implementation": {"name": operator + "-" + name, "definition": operator,
                                       "language": "triton", "entrypoint": "main.py::run",
                                       "sources": [{"path": "main.py", "content": source}]},
                    "settings": settings,
                }
                save(output / (name + ".request.json"), request)
                print("START", operator, name, flush=True)
                try:
                    result = http(args.server, "/evaluate", request, timeout=settings["timeout_seconds"] + 120)
                except urllib.error.HTTPError as exc:
                    result = {"status": "HTTP_ERROR", "http_status": exc.code, "detail": exc.read().decode()}
                except Exception as exc:
                    result = {"status": "CLIENT_ERROR", "detail": repr(exc)}
                save(output / (name + ".result.json"), result)
                row = {"operator": operator, "variant": label, "repeat": repeat,
                       **{key: result.get(key) for key in ["status", "geo_mean", "min_speedup", "device", "num_passed", "num_workloads"]}}
                summary.append(row)
                save(args.output / "summary.json", summary)
                print("DONE", json.dumps(row), flush=True)
    print("COMPLETE", str(args.output), flush=True)


if __name__ == "__main__":
    main()
