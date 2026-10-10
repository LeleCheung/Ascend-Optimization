#!/usr/bin/env python3
"""经 KGS 设备队列执行诊断脚本，保存请求、输出和下载的原始产物。"""
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from evaluate_operator_import import call, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--file", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--server", default="http://127.0.0.1:19655")
    p.add_argument("--timeout", type=int, default=600)
    a = p.parse_args()
    if not 1 <= a.timeout <= 1800:
        p.error("KGS debug 超时范围为 1～1800 秒")
    if a.output.exists():
        p.error("输出已存在，请使用新批次")
    a.output.mkdir(parents=True)
    files = [a.script, *a.file]
    if len({x.name for x in files}) != len(files):
        p.error("脚本与附加文件名称不能冲突")
    req = {"command": ["{python}", a.script.name], "timeout_seconds": a.timeout,
           "files": [{"path": x.name, "content": x.read_text(encoding="utf-8")} for x in files]}
    save(a.output / "request.json", req)
    # KGS 6.5.0 的 create 会等待整个诊断执行结束，并非立即返回 job_id。
    # 包含排队余量；超时后保留请求，不能自动重发以免重复占用设备。
    try:
        response = call(a.server, "/debug/jobs", req, a.timeout + 1800)
    except urllib.error.HTTPError as error:
        save(a.output / "http-error.json", {
            "http_status": error.code,
            "detail": error.read().decode(errors="replace"),
        })
        raise SystemExit("KGS 拒绝诊断请求，详见 http-error.json") from error
    save(a.output / "created.json", response)
    job_id = response["job_id"]
    print("DEBUG", job_id, flush=True)
    while response.get("status") in {"PENDING", "QUEUED", "RUNNING"}:
        time.sleep(5)
        response = call(a.server, "/debug/jobs/" + job_id, timeout=60)
    save(a.output / "response.json", response)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for item in response.get("artifacts", []):
        url = item["download_url"]
        if not url.startswith("/debug/jobs/" + job_id + "/artifacts/"):
            raise ValueError("unexpected artifact URL")
        with opener.open(a.server + url, timeout=120) as stream:
            data = stream.read()
        if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("debug artifact size/SHA mismatch")
        path = a.output / "artifacts" / item["path"]
        if not path.resolve().is_relative_to((a.output / "artifacts").resolve()):
            raise ValueError("unsafe artifact path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(response.get("stdout", ""), flush=True)
    print(response.get("stderr", "")[-3000:], flush=True)
    print("DONE", response.get("status"), response.get("error"), flush=True)
    if response.get("status") != "SUCCEEDED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
