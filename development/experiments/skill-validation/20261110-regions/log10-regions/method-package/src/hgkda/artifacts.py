"""统一输入身份、结果封装与不覆盖式输出。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = "hgkda.result/v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identity(path, *, kind="file"):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "kind": kind}


def envelope(operation, result, inputs=(), *, status="completed"):
    return {"schema_version": SCHEMA_VERSION, "operation": operation, "status": status,
            "inputs": list(inputs), "result": result}


def emit(value, output=None):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8", newline="\n") as stream:
            stream.write(text)
    return text


def unwrap(value):
    if value.get("schema_version") == SCHEMA_VERSION:
        if value.get("status") != "completed":
            raise ValueError("输入分析尚未完成")
        return value["result"]
    return value
