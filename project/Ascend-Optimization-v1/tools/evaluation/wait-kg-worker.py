#!/usr/bin/env python3
"""等待原生 KG 后台任务结束，并返回真实退出码；适合 tmux 串行任务。"""
import json
import sys
import time
from pathlib import Path

from kernelgen.framework.local_state import process_is_alive

workspace = Path(sys.argv[1])
metadata = workspace / ".kernelgen/run-process.json"
log = workspace / ".kernelgen/runner.log"
offset = 0
while True:
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(offset)
            chunk = stream.read()
            offset = stream.tell()
        if chunk:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    record = json.loads(metadata.read_text())
    if record.get("finished_at"):
        raise SystemExit(record.get("exit_code") or 0)
    if not process_is_alive(record["pid"], record["process_start"]):
        raise SystemExit("KG worker 已退出但没有完成元数据，请检查日志")
    time.sleep(5)
