"""只读统计仓库与依赖边界，输出紧凑的工程整理清单。"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    names = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], check=True, stdout=subprocess.PIPE).stdout.decode("utf8").split(chr(0))
    groups = {}
    file_types = Counter()
    missing = []
    for name in names:
        if not name:
            continue
        parts = name.split("/")
        group = "/".join(parts[:4]) if parts[0] == "project" else parts[0]
        value = groups.setdefault(group, {"files": 0, "bytes": 0})
        value["files"] += 1
        path = repo / name
        if os.name == "nt":
            path = Path(chr(92) * 2 + "?" + chr(92) + str(path))
        try:
            value["bytes"] += path.stat().st_size
        except FileNotFoundError:
            missing.append(name)
        file_types[Path(name).suffix or "no_extension"] += 1
    result = {"schema_version": "1.0", "repo": str(repo), "groups": groups,
              "file_types": dict(file_types.most_common()), "missing_tracked": missing,
              "head": subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10), encoding="utf8", newline=chr(10))
    print(json.dumps({"tracked": sum(v["files"] for v in groups.values()), "bytes": sum(v["bytes"] for v in groups.values()),
                      "groups": groups, "missing": len(missing)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
