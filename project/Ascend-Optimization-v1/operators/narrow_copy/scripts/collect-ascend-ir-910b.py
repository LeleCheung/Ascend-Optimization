#!/usr/bin/env python3
"""从 910B 的 Triton cache 复制一个 narrow_copy 编译阶段样本。"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    root = Path("/root/.triton/cache")
    output = Path("/tmp/narrow-copy-ir")
    output.mkdir(parents=True, exist_ok=True)
    source = next(root.rglob("_narrow_copy_flat_kernel.ttir"))
    files = []
    for path in sorted(source.parent.glob("_narrow_copy_flat_kernel.*")):
        if path.suffix not in {".source", ".ttir", ".ttadapter", ".mlirbc", ".bcmlir", ".npubin", ".json"}:
            continue
        destination = output / path.name
        shutil.copyfile(path, destination)
        files.append({"name": path.name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                      "size_bytes": destination.stat().st_size})
    manifest = {"cache_directory": str(source.parent), "files": files,
                "note": "容器内 Triton cache 的一个 narrow_copy flat kernel specialization；不是源码仓库的构建依赖"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
