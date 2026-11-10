#!/usr/bin/env python3
"""调用已安装的 HG-KDA 核心，不修改解释器路径或自动安装依赖。"""
from hgkda.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
