# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Backend device implementations for KernelGen Server.

Each sub-package implements a Device subclass for a specific hardware backend.
The registry in ``kernelgen_server.runtime.device`` lazily imports from here.

Structure:
    backends/
    ├── __init__.py          ← this file
    ├── nvidia/              ← NVIDIA CUDA (H800, A100, etc.)
    ├── ascend/              ← Huawei Ascend NPU (910B, 910C)
    ├── mthreads/            ← 摩尔线程 MTT (S5000, S4000)
    ├── hygon/               ← 海光 Hygon DCU (BW1000)
    ├── metax/               ← 沐曦 MetaX (C550)
    ├── iluvatar/            ← 天数智芯 Iluvatar (BI-V150)
    ├── cambricon/           ← 寒武纪 Cambricon (MLU590)
    ├── kunlunxin/           ← 百度昆仑芯 (P800)
    ├── thead/               ← 平头哥 T-Head PPU (ZW810E)
    ├── tsingmicro/          ← 清微智能 TXDA (TX8110)
    └── enflame/             ← 燧原 Enflame GCU

Adding a new chip: create a new sub-package with a Device subclass,
then register it in kernelgen_server/device.py::_make_device().
"""
