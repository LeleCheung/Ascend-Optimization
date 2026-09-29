"""Translate the selected KGS backend to FlagGems' vendor vocabulary."""


def flaggems_vendor(backend: str) -> str:
    return {
        "cuda": "nvidia",
        "npu": "ascend",
        "musa": "mthreads",
        "mlu": "cambricon",
    }.get(backend, backend)
