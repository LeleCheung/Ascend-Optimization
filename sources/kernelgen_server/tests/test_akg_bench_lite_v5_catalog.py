from pathlib import Path

import pytest

from kernelgen_server import Catalog


def test_v5_akg_bench_lite_catalog_is_rejected_by_pure_v6_server():
    root = Path(__file__).parents[1] / "data" / ".old" / "akg-bench-lite-v5"
    with pytest.raises(ValueError, match="unsupported catalog API version"):
        Catalog(root)
