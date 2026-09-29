from pathlib import Path

import pytest

from kernelgen_server import Catalog


def test_v5_flaggems_catalog_is_rejected_by_pure_v6_server():
    root = Path(__file__).parents[1] / "data" / ".old" / "flaggems-v5"
    with pytest.raises(ValueError, match="unsupported catalog API version"):
        Catalog(root)
