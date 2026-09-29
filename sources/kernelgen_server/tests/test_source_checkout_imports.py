"""Both source-checkout import paths expose the same public KGS API."""

import json
import os
import subprocess
import sys
from pathlib import Path


def _public_api(path: Path, cwd: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", "import json, kernelgen_server as k; "
         "assert all(hasattr(k, name) for name in k.__all__); "
         "print(json.dumps({'file': k.__file__, 'exports': sorted(k.__all__)}))"],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(path)},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_source_parent_and_package_root_export_the_same_api(tmp_path):
    checkout = Path(__file__).resolve().parents[1]
    from_parent = _public_api(checkout.parent, tmp_path)
    from_package_root = _public_api(checkout, tmp_path)

    assert Path(from_parent["file"]).resolve() == checkout / "__init__.py"
    assert Path(from_package_root["file"]).resolve() == checkout / "kernelgen_server/__init__.py"
    assert from_parent["exports"] == from_package_root["exports"]
