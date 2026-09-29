"""Normal local KG imports require only the pure KGS client distribution."""

import os
import subprocess
import sys


def test_cli_and_current_workflows_do_not_import_server_package():
    script = """
import sys
import kernelgen.cli.main
import kernelgen.service.app
import kernelgen.workflows.gems_adapter_definition
import kernelgen.workflows.multiple_device_test
import kernelgen.workflows.optimization.artifacts
assert not any(name.startswith('kernelgen_server') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
    )
