import subprocess
import sys


def test_client_import_does_not_import_torch():
    code = (
        "import sys; import kernelgen_server.client; "
        "assert 'torch' not in sys.modules; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "ok"
