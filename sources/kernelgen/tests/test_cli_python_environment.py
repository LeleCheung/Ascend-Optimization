import json
import os
import subprocess
import sys
import venv

import pytest

from kernelgen.cli import server


@pytest.mark.parametrize("symlinks", [True, False])
def test_python_resolution_preserves_selected_venv(tmp_path, monkeypatch, symlinks):
    root = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False, symlinks=symlinks).create(root)
    python = root / "bin/python"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(root / "bin") + os.pathsep + os.environ["PATH"])
    for value in (str(python), "environment/bin/python", "python"):
        selected = server._resolve_python(value)
        assert selected == str(python)
        result = subprocess.check_output([selected, "-c", "import sys,json; print(json.dumps([sys.prefix,sys.executable]))"], text=True)
        assert json.loads(result) == [str(root), str(python)]


def test_python_resolution_rejects_missing_executable(tmp_path):
    with pytest.raises(FileNotFoundError):
        server._resolve_python(str(tmp_path / "missing"))
