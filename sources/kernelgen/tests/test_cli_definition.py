"""CLI exports the real source fixture through the existing Definition workflow."""

import json
from pathlib import Path

import pytest

from kernelgen.cli import api
from kernelgen.cli.main import main
from kernelgen.tests.test_gems_adapter_definition import source as source


@pytest.fixture(autouse=True)
def isolated_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))


def test_cli_exports_real_definition_and_reuses_identical_output(source, tmp_path, capsys):
    workspace = tmp_path / "export"
    args = ["definition", "--flaggems-repo", str(source),
            "--pytest-path", "tests/test_addmm_.py", "--workspace", str(workspace)]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["operator"] == "addmm_"
    assert result["workspace"] == str(workspace)
    assert Path(result["catalog_path"]) == workspace / "catalog"
    assert Path(result["definition_path"]).is_file()
    assert len(result["source_files"]) == 5
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == result
    assert not (tmp_path / "state").exists(), "export must not submit a run or acquire a Coder lease"


def test_default_workspace_and_explicit_operator(source, tmp_path, capsys):
    assert main(["definition", "--flaggems-repo", str(source),
                 "--pytest-path", str(source / "tests/test_addmm_extra.py"), "--operator", "addmm_"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert Path(result["workspace"]).parent == tmp_path / "state/definitions"
    assert Path(result["definition_path"]).is_file()
    assert result["operator"] == "addmm_"
    assert {p.name for p in (tmp_path / "state").iterdir()} == {"definitions"}


@pytest.mark.parametrize("failure", ["dirty", "foreign", "inside_source", "unknown"])
def test_invalid_source_preserves_checkout_without_output(source, tmp_path, capsys, failure):
    workspace = tmp_path / "export"
    pytest_path = "tests/test_addmm_.py"
    extra = []
    if failure == "dirty":
        (source / pytest_path).write_text("# user changes\n")
    elif failure == "foreign":
        pytest_path = str(tmp_path / "test_elsewhere.py")
    elif failure == "inside_source":
        workspace = source / "output"
    else:
        extra = ["--operator", "missing"]
    assert main(["definition", "--flaggems-repo", str(source), "--pytest-path", pytest_path,
                 "--workspace", str(workspace), *extra]) == 2
    captured = capsys.readouterr()
    assert not captured.out and "kg:" in captured.err
    assert not workspace.exists()
    if failure == "dirty":
        assert (source / "tests/test_addmm_.py").read_text() == "# user changes\n"


def test_non_git_source_is_a_cli_error(tmp_path, capsys):
    repo = tmp_path / "not-git"
    repo.mkdir()
    assert main(["definition", "--flaggems-repo", str(repo),
                 "--pytest-path", "tests/test_add.py"]) == 2
    assert "Gems Definition export failed" in capsys.readouterr().err
    assert not (tmp_path / "state").exists()


def test_default_workspace_uses_current_kernelgen_directory(source, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("KERNELGEN_CLI_HOME")
    assert main(["definition", "--flaggems-repo", str(source),
                 "--pytest-path", "tests/test_addmm_.py"]) == 0
    assert Path(json.loads(capsys.readouterr().out)["workspace"]).parent == tmp_path / ".kernelgen/definitions"


@pytest.mark.parametrize("args", [
    [], ["--flaggems-repo", "gems"], ["--pytest-path", "tests/test_add.py"],
    ["--flaggems-repo", "gems", "--pytest-path", "test.py", "--runtime", "codex"],
])
def test_required_inputs_and_no_model_options(args):
    with pytest.raises(SystemExit) as error:
        main(["definition", *args])
    assert error.value.code == 2


def test_cli_delegates_to_shared_api(tmp_path, monkeypatch, capsys):
    seen = []
    def export(values, *, workspace):
        seen.append((values, workspace))
        return {"catalog_path": "output/catalog"}
    monkeypatch.setattr(api, "export_gems_definition", export)
    assert main(["definition", "--flaggems-repo", "gems", "--pytest-path", "tests/test_add.py",
                 "--workspace", "output"]) == 0
    assert seen == [({"flaggems_repo": Path("gems"), "pytest_path": Path("tests/test_add.py"),
                     "operator": None}, Path("output"))]
    assert json.loads(capsys.readouterr().out) == {"catalog_path": "output/catalog"}
