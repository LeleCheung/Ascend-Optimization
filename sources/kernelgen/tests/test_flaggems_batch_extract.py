from pathlib import Path

import pytest

from kernelgen.examples.flaggems_extract.batch_extract import read_operators


def test_read_operators_accepts_plain_list(tmp_path: Path):
    path = tmp_path / "operators.txt"
    path.write_text("gelu\n\naddmm_\ngelu\n", encoding="utf-8")

    assert read_operators(path) == ["gelu", "addmm_"]


def test_read_operators_accepts_markdown_table(tmp_path: Path):
    path = tmp_path / "operators.md"
    path.write_text(
        "| 算子名称 | 状态 |\n"
        "| --- | --- |\n"
        "| gelu | pending |\n"
        "| addmm_ | pending |\n",
        encoding="utf-8",
    )

    assert read_operators(path) == ["gelu", "addmm_"]


def test_read_operators_rejects_shell_metacharacters(tmp_path: Path):
    path = tmp_path / "operators.txt"
    path.write_text("gelu; touch bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid operator name"):
        read_operators(path)
