from pathlib import Path

import pytest

from kernelgen.tools.update_kernel_todo_validation import (
    read_operators,
    render_table,
)


def test_read_and_render_kernel_todo_table(tmp_path: Path):
    path = tmp_path / "device.md"
    path.write_text(
        "| 算子名称 |\n"
        "| --- |\n"
        "| gelu |\n"
        "| missing_api |\n",
        encoding="utf-8",
    )
    operators = read_operators(path)
    rendered = render_table(
        operators,
        {
            "gelu": {"status": "通过", "reason": ""},
            "missing_api": {
                "status": "未通过",
                "reason": "[API 不支持] vendor Torch 未实现",
            },
        },
    )

    assert operators == ["gelu", "missing_api"]
    assert "| gelu | 通过 | — |" in rendered
    assert "[API 不支持]" in rendered


def test_render_requires_classified_failures():
    with pytest.raises(ValueError, match="unclassified"):
        render_table(
            ["op"],
            {"op": {"status": "未通过", "reason": "unknown"}},
        )


def test_render_rejects_result_mismatch():
    with pytest.raises(ValueError, match="missing"):
        render_table([], {"extra": {"status": "通过", "reason": ""}})
