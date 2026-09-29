from kernelgen_server.evaluation.adapters.flaggems.discovery import _discover_suite


def test_flash_attention_internal_operator_uses_distinct_pytest_marker(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    path = tests / "test_flash_attention.py"
    path.write_text(
        "import pytest\n@pytest.mark.underscore_flash_attention_forward\ndef test_op(): pass\n",
        encoding="utf-8",
    )

    paths, marker = _discover_suite(tmp_path, "tests", "_flash_attention_forward")

    assert paths == (path,)
    assert marker == "underscore_flash_attention_forward"
