import pytest

pytest.importorskip("torch")

from kernelgen_server.evaluation.adapters.native import _format_preflight_errors


def test_native_preflight_log_keeps_case_ids():
    log = _format_preflight_errors(
        {
            "case-ok": {"status": "PASSED", "error": ""},
            "case-bad": {"status": "RUNTIME_ERROR", "error": "traceback"},
        }
    )

    assert log == "[case-bad]\ntraceback"


def test_profile_does_not_repeat_candidate_admission(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import kernelgen_server.evaluation.adapters.native as module
    adapter = object.__new__(module.NativeEvaluationAdapter)
    def unexpected(*args, **kwargs):
        raise AssertionError("admission belongs only to preflight")
    monkeypatch.setattr(module, "require_candidate_admission", unexpected, raising=False)
    class ReachedProfilePreparation(Exception):
        pass
    def preparation():
        raise ReachedProfilePreparation()
    monkeypatch.setattr(adapter, "inspect", preparation)
    with pytest.raises(ReachedProfilePreparation):
        adapter.build_profile_command(SimpleNamespace(), None, tmp_path, "cpu")
