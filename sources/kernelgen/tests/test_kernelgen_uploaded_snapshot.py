import pytest

from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot, snapshot_optimization_context
from kernelgen.workflows.optimization.kernelgen.contracts import KernelGenInput
from kernelgen.workflows.optimization.kernelgen.preparation import freeze_run_evaluation_contract
from kernelgen.workflows.optimization.kernelgen.epoch import build_epoch_inputs


def snapshot():
    return CatalogEvaluationSnapshot(
        catalog_name="uploaded-test", catalog_api_version="v6.2", definition_name="add",
        definition={"api_version":"v6.2","name":"add","parameters":[{"name":"x","required":True}],"outputs":["out"]},
        correctness_workloads=[{"name":"c","inputs":{"x":{"type":"scalar","value":1}}}],
        timing_workloads=[{"name":"t","inputs":{"x":{"type":"scalar","value":1}}}],
        bundle_id="sha256:"+"a"*64, benchmark_fingerprint="test-fingerprint",
    )


def test_uploaded_snapshot_reaches_all_coders_without_builtin_lookup(tmp_path, monkeypatch):
    from kernelgen.workflows.optimization.kernelgen import epoch
    monkeypatch.setattr(epoch,"resolve_builtin_catalog_path",lambda *_:pytest.fail("builtin lookup"))
    s=snapshot()
    definition,_=snapshot_optimization_context(s)
    inp=KernelGenInput(definition=definition,target_hardware="PPU-ZW810E",catalog_name=s.catalog_name,
        evaluation_snapshot=s.model_dump(mode="json"),n_parallel=2,warmup_ms=0,benchmark_ms=17,
        num_trials=2,eval_timeout_seconds=91,max_coder_sessions=1)
    prepared,frozen=freeze_run_evaluation_contract(tmp_path,inp)
    assert frozen==s.model_dump(mode="json")
    for coder in build_epoch_inputs(prepared,{},[],"unvalidated seed",1,evaluation_snapshot=frozen):
        assert coder['evaluation_snapshot']==frozen
        assert coder['seed_code']=='unvalidated seed'
        assert coder['seed_is_validated_baseline'] is False
        assert coder['warmup_ms']==0 and coder['benchmark_ms']==17 and coder['num_trials']==2
        assert coder['eval_timeout_seconds']==91 and coder['max_coder_sessions']==1
    assert freeze_run_evaluation_contract(tmp_path,inp)[1]==frozen
    changed=s.model_copy(update={'bundle_id':'sha256:'+'b'*64})
    with pytest.raises(ValueError,match='different request'):
        freeze_run_evaluation_contract(tmp_path,inp.model_copy(update={'evaluation_snapshot':changed.model_dump(mode='json')}))


def test_snapshot_identity_mismatch_is_rejected(tmp_path):
    s=snapshot()
    inp=KernelGenInput(definition={'name':'other'},target_hardware='PPU',catalog_name=s.catalog_name,
        evaluation_snapshot=s.model_dump(mode='json'))
    with pytest.raises(ValueError,match='identities differ'):
        freeze_run_evaluation_contract(tmp_path,inp)
