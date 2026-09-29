import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelgen.data._atomic import atomic_write_json
from kernelgen.framework.run_control import WorkspaceRunControl, RunCancelled
from kernelgen.framework.worker_pool import WorkerLeasePool, set_max_workers
from kernelgen.workflows.optimization import OperatorOptimizeInput, OperatorOptimizeWorkflow
from kernelgen.workflows.optimization import operations
from kernelgen.workflows.optimization.kernelgen import KernelGenWorkflow, KernelGenOutput
from kernelgen.tests.test_kernelgen_uploaded_snapshot import snapshot


def test_paths_are_frozen_without_executing_or_validating_seed(tmp_path):
    seed=tmp_path/'seed.py';seed.write_text('not even valid Python')
    reference=tmp_path/'reference.py';reference.write_text('read-only source')
    inp=OperatorOptimizeInput(operator='add',catalog_path=tmp_path,optimization={
        'definition_name':'add','mode':'kernelgen','seed_code_path':seed,'reference_code_path':reference})
    workflow=OperatorOptimizeWorkflow(cwd=tmp_path/'run')
    prepared=workflow.prepare_code_inputs(inp)
    assert prepared.code_inputs_sha256['seed_code_path']==hashlib.sha256(seed.read_bytes()).hexdigest()
    assert prepared.optimization.n_parallel==1
    seed.write_text('changed seed')
    with pytest.raises(ValueError,match='digest'):
        workflow.prepare_code_inputs(prepared)


@pytest.mark.parametrize('outcome',['success','failure','exception','cancel'])
def test_kernelgen_consumes_bundle_and_weighted_lease(tmp_path,monkeypatch,outcome):
    monkeypatch.setenv('KERNELGEN_CLI_HOME',str(tmp_path/'state'))
    s=snapshot();path=tmp_path/'snapshot.json';atomic_write_json(path,s.model_dump(mode='json'))
    monkeypatch.setattr(operations,'receipt_data',lambda *_:{'snapshot':str(path)})
    seed=tmp_path/'seed.py';seed.write_text('initial unverified candidate')
    ref=tmp_path/'reference.py';ref.write_text('design evidence')
    inp=OperatorOptimizeInput(operator='add',catalog_path=tmp_path,optimization={
        'definition_name':'add','mode':'kernelgen','seed_code_path':seed,'reference_code_path':ref,
        'eval_server_url':'http://kgs:8000','target_hardware':'PPU-ZW810E','n_parallel':2,'n_epoch':2,
        'max_round':2,'warmup_ms':0,'benchmark_ms':13,'eval_timeout_seconds':91})
    root=tmp_path/'run';root.mkdir()
    inp=OperatorOptimizeWorkflow(cwd=root).prepare_code_inputs(inp)
    context=SimpleNamespace(control=WorkspaceRunControl(root))
    pool=WorkerLeasePool(inp.optimization.eval_server_url)
    set_max_workers(inp.optimization.eval_server_url,2)
    code='def run(x): return x'
    def run(workflow,args):
        assert pool.snapshot()['used_workers']==2
        assert args['evaluation_snapshot']==s.model_dump(mode='json')
        assert args['initial_seed_code']==seed.read_text()
        assert args['initial_seed_is_validated_baseline'] is False
        assert args['knowledge_run_id'] == f"run-{hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]}"
        assert args['reference_code_source']==ref.read_text()
        assert args['n_parallel']==args['n_epoch']==args['max_round']==2
        assert args['warmup_ms']==0 and args['benchmark_ms']==13 and args['eval_timeout_seconds']==91
        if outcome=='exception':raise RuntimeError('test failure')
        if outcome=='cancel':
            context.control.request_cancel('test cancel');workflow._run_control.checkpoint('SAFE_POINT')
        if outcome=='success':
            winner=workflow._cwd/'1R/agent0';winner.mkdir(parents=True)
            (winner/'.best_kernel.py').write_text(code)
            for file in ['.ledger.json','optimize_definition_output.json','.kernelgen/final-verification.json']:
                atomic_write_json(winner/file,{})
        return KernelGenOutput(definition_name='add',status='PASSED' if outcome=='success' else 'FAILED',best_code=code,best_geo_mean=1.2)
    monkeypatch.setattr(KernelGenWorkflow,'run',run)
    from kernelgen.workflows.optimization.kernelgen import epoch
    monkeypatch.setattr(epoch,'confirmed_workspace_best',lambda _:('PASSED',{'code':code,'geo_mean':1.2}))
    if outcome in {'exception','cancel'}:
        with pytest.raises(RuntimeError if outcome=='exception' else RunCancelled):
            operations.optimize(inp,root,lambda _:None,context)
    else:
        result=operations.optimize(inp,root,lambda _:None,context)
        assert result.state==('SUCCEEDED' if outcome=='success' else 'FAILED')
        if outcome=='success':
            assert Path(result.output['candidate']).is_file()
            assert any(p.endswith('/.ledger.json') for p in result.output['artifacts'])
    assert pool.snapshot()['used_workers']==0


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        OperatorOptimizeInput(operator='add',catalog_path=tmp_path,optimization={'definition_name':'add','mode':'unknown'})


def test_kernelgen_does_not_silently_ignore_dps_override(tmp_path):
    with pytest.raises(ValueError, match='destination_passing_style'):
        OperatorOptimizeInput(operator='add', catalog_path=tmp_path, optimization={
            'definition_name': 'add', 'mode': 'kernelgen', 'destination_passing_style': True})


@pytest.mark.parametrize('knowledge', [False, True])
def test_simple_opt_receives_unvalidated_seed_and_reference(tmp_path, monkeypatch, knowledge):
    from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationWorkflow
    from kernelgen.workflows.optimization import knowledge as knowledge_module

    monkeypatch.setenv('KERNELGEN_CLI_HOME', str(tmp_path/'state'))
    path = tmp_path/'snapshot.json'
    atomic_write_json(path, snapshot().model_dump(mode='json'))
    monkeypatch.setattr(operations, 'receipt_data', lambda *_: {'snapshot': str(path)})
    seed = tmp_path/'seed.py'
    seed.write_text('unvalidated seed')
    reference = tmp_path/'reference.py'
    reference.write_text('reference only')
    root = tmp_path/'run'
    root.mkdir()
    inp = OperatorOptimizeInput(operator='add', catalog_path=tmp_path, optimization={
        'definition_name': 'add', 'mode': 'simple_opt', 'seed_code_path': seed, 'reference_code_path': reference,
        'knowledge_catalog_path': tmp_path if knowledge else None})
    inp = OperatorOptimizeWorkflow(cwd=root).prepare_code_inputs(inp)
    pool = WorkerLeasePool(inp.optimization.eval_server_url)
    visited = []
    materialized = []
    monkeypatch.setattr(knowledge_module, 'materialize_knowledge', lambda *a, **kw: materialized.append(kw))

    def run(workflow, args):
        assert pool.snapshot()['used_workers'] == 1
        assert args['seed_code'] == seed.read_text()
        assert args['seed_is_validated_baseline'] is False
        assert args['reference_code_source'] == reference.read_text()
        assert args['evaluation_snapshot']['bundle_id'] == snapshot().bundle_id
        assert bool(args.get('knowledge_enabled')) == knowledge
        visited.append(workflow)
        raise RuntimeError('stop before model')

    monkeypatch.setattr(SingleCoderOptimizationWorkflow, 'run', run)
    with pytest.raises(RuntimeError, match='stop before model'):
        operations.optimize(inp, root, lambda _: None, SimpleNamespace(control=WorkspaceRunControl(root)))
    assert len(visited) == 1
    assert pool.snapshot()['used_workers'] == 0
    if knowledge:
        assert materialized == [{
            'run_id': f"run-{hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]}",
            'benchmark_id': f"{snapshot().catalog_name}-{snapshot().catalog_api_version}",
        }]
    else:
        assert materialized == []
