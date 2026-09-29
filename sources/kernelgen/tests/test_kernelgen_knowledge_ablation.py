"""Host-only controls for distinguishing knowledge reuse from seed evolution."""
from pathlib import Path

import pytest

from kernelgen.framework.run_options import optimization_argv, optimization_parser, resolve_run_options
from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.framework.catalog_options import catalog_input
from kernelgen.knowledge.config import KnowledgeConfig
from kernelgen.workflows.optimization.kernelgen.contracts import KernelGenInput, EpochResult
from kernelgen.workflows.optimization.kernelgen import epoch, finalization, knowledge


def inp(**kwargs):
    return KernelGenInput(definition={'name':'attention'},target_hardware='Ascend910B',**kwargs)


def test_cli_default_and_explicit_ablation():
    assert resolve_run_options({'mode':'kernelgen'})['cross_epoch_knowledge'] is True
    assert resolve_run_options({'mode':'kernelgen','cross_epoch_knowledge':False})['cross_epoch_knowledge'] is False


@pytest.mark.parametrize('enabled',[False,True])
def test_directions_control_preserves_seed_and_shared_analysis(monkeypatch,enabled):
    monkeypatch.setattr(epoch,'build_coder_input',lambda _inp,analysis,**kw: {'analysis':analysis,**kw})
    analysis={'core_math':'attention'}
    direction={'direction':'pipeline'}
    result=epoch.build_epoch_inputs(inp(cross_epoch_knowledge=enabled, n_parallel=3),analysis,[direction],'measured seed',2)
    assert len(result)==3
    for item in result:
        assert item['seed_code']=='measured seed'
        assert item['analysis']['core_math']=='attention'
        assert ('assigned_direction' in item['analysis']) is enabled
    assert analysis=={'core_math':'attention'}


def test_disabled_legacy_kb_not_materialized(tmp_path):
    (tmp_path/'kb').mkdir()
    (tmp_path/'kb'/'sentinel.md').write_text('old knowledge')
    disabled=epoch.create_epoch_workspace(tmp_path,'2R',None,inherit_legacy_knowledge=False)
    path=Path(disabled.allocate('agent0'))
    assert not (path/'kb').exists()
    enabled=epoch.create_epoch_workspace(tmp_path,'3R',None)
    assert (Path(enabled.allocate('agent0'))/'kb'/'sentinel.md').read_text()=='old knowledge'


def test_disabled_rejects_writable_v1_in_execution_and_finalization(tmp_path):
    with pytest.raises(ValueError,match='requires read_only_v1 or no KnowledgeConfig'):
        knowledge.build_knowledge_bridge(config=KnowledgeConfig(catalog_root=tmp_path/'kb'),
            cwd=tmp_path,inp=inp(cross_epoch_knowledge=False),start_mode='fresh')


def test_readonly_common_kb_is_independent_of_epoch_reuse(tmp_path, monkeypatch):
    captured = {}
    def bridge(**kwargs):
        captured.update(kwargs)
        return 'readonly-bridge'
    monkeypatch.setattr(knowledge, 'KernelGenKnowledgeBridge', bridge)
    monkeypatch.setattr(knowledge, 'catalog_benchmark_id', lambda _: 'fixture-v6.2')
    config = KnowledgeConfig(catalog_root=tmp_path/'kb', mode='read_only_v1')
    assert knowledge.build_knowledge_bridge(config=config, cwd=tmp_path,
        inp=inp(cross_epoch_knowledge=False), start_mode='fresh') == 'readonly-bridge'
    assert captured['config'] is config
    assert captured['start_mode'] == 'fresh'


def test_cli_ablation_survives_launcher_serialization():
    options = resolve_run_options({'mode': 'kernelgen', 'cross_epoch_knowledge': False,
                                   'knowledge_mode': 'read_only_v1'})
    argv = optimization_argv(options)
    assert '--no-cross-epoch-knowledge' in argv
    parsed = optimization_parser('kernelgen').parse_args(argv)
    assert parsed.cross_epoch_knowledge is False
    assert parsed.knowledge_mode == 'read_only_v1'


def test_catalog_input_preserves_ablation_with_readonly_kb(tmp_path):
    options = resolve_run_options({'mode': 'kernelgen', 'cross_epoch_knowledge': False,
        'knowledge_mode': 'read_only_v1', 'knowledge_catalog_path': str(tmp_path)})
    optimization = catalog_input(options, 'attention').optimization
    assert optimization.cross_epoch_knowledge is False
    assert optimization.knowledge_config.mode == 'read_only_v1'


def test_legacy_setup_does_not_copy_kb_when_disabled(tmp_path, monkeypatch):
    from kernelgen.examples.kernel_gen import run_example
    source = tmp_path / 'source'
    (source / 'kb').mkdir(parents=True)
    (source / 'kb' / 'sentinel.md').write_text('prior knowledge')
    monkeypatch.setattr(run_example, 'copy_claude_directory', lambda *args: None)
    workspace = tmp_path / 'run'
    run_example.setup_workspace(workspace, source, cross_epoch_knowledge=False)
    assert not (workspace / 'kb').exists()


def test_fresh_epoch_has_no_external_seed(monkeypatch):
    monkeypatch.setattr(epoch, 'build_coder_input', lambda _inp, analysis, **kw: kw)
    values = epoch.build_epoch_inputs(inp(cross_epoch_knowledge=False, n_parallel=3), {}, [], '', 1)
    assert len(values) == 3
    assert all(not value['seed_code'] for value in values)


def test_disabled_finalization_keeps_completion_without_merging(tmp_path,monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError('legacy merge should not execute')
    monkeypatch.setattr(knowledge,'merge_epoch_kb',forbidden)
    workspace=epoch.create_epoch_workspace(tmp_path,'1R',None,inherit_legacy_knowledge=False)
    workspace.allocate('agent0')
    summary=finalization.finalize_epoch_outputs(cwd=tmp_path,
        run_control=WorkspaceRunControl(tmp_path),inp=inp(cross_epoch_knowledge=False,n_parallel=1),
        results=[(None,'agent0')],best_result=EpochResult('attention'),epoch_num=1,
        epoch_workspace=workspace,runtime_factory=lambda _: (_ for _ in ()).throw(AssertionError('unexpected model')),
        knowledge=None)
    assert summary.next_directions==[]
    assert (tmp_path/'1R/epoch-completion.json').is_file()
    assert (tmp_path/'1R/synthesis/synthesis.json').is_file()
