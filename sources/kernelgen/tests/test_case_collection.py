import json
from pathlib import Path
import sys

import pytest
from kernelgen.tests.extraction_review_fixtures import stub_extraction_review

from kernelgen.agents.extractor.flaggems.case_collection import prepare_case_list
from kernelgen.framework.runtime.base import FakeRuntime
from kernelgen.tests.test_flaggems_v62_direct_agent import _repo, _case_report
from kernelgen.agents.extractor.flaggems import collection_executor, collection_worker
from kernelgen.agents.extractor.flaggems.collection_config import set_extraction_setting


@pytest.fixture(autouse=True)
def collection_transport(tmp_path, monkeypatch):
    # Exercise the actual standalone worker and stdio protocol, but not SSH/Docker.
    monkeypatch.setenv('KERNELGEN_CLI_HOME', str(tmp_path / 'state'))
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    for key, value in {'host': 'fixture-extract', 'container': 'fixture', 'python': sys.executable}.items():
        set_extraction_setting(key, value)
    monkeypatch.setattr(collection_executor, 'ssh_command', lambda _: [sys.executable, '-u', collection_worker.__file__])


@pytest.fixture
def source(tmp_path):
    root = _repo(tmp_path)
    report = tmp_path / 'fixture.json'
    _case_report(report)
    # This fixture genuinely runs in the subprocess; the model only supplies the adapter.
    (root / 'benchmark/collect_backend.py').write_text(
        'import os\ndef collect():\n'
        '    assert "ANTHROPIC_AUTH_TOKEN" not in os.environ\n'
        f'    return {json.loads(report.read_text())!r}\n')
    return root


ADAPTER = '''import importlib.util
from pathlib import Path
def collect_cases(source_root, operator):
    path = Path(source_root) / "benchmark/collect_backend.py"
    spec = importlib.util.spec_from_file_location("original_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect()
'''


def factory(*sources):
    runtime = FakeRuntime([json.dumps({'source': source}) for source in sources])
    return lambda _: runtime


def test_executes_original_builder_without_user_case_path(source, tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_AUTH_TOKEN', 'fixture-not-a-real-secret')
    path = prepare_case_list(source, 'adaptive_max_pool3d_backward', tmp_path / 'collection', factory(ADAPTER))
    assert json.loads(path.read_text())['benchmarks'][0]['cases']
    assert 'benchmark/collect_backend.py' in json.loads(path.with_name('executed-source.json').read_text())
    assert prepare_case_list(source, 'adaptive_max_pool3d_backward', tmp_path / 'collection', factory()) == path
    path.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        prepare_case_list(source, 'adaptive_max_pool3d_backward', tmp_path / 'collection', factory())


def test_execution_failure_repaired_and_evidence_retained(source, tmp_path):
    path = prepare_case_list(source, 'adaptive_max_pool3d_backward', tmp_path / 'collection', factory(
        'def collect_cases(source_root, operator):\n    raise RuntimeError("fixture error")\n', ADAPTER))
    assert path.parent.name == '02'
    assert 'fixture error' in (path.parent.parent / '01/pytest.log').read_text()


@pytest.mark.parametrize('source_code', [
    'def collect_cases(source_root, operator):\n    return {}\n',
    'import pytest\ndef collect_cases(source_root, operator):\n    pytest.skip("unsupported")\n',
    ADAPTER.replace('return module.collect()', 'report = module.collect()\n    report["benchmarks"][0]["cases"] = []\n    return report'),
    ADAPTER.replace('return module.collect()', 'report = module.collect()\n    report["benchmarks"][0]["cases"] *= 2\n    return report'),
    ADAPTER.replace('return module.collect()', 'report = module.collect()\n    report["benchmarks"][0]["phase"] = "correctness"\n    return report'),
])
def test_static_skipped_or_empty_collection_never_succeeds(source, tmp_path, source_code):
    with pytest.raises(RuntimeError, match='two attempts'):
        prepare_case_list(source, 'adaptive_max_pool3d_backward', tmp_path / 'collection', factory(source_code, source_code))
    assert not (tmp_path / 'collection/collection.json').exists()


def test_changed_source_not_reused(source, tmp_path):
    prepare_case_list(source, 'adaptive_max_pool3d_backward', tmp_path / 'collection', factory(ADAPTER))
    (source / 'benchmark/collect_backend.py').write_text('changed')
    with pytest.raises(ValueError, match='source/operator changed'):
        prepare_case_list(source, 'adaptive_max_pool3d_backward', tmp_path / 'collection', factory())


def test_native_source_change_invalidates_receipt(source, tmp_path):
    (source / 'kernel.cpp').write_text('original')
    prepare_case_list(source, 'adaptive_max_pool3d_backward', tmp_path / 'collection', factory(ADAPTER))
    (source / 'kernel.cpp').write_text('changed')
    with pytest.raises(ValueError, match='source/operator changed'):
        prepare_case_list(source, 'adaptive_max_pool3d_backward', tmp_path / 'collection', factory())


def test_workflow_collects_before_extraction(source, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from kernelgen.workflows import catalog_extract as workflow
    from kernelgen.agents.extractor.flaggems.v62_agent import FlagGemsV62ExtractorOutput
    seen = []
    def extract(agent, inp, runtime):
        assert Path(inp['case_list_path']).is_file()
        seen.append(inp)
        return FlagGemsV62ExtractorOutput(oracle='fixture', correctness_workloads=[{'name': 'c', 'inputs': {}}], timing_workloads=[{'name': 't', 'inputs': {}}])
    monkeypatch.setattr(workflow.FlagGemsV62ExtractorAgent, 'run', extract)
    monkeypatch.setattr(workflow, 'build_flaggems_v62_accuracy_coverage', lambda *_: {})
    monkeypatch.setattr(workflow, 'persist_flaggems_v62_extraction',
                        lambda inp, output, root: SimpleNamespace(operator_root=root / 'ops/adaptive_max_pool3d_backward'))
    result = workflow.CatalogExtractWorkflow(cwd=str(tmp_path / 'workflow'), runtime_factory=factory(ADAPTER)).run({
        'flaggems_repo': str(source), 'operator': 'adaptive_max_pool3d_backward'})
    assert result.case_list_path == seen[0]['case_list_path']
    from kernelgen.framework.run_control import WorkspaceRunControl
    nested = WorkspaceRunControl(tmp_path / 'workflow/case_collection/01/agent')
    assert nested.root_workspace == tmp_path / 'workflow'


def test_collector_timeout_keeps_log_without_success_receipt(source, tmp_path):
    from kernelgen.agents.extractor.flaggems.collection_config import load_extraction_config
    attempt = tmp_path / 'timeout'
    attempt.mkdir()
    (attempt / 'test_collect_cases.py').write_text('import time\ndef test_wait():\n    time.sleep(30)\n')
    (attempt / 'collector.py').write_text('')
    (attempt / 'request.json').write_text(json.dumps({'source_root': str(source)}))
    with pytest.raises(RuntimeError, match='timed out'):
        collection_executor.execute_pytest(attempt, 0.3, load_extraction_config())
    assert (attempt / 'pytest.log').exists()


def test_modified_source_during_collection_aborts_without_retry(source, tmp_path):
    code = ADAPTER.replace('return module.collect()', 'Path(source_root, "benchmark/collect_backend.py").write_text("changed")\n    return module.collect()')
    with pytest.raises(RuntimeError, match='source changed during collection'):
        prepare_case_list(source, 'adaptive_max_pool3d_backward', tmp_path / 'collection', factory(code))
    assert not (tmp_path / 'collection/collection.json').exists()
