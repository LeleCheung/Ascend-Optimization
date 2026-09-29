import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelgen.agents.artifact_reviewer import ArtifactReviewerAgent, ReviewOutput
from kernelgen.framework.run_control import RunCancelled, WorkspaceRunControl
from kernelgen.workflows import catalog_extract as module
from kernelgen.workflows.catalog_extract_review import CatalogReviewRequired, reusable_review, review_link_path, publish_review_link
from kernelgen.workflows.catalog_extract_review import CatalogExtractionBlocked


def blocker(source):
    return dict(kind='protocol', source_requirement='Preserve source runtime precision branch',
                evidence_paths=[str(source)], missing_contract='No target policy input',
                resolution='Provide a target-owned precision policy')


@pytest.mark.parametrize('at_review', [False, True])
def test_blocker_stops_first_attempt_and_preserves_evidence(fixture, monkeypatch, at_review):
    root, source, calls, inp = fixture
    if at_review:
        monkeypatch.setattr(ArtifactReviewerAgent, 'run', lambda _, data, runtime:
            reviewer(data).model_copy(update={'blocker': module.FlagGemsV62ExtractorOutput(
                blocker=blocker(source)).blocker}))
    else:
        monkeypatch.setattr(module.FlagGemsV62ExtractorAgent, 'run', lambda *args:
            module.FlagGemsV62ExtractorOutput(blocker=blocker(source)))
    with pytest.raises(CatalogExtractionBlocked) as exc:
        module.CatalogExtractWorkflow(cwd=root, runtime_factory=lambda _: object()).run(inp)
    report = json.loads(exc.value.report_path.read_text())
    assert report['blocker']['evidence_paths'] == [str(source)]
    assert not (root / 'catalog').exists()
    assert not (root / 'attempts/02').exists()
    assert WorkspaceRunControl(root).progress().stage == 'BLOCKED'
    assert json.loads((root / 'review-loop.json').read_text())['state'] == 'BLOCKED'
    if at_review:
        assert report['accepted'] is False
        assert calls == ['extract']


def test_extractor_blocker_must_cite_bound_source(fixture, monkeypatch):
    root, source, calls, inp = fixture
    monkeypatch.setattr(module.FlagGemsV62ExtractorAgent, 'run', lambda *args:
        module.FlagGemsV62ExtractorOutput(blocker=blocker(source.with_name('invented.py'))))
    with pytest.raises(ValueError, match='supplied source evidence'):
        module.CatalogExtractWorkflow(cwd=root, runtime_factory=lambda _: object()).run(inp)
    assert WorkspaceRunControl(root).progress().stage == 'FAILED'
    assert not (root / 'catalog').exists()


def test_blocker_without_full_review_evidence_does_not_early_stop(fixture, monkeypatch):
    root, source, calls, inp = fixture
    monkeypatch.setattr(ArtifactReviewerAgent, 'run', lambda _, data, runtime:
        reviewer(data, missing=True).model_copy(update={'blocker':
            module.FlagGemsV62ExtractorOutput(blocker=blocker(source)).blocker}))
    with pytest.raises(CatalogReviewRequired):
        module.CatalogExtractWorkflow(cwd=root, runtime_factory=lambda _: object()).run(
            {**inp, 'max_review_rounds': 2})
    assert len(calls) == 2


def test_blocked_output_cannot_mix_success_assets_or_publish(tmp_path):
    proposal = dict(blocker=blocker(tmp_path / 'source.py'))
    with pytest.raises(ValueError, match='not both'):
        module.FlagGemsV62ExtractorOutput(**proposal, oracle='def run(x): return x')
    with pytest.raises(ValueError, match='requires oracle'):
        module.FlagGemsV62ExtractorOutput()
    output = module.FlagGemsV62ExtractorOutput(**proposal)
    with pytest.raises(ValueError, match='cannot be published'):
        module.persist_flaggems_v62_extraction(None, output, tmp_path / 'catalog')
    assert not (tmp_path / 'catalog').exists()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source=tmp_path/'source.py';source.write_text('def run(x): return x\n')
    cases=tmp_path/'cases.json';cases.write_text('{}')
    root=tmp_path/'run'
    monkeypatch.setattr(module,'collect_source_inventory',lambda *_:SimpleNamespace(
        implementation_file=source,test_files=[],benchmark_files=[]))
    monkeypatch.setattr(module,'build_flaggems_v62_accuracy_coverage',lambda *_:{'covered':True})
    proposal=module.FlagGemsV62ExtractorOutput(oracle='def run(x): return x\n',
        correctness_workloads=[{'name':'c','inputs':{'x':{'type':'scalar','value':1}}}],
        timing_workloads=[{'name':'t','inputs':{'x':{'type':'scalar','value':1}}}])
    calls=[]
    def run(agent, inp, runtime):
        calls.append('extract');return proposal
    def revise(agent, feedback, runtime):
        calls.append(feedback)
        assert 'requested_change' in feedback
        return proposal.model_copy(update={'oracle':'def run(x):\n    return x\n'})
    monkeypatch.setattr(module.FlagGemsV62ExtractorAgent,'run',run)
    monkeypatch.setattr(module.FlagGemsV62ExtractorAgent,'continue_session',revise)
    def persist(inp, output, catalog):
        op=catalog/'ops/add';op.mkdir(parents=True)
        (catalog/'manifest.json').write_text(json.dumps({'api_version':'v6.2','evaluator':'native','layout':'per-operator'}))
        (op/'definition.json').write_text(json.dumps({'api_version':'v6.2','name':'add','parameters':[{'name':'x','required':True}],'outputs':['out']}))
        (op/'oracle.py').write_text("REFERENCE_DEVICE = 'target'\n"+output.oracle)
        for kind in ['correctness','timing']:
            (op/(kind+'.jsonl')).write_text(json.dumps({'name':kind,'inputs':{'x':{'type':'scalar','value':1}}})+'\n')
        return SimpleNamespace(operator_root=op)
    monkeypatch.setattr(module,'persist_flaggems_v62_extraction',persist)
    return root,source,calls,dict(operator='add',flaggems_repo=str(tmp_path),case_list_path=str(cases))


def reviewer(inp, *, block=False, missing=False, category="conversion"):
    return ReviewOutput(summary='source fidelity review',reviewed_files=inp['evidence_paths'][1:] if missing else inp['evidence_paths'],
        findings=[dict(priority='P0' if block else 'P1',category=category,evidence='oracle.py:1',reason='fixture',requested_change='repair oracle')])


@pytest.mark.parametrize('category', ['source_quality', 'target_capability'])
def test_source_and_target_findings_do_not_retry_and_can_be_reused(fixture, monkeypatch, category):
    root, source, calls, inp = fixture
    monkeypatch.setattr(ArtifactReviewerAgent, 'run',
        lambda _, data, runtime: reviewer(data, block=True, category=category))
    result = module.CatalogExtractWorkflow(cwd=root, runtime_factory=lambda _: object()).run(inp)
    assert calls == ['extract']
    report = reusable_review(result.catalog_path, 'add')
    assert report['accepted'] is True
    assert report['findings'][0]['priority'] == 'P0'
    assert report['findings'][0]['category'] == category


@pytest.mark.parametrize('category', ['source_quality', 'target_capability'])
def test_advisory_category_does_not_excuse_unread_evidence(fixture, monkeypatch, category):
    root, source, calls, inp = fixture
    monkeypatch.setattr(ArtifactReviewerAgent, 'run',
        lambda _, data, runtime: reviewer(data, missing=True, category=category))
    with pytest.raises(CatalogReviewRequired):
        module.CatalogExtractWorkflow(cwd=root, runtime_factory=lambda _: object()).run(
            {**inp, 'max_review_rounds': 1})
    assert not (root/'catalog').exists()


@pytest.mark.parametrize('category,expected', [
    ('conversion', 'WAITING'), ('source_quality', 'SUCCEEDED'), ('target_capability', 'SUCCEEDED'),
])
def test_unreviewed_downstream_catalog_uses_same_policy(tmp_path, monkeypatch, category, expected):
    from kernelgen.workflows.optimization.operations import _review
    context = SimpleNamespace(operator='add', workspace=tmp_path)
    monkeypatch.setattr(ArtifactReviewerAgent, 'run',
        lambda _, data, runtime: reviewer(data, block=True, category=category))
    result = _review(lambda _: object(), context, 'catalog', 'digest', [tmp_path/'source.py'])
    assert result.state == expected


def test_unclassified_findings_are_not_silently_advisory():
    report = ReviewOutput(summary='Missing backward assertion', reviewed_files=['source.py'],
        findings=[dict(priority='P0', evidence='source.py:1 vs oracle.py:2',
                       reason='source assertion omitted', requested_change='preserve assertion')])
    assert report.blocks_catalog()


@pytest.mark.parametrize('package', ['flag_gems', 'flaggems_vllm'])
@pytest.mark.parametrize('changed_helper', [1, 3])
def test_review_binds_source_helpers_and_invalidates_changed_defaults(fixture, monkeypatch, package, changed_helper):
    root, source, calls, inp = fixture
    helpers = []
    for relative in ('tests/accuracy_utils.py', 'tests/conftest.py', 'benchmark/base.py',
                     f'src/{package}/testing/__init__.py'):
        path = Path(inp['flaggems_repo']) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('TO_CPU = False\n')
        helpers.append(path)
    def review(agent, data, runtime):
        assert {str(p.resolve()) for p in helpers} <= set(data['evidence_paths'])
        return reviewer(data)
    monkeypatch.setattr(ArtifactReviewerAgent, 'run', review)
    result = module.CatalogExtractWorkflow(cwd=root, runtime_factory=lambda _: object()).run(inp)
    assert reusable_review(result.catalog_path, 'add') is not None
    helpers[changed_helper].write_text('changed helper contract\n')
    assert reusable_review(result.catalog_path, 'add') is None


def test_repair_and_re_review_publishes_only_accepted_revision(fixture,monkeypatch):
    root,source,calls,inp=fixture
    reviews=[]
    def review(agent, data, runtime):
        reviews.append(data['subject_sha256']);return reviewer(data,block=len(reviews)==1)
    monkeypatch.setattr(ArtifactReviewerAgent,'run',review)
    result=module.CatalogExtractWorkflow(cwd=root,runtime_factory=lambda _:object()).run(inp)
    assert len(calls)==len(reviews)==2 and reviews[0]!=reviews[1]
    assert 'repair oracle' in calls[1]
    assert result.review_path==root/'attempts/02/review/review.json'
    assert (root/'catalog/ops/add/oracle.py').read_text()==(root/'attempts/02/catalog/ops/add/oracle.py').read_text()
    assert (root/'attempts/01/review/review.json').is_file()
    assert WorkspaceRunControl(root).progress().state.value=='SUCCEEDED'
    assert reusable_review(result.catalog_path, 'add')['accepted'] is True
    assert review_link_path(result.catalog_path).is_file()


@pytest.mark.parametrize('missing',[False,True])
def test_budget_exhaustion_never_publishes_catalog(fixture,monkeypatch,missing):
    root,source,calls,inp=fixture
    monkeypatch.setattr(ArtifactReviewerAgent,'run',lambda _,data,runtime:reviewer(data,block=not missing,missing=missing))
    with pytest.raises(CatalogReviewRequired,match='after 2 rounds'):
        module.CatalogExtractWorkflow(cwd=root,runtime_factory=lambda _:object()).run({**inp,'max_review_rounds':2})
    assert len(calls)==2 and not (root/'catalog').exists()
    assert WorkspaceRunControl(root).progress().stage=='WAITING_REVIEW'


def test_review_cannot_modify_evidence(fixture,monkeypatch):
    root,source,calls,inp=fixture
    def review(agent,data,runtime):
        source.write_text('changed');return reviewer(data)
    monkeypatch.setattr(ArtifactReviewerAgent,'run',review)
    with pytest.raises(ValueError,match='changed'):
        module.CatalogExtractWorkflow(cwd=root,runtime_factory=lambda _:object()).run(inp)
    assert not (root/'catalog').exists()


@pytest.mark.parametrize('change', ['source', 'catalog', 'missing_source', 'report', 'malformed_link', 'missing_link', 'operator'])
def test_changed_review_or_evidence_is_not_reused(fixture, monkeypatch, change):
    root, source, calls, inp = fixture
    monkeypatch.setattr(ArtifactReviewerAgent, 'run', lambda _, data, runtime: reviewer(data))
    result = module.CatalogExtractWorkflow(cwd=root, runtime_factory=lambda _: object()).run(inp)
    assert reusable_review(result.catalog_path, 'add') is not None
    if change == 'source':
        source.write_text('changed source')
    elif change == 'catalog':
        with (result.catalog_path/'ops/add/oracle.py').open('a') as f:
            f.write('# changed\n')
    elif change == 'missing_source':
        source.unlink()
    elif change == 'report':
        result.review_path.write_text('{}')
    elif change == 'malformed_link':
        review_link_path(result.catalog_path).write_text('[]')
    elif change == 'missing_link':
        review_link_path(result.catalog_path).unlink()
    assert reusable_review(result.catalog_path, 'other' if change == 'operator' else 'add') is None


@pytest.mark.parametrize('stale', [False, True])
@pytest.mark.parametrize('target_state', ['SUCCEEDED', 'FAILED'])
def test_downstream_reviews_even_with_matching_extraction_review(fixture, monkeypatch, tmp_path, stale, target_state):
    from kernelgen.workflows.optimization import operations
    from kernelgen.workflows.optimization.artifacts import snapshot_catalog, catalog_identity, workflow_result
    root, source, calls, inp = fixture
    monkeypatch.setattr(ArtifactReviewerAgent, 'run', lambda _, data, runtime: reviewer(data))
    result = module.CatalogExtractWorkflow(cwd=root, runtime_factory=lambda _: object()).run(inp)
    snapshot_dir = tmp_path/'prepared'
    snapshot_dir.mkdir()
    prepared = snapshot_catalog(SimpleNamespace(catalog_path=result.catalog_path, operator='add',
        catalog_sha256=catalog_identity(result.catalog_path, 'add')), snapshot_dir).output
    prepared['snapshot'] = str(tmp_path/'evaluation-contract.json')
    (tmp_path/'evaluation-contract.json').write_text('{}')
    prepared['test_evidence'] = str(tmp_path/'test-sources.json')
    (tmp_path/'test-sources.json').write_text(json.dumps({
        'files': [{'path': 'test_add.py', 'content': '# original source\n'}]}))
    if stale:
        source.write_text('changed source')
    semantic_calls, target_calls = [], []
    def review(*args):
        semantic_calls.append(True)
        return workflow_result({})
    def target(inp, context, prepared, files):
        target_calls.append(list(files))
        return workflow_result({}, files, state=target_state)
    monkeypatch.setattr(operations, '_review', review)
    monkeypatch.setattr(operations, 'validate_target', target)
    context = SimpleNamespace(operator='add', workspace=tmp_path/'downstream', outputs={'prepare_catalog': prepared})
    context.workspace.mkdir()
    output = operations.review_tests(SimpleNamespace(skip_review=False), None, None, context)
    assert len(semantic_calls) == 1
    assert len(target_calls) == 1
    assert output.state == target_state
    assert output.output['test_review'] == 'ACCEPTED'
    assert not (context.workspace/'reused-review.json').exists()


@pytest.mark.parametrize('invalid', ['rejected', 'p0', 'unread_evidence'])
def test_digest_match_alone_does_not_authorize_review_reuse(fixture, monkeypatch, invalid):
    root, source, calls, inp = fixture
    monkeypatch.setattr(ArtifactReviewerAgent, 'run', lambda _, data, runtime: reviewer(data))
    result = module.CatalogExtractWorkflow(cwd=root, runtime_factory=lambda _: object()).run(inp)
    report = json.loads(result.review_path.read_text())
    if invalid == 'rejected':
        report['accepted'] = False
    elif invalid == 'p0':
        report['findings'][0]['priority'] = 'P0'
    else:
        report['reviewed_files'] = report['reviewed_files'][1:]
    result.review_path.write_text(json.dumps(report))
    publish_review_link(result.catalog_path, 'add', result.review_path)
    assert reusable_review(result.catalog_path, 'add') is None


@pytest.mark.parametrize('blocked', [False, True])
def test_cancel_after_complete_review_stops_before_repair(fixture,monkeypatch,blocked):
    root,source,calls,inp=fixture
    def review(agent,data,runtime):
        WorkspaceRunControl(root).request_cancel('test safe point')
        report = reviewer(data,block=True)
        if blocked:
            report.blocker = module.FlagGemsV62ExtractorOutput(blocker=blocker(source)).blocker
        return report
    monkeypatch.setattr(ArtifactReviewerAgent,'run',review)
    with pytest.raises(RunCancelled):
        module.CatalogExtractWorkflow(cwd=root,runtime_factory=lambda _:object()).run(inp)
    assert len(calls)==1 and not (root/'catalog').exists()
