import json

import pytest
from kernelgen.tests.extraction_review_fixtures import stub_extraction_review

from kernelgen.agents.extractor.flaggems import pr_source
from kernelgen.agents.extractor.flaggems.source_profile import PROFILES, checkout_profile
from kernelgen.agents.extractor.flaggems.source_inventory import collect_source_inventory
from kernelgen.agents.extractor.flaggems.v62_agent import (
    FlagGemsV62ExtractorAgent, FlagGemsV62ExtractorInput, persist_flaggems_v62_extraction,
)
from kernelgen.tests.test_flaggems_v62_direct_agent import _repo, _case_report, _proposal, _Runtime


@pytest.mark.parametrize('profile', PROFILES)
def test_pr_resolution_preserves_repository(profile, monkeypatch):
    calls = []
    def api(path):
        calls.append(path)
        if '/compare/' in path:
            return {'merge_base_commit': {'sha': 'c' * 40}}
        return {'number': 731, 'base': {'sha': 'a' * 40, 'repo': {'full_name': profile.repository}},
                'head': {'sha': 'b' * 40}}
    monkeypatch.setattr(pr_source, '_github', api)
    url = f'https://github.com/{profile.repository}/pull/731'
    source = pr_source.resolve_pull_request(url)
    assert source.url == url and source.remote == profile.remote
    assert all(path.startswith(f'repos/{profile.repository}/') for path in calls)
    assert pr_source.PullRequestSource.model_validate_json(source.model_dump_json()) == source


def test_frozen_same_number_different_repository_rejected(tmp_path):
    source = pr_source.PullRequestSource(number=731, base_sha='a' * 40, head_sha='b' * 40, merge_base_sha='c' * 40)
    (tmp_path / 'source.json').write_text(source.model_dump_json())
    with pytest.raises(ValueError, match='different PR'):
        pr_source.prepare_pull_request('https://github.com/flagos-ai/FlagGems-vllm/pull/731', tmp_path)


def test_ambiguous_layout_rejected(tmp_path):
    for profile in PROFILES:
        profile.package_root(tmp_path).mkdir(parents=True)
    with pytest.raises(ValueError, match='exactly one'):
        checkout_profile(tmp_path)


@pytest.mark.parametrize('profile', PROFILES)
def test_shared_agent_and_persistence_use_selected_package(profile, tmp_path):
    repo = _repo(tmp_path)
    if profile.package != 'flag_gems':
        old = repo / 'src/flag_gems'
        old.rename(profile.package_root(repo))
        for path in repo.rglob('*.py'):
            path.write_text(path.read_text().replace('flag_gems', profile.package))
    op = 'adaptive_max_pool3d_backward'
    inventory = collect_source_inventory(str(repo), op)
    assert inventory.implemented
    assert inventory.implementation_file == profile.package_root(repo) / 'ops' / f'{op}.py'
    cases = tmp_path / 'cases.json'
    _case_report(cases)
    inp = FlagGemsV62ExtractorInput(operator=op, flaggems_repo=str(repo), case_list_path=str(cases), timing_reference='flaggems')
    agent = FlagGemsV62ExtractorAgent()
    prompt = agent.preprocess(inp, _Runtime(''))
    assert f'{profile.package}.{op}' in prompt
    proposal = json.loads(_proposal())
    proposal['oracle'] = proposal['oracle'].replace('def run(', 'def correctness_run(')
    proposal['oracle'] += f'\nimport {profile.package}\ndef timing_run(grad_output, self, indices):\n    return {profile.package}.{op}(grad_output, self, indices)\n'
    output = agent.postprocess(json.dumps(proposal), _Runtime(''))
    result = persist_flaggems_v62_extraction(inp, output, tmp_path / 'catalog')
    assert result.catalog_root == tmp_path / 'catalog'
    wrong = next(p.package for p in PROFILES if p != profile)
    proposal['oracle'] = proposal['oracle'].replace(profile.package, wrong)
    with pytest.raises(ValueError, match='cannot import|exact'):
        agent.postprocess(json.dumps(proposal), _Runtime(''))


@pytest.mark.parametrize('url', [
    'https://github.com/flagos-ai/FlagGems-vllm-evil/pull/731',
    'https://github.com/other/FlagGems-vllm/pull/731',
    'https://github.com/flagos-ai/FlagGems-vllm/pull/731?token=secret',
])
def test_repository_allowlist_not_arbitrary_url(url):
    with pytest.raises(ValueError, match='expected|unsupported'):
        pr_source.pull_request_number(url)


@pytest.mark.parametrize('profile', PROFILES)
def test_pr_workflow_catalog_preserves_binding_for_optimizer(profile, tmp_path, monkeypatch):
    from kernelgen.tests.test_flaggems_pr_extract import git
    from kernelgen.workflows.catalog_extract import CatalogExtractWorkflow
    from kernelgen.workflows.optimization import OperatorOptimizeInput, OperatorOptimizeWorkflow
    from kernelgen.workflows.optimization.artifacts import snapshot_catalog, catalog_identity

    repo = _repo(tmp_path)
    if profile.package != 'flag_gems':
        (repo / 'src/flag_gems').rename(profile.package_root(repo))
        for path in repo.rglob('*.py'):
            path.write_text(path.read_text().replace('flag_gems', profile.package))
    git(repo, 'init', '--quiet')
    git(repo, 'config', 'user.name', 'Fixture')
    git(repo, 'config', 'user.email', 'fixture@example.invalid')
    git(repo, 'add', '--', 'src', 'tests', 'benchmark')
    git(repo, 'commit', '--quiet', '-m', 'source fixture')
    revision = git(repo, 'rev-parse', 'HEAD')
    source = pr_source.PullRequestSource(repository=profile.repository, number=731,
                                         base_sha=revision, head_sha=revision, merge_base_sha=revision)
    monkeypatch.setattr(pr_source, 'resolve_pull_request', lambda _: source)
    monkeypatch.setattr(pr_source.PullRequestSource, 'remote', property(lambda _: str(repo)))
    cases = tmp_path / 'cases.json'
    _case_report(cases)
    operator = 'adaptive_max_pool3d_backward'
    proposal = json.loads(_proposal())
    proposal['oracle'] = proposal['oracle'].replace('def run(', 'def correctness_run(')
    proposal['oracle'] += f'\nimport {profile.package}\ndef timing_run(grad_output, self, indices):\n    return {profile.package}.{operator}(grad_output, self, indices)\n'
    # Actual preprocess/postprocess and persistence; only the model reply is a fixture.
    from kernelgen.framework.runtime.base import FakeRuntime
    runtime = FakeRuntime([json.dumps(proposal)])
    root = tmp_path / 'extracted'
    output = CatalogExtractWorkflow(cwd=root, runtime_factory=lambda _: runtime).run({
        'operator': operator, 'pr_url': source.url, 'case_list_path': str(cases)})
    manifest = json.loads((output.catalog_path / 'manifest.json').read_text())
    assert manifest['framework_revision'] == revision
    assert manifest['framework'] == profile.framework
    if profile.framework != 'flaggems':
        # KGS does not yet accept this native framework binding. Never erase or
        # relabel it merely to make Catalog optimization accept the artifact.
        with pytest.raises(ValueError, match='native catalog framework must be flaggems'):
            catalog_identity(output.catalog_path, operator)
        return
    before = catalog_identity(output.catalog_path, operator)
    optimizer = OperatorOptimizeWorkflow(cwd=tmp_path / 'optimized')
    inp = optimizer.prepare_input(OperatorOptimizeInput.model_validate({
        'operator': operator, 'catalog_path': output.catalog_path,
        'optimization': {'definition_name': operator}}))
    workspace = optimizer.root / 'stages/prepare_catalog/attempts/01'
    workspace.mkdir(parents=True)
    prepared = snapshot_catalog(inp, workspace)
    optimizer.validate_artifacts(prepared)
    assert catalog_identity(workspace / 'catalog', operator) == before
    assert catalog_identity(output.catalog_path, operator) == before
    with pytest.raises(ValueError, match='already exist'):
        CatalogExtractWorkflow(cwd=root, runtime_factory=lambda _: runtime).run({
            'operator': operator, 'pr_url': source.url, 'case_list_path': str(cases)})
