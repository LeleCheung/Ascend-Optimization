import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from kernelgen.tests.extraction_review_fixtures import stub_extraction_review

from kernelgen.agents.extractor.flaggems import pr_source
from kernelgen.agents.extractor.flaggems.source_profile import repository_profile
from kernelgen.agents.extractor.flaggems.pr_agent import FlagGemsPRExtractorAgent, FlagGemsPRExtractorInput, extraction_input
from kernelgen.agents.extractor.flaggems.v62_agent import FlagGemsV62ExtractorAgent, FlagGemsV62ExtractorInput

URL = "https://github.com/flagos-ai/FlagGems/pull/5623"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()


@pytest.fixture
def source_repo(tmp_path, monkeypatch, request):
    profile = repository_profile(getattr(request, "param", "flagos-ai/FlagGems"))
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "--quiet")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    profile.package_root(repo).mkdir(parents=True)
    (profile.package_root(repo) / "__init__.py").write_text("")
    (repo / "operator.py").write_text("before")
    git(repo, "add", "operator.py", "src")
    git(repo, "commit", "--quiet", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / "operator.py").write_text("after")
    git(repo, "commit", "--quiet", "-am", "head")
    head = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(pr_source, "REMOTE", str(repo))
    monkeypatch.setattr(pr_source.PullRequestSource, "remote", property(lambda self: str(repo)))
    source = pr_source.PullRequestSource(repository=profile.repository, number=5623, base_sha=base, head_sha=head, merge_base_sha=base)
    monkeypatch.setattr(pr_source, "resolve_pull_request", lambda _: source)
    return repo, source


@pytest.mark.parametrize("url", [
    "http://github.com/flagos-ai/FlagGems/pull/5623",
    "https://github.com.evil/flagos-ai/FlagGems/pull/5623",
    "https://user:password@github.com/flagos-ai/FlagGems/pull/5623",
    "https://github.com/flagos-ai/FlagGems/pull/5623?token=secret",
    "https://github.com/flagos-ai/FlagGems/pull/5623#issue",
    "https://github.com/flagos-ai/FlagGems/pull/0",
    "https://github.com/another/project/pull/5623",
    "file:///tmp/repo",
])
def test_rejects_unsupported_or_credential_urls(url):
    with pytest.raises(ValueError, match="expected|unsupported"):
        pr_source.pull_request_number(url)


def test_api_resolves_exact_pair_and_merge_base(monkeypatch):
    calls = []
    def api(path):
        calls.append(path)
        if "compare/" in path:
            return {"merge_base_commit": {"sha": "c" * 40}}
        return {"number": 5623, "base": {"sha": "a" * 40, "repo": {"full_name": "flagos-ai/FlagGems"}},
                "head": {"sha": "b" * 40}}
    monkeypatch.setattr(pr_source, "_github", api)
    source = pr_source.resolve_pull_request(URL)
    assert source.head_sha == "b" * 40 and source.merge_base_sha == "c" * 40
    assert calls[1].endswith("a" * 40 + "..." + "b" * 40)


def test_frozen_retry_does_not_contact_github_or_follow_pr(source_repo, tmp_path, monkeypatch):
    repo, expected = source_repo
    root = tmp_path / "source"
    assert pr_source.prepare_pull_request(URL, root) == expected
    assert pr_source.changed_paths(root, expected) == ["operator.py"]
    assert (root / "head/operator.py").read_text() == "after"
    assert (root / "base/operator.py").read_text() == "before"
    (repo / "operator.py").write_text("new head")
    git(repo, "commit", "--quiet", "-am", "advance")
    monkeypatch.setattr(pr_source, "resolve_pull_request", lambda *_: pytest.fail("must reuse frozen PR"))
    original = pr_source._git
    def no_fetch(root, *args, **kwargs):
        assert args[0] != "fetch"
        return original(root, *args, **kwargs)
    monkeypatch.setattr(pr_source, "_git", no_fetch)
    assert pr_source.prepare_pull_request(URL, root) == expected


@pytest.mark.parametrize("change", ["tracked", "untracked", "head", "origin", "different_pr"])
def test_source_reuse_rejects_changes(source_repo, tmp_path, change):
    _, expected = source_repo
    root = tmp_path / "source"
    pr_source.prepare_pull_request(URL, root)
    if change == "tracked":
        (root / "head/operator.py").write_text("tamper")
    elif change == "untracked":
        (root / "head/untracked.py").write_text("tamper")
    elif change == "head":
        git(root / "head", "checkout", "--detach", expected.base_sha)
    elif change == "origin":
        git(root / "head", "remote", "set-url", "origin", "https://invalid.example/repo")
    with pytest.raises(ValueError):
        pr_source.prepare_pull_request(URL.replace("5623", "5624") if change == "different_pr" else URL, root)


def test_failed_first_fetch_is_retryable_with_same_pins(source_repo, tmp_path, monkeypatch):
    _, expected = source_repo
    root = tmp_path / "source"
    original = pr_source._git
    def fail_fetch(root, *args, **kwargs):
        if args[0] == "fetch":
            raise RuntimeError("synthetic interruption")
        return original(root, *args, **kwargs)
    monkeypatch.setattr(pr_source, "_git", fail_fetch)
    with pytest.raises(RuntimeError):
        pr_source.prepare_pull_request(URL, root)
    assert json.loads((root / "source.json").read_text())["head_sha"] == expected.head_sha
    monkeypatch.setattr(pr_source, "_git", original)
    monkeypatch.setattr(pr_source, "resolve_pull_request", lambda *_: pytest.fail("must keep first resolution"))
    assert pr_source.prepare_pull_request(URL, root) == expected


def test_pr_agent_shares_validator_and_injects_fixed_context(source_repo, tmp_path, monkeypatch):
    _, source = source_repo
    root = tmp_path / "source"
    pr_source.prepare_pull_request(URL, root)
    seen = []
    def preprocess(agent, inp, runtime):
        seen.append(inp)
        return "common source contract"
    monkeypatch.setattr(FlagGemsV62ExtractorAgent, "preprocess", preprocess)
    inp = FlagGemsPRExtractorInput(operator="fused_experts_impl", source_workspace=root,
                                   case_list_path=tmp_path / "cases.json")
    prompt = FlagGemsPRExtractorAgent().preprocess(inp, SimpleNamespace())
    assert seen == [extraction_input(inp)]
    assert seen[0].timing_reference == "flaggems"
    assert source.head_sha in prompt and source.base_sha in prompt
    assert "operator.py" in prompt and "not vLLM" in prompt
    assert FlagGemsPRExtractorAgent.postprocess is FlagGemsV62ExtractorAgent.postprocess
    assert FlagGemsPRExtractorAgent.OutputModel is FlagGemsV62ExtractorAgent.OutputModel


def test_explicit_gems_baseline_uses_existing_oracle_gate(tmp_path):
    from kernelgen.tests.test_flaggems_v62_direct_agent import _repo, _case_report, _Runtime, _proposal

    repo = _repo(tmp_path)
    cases = tmp_path / "cases.json"
    _case_report(cases)
    inp = FlagGemsV62ExtractorInput(operator="adaptive_max_pool3d_backward", flaggems_repo=str(repo),
                                    case_list_path=str(cases), timing_reference="flaggems")
    agent = FlagGemsV62ExtractorAgent()
    prompt = agent.preprocess(inp, _Runtime(""))
    assert agent._allow_flaggems_timing_reference
    assert "flag_gems.adaptive_max_pool3d_backward" in prompt
    with pytest.raises(ValueError, match="self-baselined"):
        agent.postprocess(_proposal(), _Runtime(""))


@pytest.mark.parametrize("source_repo", ["flagos-ai/FlagGems", "flagos-ai/FlagGems-vllm"], indirect=True)
@pytest.mark.parametrize("supplied_cases", [True, False])
def test_extract_launcher_preserves_pins_and_marks_target_unverified(source_repo, tmp_path, monkeypatch, capsys, supplied_cases):
    import sys
    from kernelgen.examples.flaggems_pr_extract import run_example as example
    from kernelgen.agents.extractor.flaggems.v62_agent import FlagGemsV62ExtractorOutput

    _, source = source_repo
    root = tmp_path / "run"
    cases = tmp_path / "cases.json"
    cases.write_text("fixture")
    output = FlagGemsV62ExtractorOutput(oracle="def run(x): return x",
        correctness_workloads=[{"name": "accuracy", "inputs": {}}],
        timing_workloads=[{"name": "timing", "inputs": {}}])
    monkeypatch.setattr(example, "_runtime", lambda *_: object())
    from kernelgen.workflows import catalog_extract
    calls = []
    def run(agent, inp, runtime):
        calls.append("extract")
        assert inp["source_workspace"] == root / "source"
        return output
    monkeypatch.setattr(FlagGemsPRExtractorAgent, "run", run)
    def collect(*_):
        assert not supplied_cases
        return cases
    monkeypatch.setattr(catalog_extract, "prepare_case_list", collect)
    monkeypatch.setattr(catalog_extract, "build_flaggems_v62_accuracy_coverage", lambda *_: {"fixture": True})
    monkeypatch.setattr(catalog_extract, "collect_source_inventory", lambda *_: SimpleNamespace(
        implementation_file=root / "source/head/operator.py", test_files=[], benchmark_files=[]))
    def persist(inp, proposed, catalog):
        assert inp.timing_reference == "flaggems"
        assert inp.flaggems_repo == str(root / "source/head")
        assert proposed == output
        catalog.mkdir()
        (catalog / "manifest.json").write_text("{}")
        calls.append("persist")
        return SimpleNamespace(operator_root=catalog / "ops/fused_experts_impl")
    monkeypatch.setattr(catalog_extract, "persist_flaggems_v62_extraction", persist)
    monkeypatch.setattr(example, "prepare_pull_request", lambda *_: pytest.fail("extract launcher must not prepare source twice"))
    monkeypatch.setattr(sys, "argv", ["example", "extract", "--pr-url", source.url,
        "--workspace", str(root), "--operator", "fused_experts_impl"] + (["--case-list-path", str(cases)] if supplied_cases else []))
    assert example.main() == 0
    assert json.loads(capsys.readouterr().out)["target_validation"] == "NOT_RUN"
    manifest = json.loads((root / "catalog/manifest.json").read_text())
    assert manifest["framework_revision"] == source.head_sha
    assert manifest["framework"] == repository_profile(source.repository).framework
    assert manifest["framework_repository"] == source.remote
    with pytest.raises(ValueError, match="already exist"):
        example.main()
    assert calls == ["extract", "persist"]


def test_checkout_rejects_tracked_symlinks(source_repo, tmp_path):
    repo, _ = source_repo
    (repo / "outside").symlink_to("/etc/passwd")
    git(repo, "add", "outside")
    git(repo, "commit", "--quiet", "-m", "symlink")
    git(repo, "remote", "add", "origin", str(repo))
    with pytest.raises(ValueError, match="symlinks"):
        pr_source.verify_checkout(repo, git(repo, "rev-parse", "HEAD"))


@pytest.mark.parametrize("target", ["operator.py", "../outside", "missing", ".git/config", "src", "link"])
def test_checkout_only_allows_internal_tracked_file_links(source_repo, target):
    repo, _ = source_repo
    (repo / "link").symlink_to(target)
    git(repo, "add", "link")
    git(repo, "commit", "--quiet", "-m", "link fixture")
    git(repo, "remote", "add", "origin", str(repo))
    revision = git(repo, "rev-parse", "HEAD")
    if target == "operator.py":
        pr_source.verify_checkout(repo, revision)
    else:
        with pytest.raises(ValueError, match="symlinks"):
            pr_source.verify_checkout(repo, revision)


@pytest.mark.parametrize("kwargs", [{}, {"flaggems_repo": "/tmp/gems", "pr_url": URL}])
def test_workflow_requires_exactly_one_source(kwargs):
    from kernelgen.workflows.catalog_extract import CatalogExtractInput
    with pytest.raises(ValueError, match="exactly one"):
        CatalogExtractInput(operator="add", case_list_path="cases.json", **kwargs)


@pytest.mark.parametrize("source_repo", ["flagos-ai/FlagGems", "flagos-ai/FlagGems-vllm"], indirect=True)
@pytest.mark.parametrize("supplied_cases", [True, False])
def test_workflow_dispatches_pr_to_shared_extractor(source_repo, tmp_path, monkeypatch, supplied_cases):
    from kernelgen.workflows import catalog_extract
    from kernelgen.agents.extractor.flaggems.v62_agent import FlagGemsV62ExtractorOutput
    root = tmp_path / "run"
    (tmp_path / "cases.json").write_text("fixture")
    def collect(repo, operator, workspace, runtime_factory):
        assert not supplied_cases
        assert repo == root / "source/head"
        return tmp_path / "cases.json"
    monkeypatch.setattr(catalog_extract, "prepare_case_list", collect)
    output = FlagGemsV62ExtractorOutput(oracle="def run(x): return x",
        correctness_workloads=[{"name": "accuracy", "inputs": {}}],
        timing_workloads=[{"name": "timing", "inputs": {}}])
    seen = []
    def run(agent, inp, runtime):
        assert inp["source_workspace"] == root / "source"
        seen.append(inp["operator"])
        return output
    monkeypatch.setattr(FlagGemsPRExtractorAgent, "run", run)
    monkeypatch.setattr(catalog_extract, "build_flaggems_v62_accuracy_coverage", lambda *_: {"fixture": True})
    def persist(inp, proposed, catalog):
        assert inp.flaggems_repo == str(root / "source/head")
        assert inp.case_list_path == str(tmp_path / "cases.json")
        assert inp.timing_reference == "flaggems"
        catalog.mkdir()
        (catalog / "manifest.json").write_text("{}")
        return SimpleNamespace(operator_root=catalog / "ops/fused_experts_impl")
    monkeypatch.setattr(catalog_extract, "persist_flaggems_v62_extraction", persist)
    monkeypatch.setattr(catalog_extract, "collect_source_inventory", lambda repo, op: SimpleNamespace(
        implementation_file=Path(repo) / "operator.py", test_files=[], benchmark_files=[]))
    result = catalog_extract.CatalogExtractWorkflow(cwd=str(root), runtime_factory=lambda _: None).run({
        "operator": "fused_experts_impl", "pr_url": source_repo[1].url,
        "case_list_path": str(tmp_path / "cases.json") if supplied_cases else None})
    assert seen == ["fused_experts_impl"] and result.extraction == output
    assert result.accuracy_coverage == {"fixture": True}
    assert result.catalog_path == root / "catalog"
    assert result.source_evidence == [root / "source/head/operator.py", tmp_path / "cases.json", root / "source/source.json"]
    manifest = json.loads((result.catalog_path / "manifest.json").read_text())
    assert manifest["framework_revision"] == source_repo[1].head_sha
    assert manifest["framework_repository"] == source_repo[1].remote
    assert manifest["framework_branch"] == "refs/pull/5623/head"
    assert manifest["framework"] == repository_profile(source_repo[1].repository).framework
