"""Tests for the Knowledge-enabled multi-operator campaign command."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "kernel_gen"
    / "run_campaign.py"
)
RUN_EXAMPLE = SCRIPT.with_name("run_example.py")


def test_dry_run_uses_epoch_barrier_commands(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--definitions",
            "op_a",
            "op_b",
            "--workspace-root",
            str(tmp_path / "runs"),
            "--knowledge-catalog-path",
            str(tmp_path / "catalog"),
            "--n-epoch",
            "2",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    assert output.count("--knowledge-mode read_write_v1") == 4
    assert output.count("--knowledge-reviewer-mode off") == 4
    assert output.count("--knowledge-run-id runs--op_a") == 2
    assert output.count("--knowledge-run-id runs--op_b") == 2
    assert output.count("--catalog-name flaggems-adapter-definitions") == 4
    assert output.count("--start-mode fresh") == 2
    assert output.count("--start-mode resume") == 2
    assert output.count("--start-epoch 1 --n-epoch 1") == 2
    assert output.count("--start-epoch 2 --n-epoch 2") == 2
    assert output.count("--max-round 15") == 4
    assert "--finalize-epoch" not in output
    assert output.index("# 1R") < output.index("# 2R")


def test_dry_run_forwards_explicit_reviewer_mode(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--definitions",
            "op_a",
            "--workspace-root",
            str(tmp_path / "runs"),
            "--knowledge-catalog-path",
            str(tmp_path / "catalog"),
            "--knowledge-reviewer-mode",
            "shadow",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("--knowledge-reviewer-mode shadow") == 1


def test_dry_run_forwards_codex_runtime_and_model(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--definitions",
            "op_a",
            "--workspace-root",
            str(tmp_path / "runs"),
            "--knowledge-catalog-path",
            str(tmp_path / "catalog"),
            "--runtime",
            "codex",
            "--model",
            "gpt-codex-test",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--runtime codex" in completed.stdout
    assert "--model gpt-codex-test" in completed.stdout


def test_dry_run_forwards_read_only_mode_and_external_indexes(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--definitions",
            "op_a",
            "--workspace-root",
            str(tmp_path / "runs"),
            "--knowledge-catalog-path",
            str(tmp_path / "catalog"),
            "--knowledge-mode",
            "read_only_v1",
            "--knowledge-derived-path",
            str(tmp_path / "derived"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--knowledge-mode read_only_v1" in completed.stdout
    assert (
        f"--knowledge-derived-path {tmp_path / 'derived'}"
        in completed.stdout
    )


def test_dry_run_forwards_definition_scoped_seed_only_to_epoch_one(tmp_path):
    seed = tmp_path / "op_a.py"
    seed.write_text("def run(input):\n    return input\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--definitions",
            "op_a",
            "--workspace-root",
            str(tmp_path / "runs"),
            "--knowledge-catalog-path",
            str(tmp_path / "catalog"),
            "--agents-per-operator",
            "2",
            "--n-epoch",
            "2",
            "--seed-code-path",
            f"op_a={seed}",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count(f"--seed-code-path {seed}") == 1
    epoch_one, epoch_two = completed.stdout.split("# 2R", 1)
    assert "--seed-code-path" in epoch_one
    assert "--seed-code-path" not in epoch_two


def test_campaign_rejects_reviewer_in_read_only_mode(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--definitions",
            "op_a",
            "--workspace-root",
            str(tmp_path / "runs"),
            "--knowledge-catalog-path",
            str(tmp_path / "catalog"),
            "--knowledge-mode",
            "read_only_v1",
            "--knowledge-reviewer-mode",
            "shadow",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "read_only_v1 requires --knowledge-reviewer-mode off" in (
        completed.stderr
    )


def test_run_example_requires_catalog_for_active_reviewer_mode():
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_EXAMPLE),
            "--definition",
            "unused-definition",
            "--knowledge-reviewer-mode",
            "enforce",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--knowledge-catalog-path is required" in completed.stderr


def test_campaign_completion_requires_publish_and_synthesis(tmp_path):
    published_epoch = runpy.run_path(str(SCRIPT))["_published_epoch"]
    epoch = tmp_path / "op_a" / "1R"
    for index in range(2):
        agent = epoch / f"agent{index}"
        result = agent / ".kernelgen" / "knowledge" / "publish-result.json"
        result.parent.mkdir(parents=True)
        (agent / ".ledger.json").write_text("{}", encoding="utf-8")
        result.write_text(
            json.dumps({"status": "noop"}),
            encoding="utf-8",
        )

    assert published_epoch(tmp_path / "op_a", 1, 2) is False

    synthesis = epoch / "synthesis" / "synthesis.json"
    synthesis.parent.mkdir()
    synthesis.write_text("{}", encoding="utf-8")
    assert published_epoch(tmp_path / "op_a", 1, 2) is True

    failed_result = (
        epoch
        / "agent1"
        / ".kernelgen"
        / "knowledge"
        / "publish-result.json"
    )
    failed_result.write_text(
        json.dumps({"status": "rejected"}),
        encoding="utf-8",
    )
    assert published_epoch(tmp_path / "op_a", 1, 2) is False


def test_partition_epoch_skips_only_durable_completions(tmp_path):
    module = runpy.run_path(str(SCRIPT))
    partition_epoch = module["_partition_epoch"]
    epoch = tmp_path / "op_a" / "1R"
    for index in range(2):
        agent = epoch / f"agent{index}"
        result = agent / ".kernelgen" / "knowledge" / "publish-result.json"
        result.parent.mkdir(parents=True)
        (agent / ".ledger.json").write_text("{}", encoding="utf-8")
        result.write_text(json.dumps({"status": "noop"}), encoding="utf-8")
    synthesis = epoch / "synthesis" / "synthesis.json"
    synthesis.parent.mkdir()
    synthesis.write_text("{}", encoding="utf-8")

    completed, pending = partition_epoch(
        ["op_a", "op_b"],
        tmp_path,
        1,
        2,
        skip_completed=True,
    )
    assert completed == ["op_a"]
    assert pending == ["op_b"]

    completed, pending = partition_epoch(
        ["op_a", "op_b"],
        tmp_path,
        1,
        2,
        skip_completed=False,
    )
    assert completed == []
    assert pending == ["op_a", "op_b"]

    dry_run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--definitions",
            "op_a",
            "op_b",
            "--workspace-root",
            str(tmp_path),
            "--knowledge-catalog-path",
            str(tmp_path / "catalog"),
            "--agents-per-operator",
            "2",
            "--skip-completed",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "# SKIP op_a (durably complete)" in dry_run.stdout
    assert "--definition op_a " not in dry_run.stdout
    assert "--definition op_b " in dry_run.stdout


def test_campaign_accepts_partial_success_manifest(tmp_path):
    published_epoch = runpy.run_path(str(SCRIPT))["_published_epoch"]
    epoch = tmp_path / "op_a" / "1R"
    agent = epoch / "agent1"
    result = agent / ".kernelgen" / "knowledge" / "publish-result.json"
    result.parent.mkdir(parents=True)
    (agent / ".ledger.json").write_text("{}", encoding="utf-8")
    result.write_text(json.dumps({"status": "noop"}), encoding="utf-8")
    synthesis = epoch / "synthesis" / "synthesis.json"
    synthesis.parent.mkdir()
    synthesis.write_text("{}", encoding="utf-8")
    (epoch / "epoch-completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "attempted_agents": ["agent0", "agent1"],
                "successful_agents": ["agent1"],
                "failed_agents": [
                    {
                        "name": "agent0",
                        "error_type": "TimeoutError",
                        "error": "provider timed out",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert published_epoch(tmp_path / "op_a", 1, 2) is True

    malformed = json.loads(
        (epoch / "epoch-completion.json").read_text(encoding="utf-8")
    )
    malformed["failed_agents"] = []
    (epoch / "epoch-completion.json").write_text(
        json.dumps(malformed),
        encoding="utf-8",
    )
    assert published_epoch(tmp_path / "op_a", 1, 2) is False


def test_campaign_rejects_single_agent_multi_epoch(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--definitions",
            "op_a",
            "--workspace-root",
            str(tmp_path / "runs"),
            "--knowledge-catalog-path",
            str(tmp_path / "catalog"),
            "--agents-per-operator",
            "1",
            "--n-epoch",
            "2",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "multi-epoch runs require --agents-per-operator >= 2" in (
        completed.stderr
    )
