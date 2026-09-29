import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/experiments/validate_nvidia_candidate_portability.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_nvidia_candidate_portability", SCRIPT_PATH
)
assert SPEC and SPEC.loader
portability = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = portability
SPEC.loader.exec_module(portability)


def _results(operator, *, reference="通过", timing="通过", status="未跑", hack="未检查"):
    speed = "1.200x" if status == "成功" else "—"
    return (
        "| 总算子数 | reference通过数 | Gems 可计时数 | 已完成数 | "
        "达标数（加速比≥0.8） | 未达标数（加速比<0.8） | "
        "失败数（未通过正确性测试） |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        "| 1 | 1 | 1 | 0 | 0 | 0 | 0 |\n\n"
        "| 算子名 | Reference | Gems 可计时 | 优化状态 | 加速比 | "
        "Hack 情况 | 原因 | 后续方向 | code_path |\n"
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |\n"
        f"| `{operator}` | {reference} | {timing} | {status} | {speed} | "
        f"{hack} | — | 原方向。 | — |\n"
    )


def _repo(
    tmp_path,
    *,
    target_reference="通过",
    target_timing="通过",
    nvidia_hack="未发现",
):
    root = tmp_path / "repo"
    todo = root / "kernel_todo_v2"
    (todo / "nvidia").mkdir(parents=True)
    (todo / "muxi").mkdir()
    (todo / "pytest_conversion_inventory.json").write_text(
        json.dumps(
            {
                "operators": [
                    {
                        "source_operator": "foo.out",
                        "operator": "foo_out",
                        "chips": ["nvidia", "muxi"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (todo / "nvidia/results.md").write_text(
        _results("foo.out", status="成功", hack=nvidia_hack), encoding="utf-8"
    )
    (todo / "muxi/results.md").write_text(
        _results(
            "foo.out", reference=target_reference, timing=target_timing
        ),
        encoding="utf-8",
    )
    candidate = tmp_path / "best.py"
    candidate.write_text("def run(x):\n    return x\n", encoding="utf-8")
    return root, candidate


class FakeClient:
    def __init__(self, server_url, *, outcome="reusable"):
        self.server_url = server_url
        self.outcome = outcome
        self.calls = []

    def request(self, method, path, payload=None, *, timeout):
        self.calls.append((method, path, payload, timeout))
        if path == "/status":
            busy = self.outcome == "busy"
            return {
                "api_version": "v6.2",
                "backend": "metax",
                "evaluation_adapters": ["native", "flaggems"],
                "scheduler": {
                    "device_slots": 2,
                    "healthy": 2,
                    "available": 1 if busy else 2,
                    "active": 1 if busy else 0,
                    "waiting": 0,
                    "checking": 0,
                    "broken": 0,
                },
            }
        if path == "/inspect":
            if self.outcome == "transport":
                raise portability.HttpError("synthetic inspect failure")
            return {
                "benchmark_fingerprint": "sha256:fixture",
                "candidate_contract": {"entrypoint": "run", "signature": "(x)"},
                "case_list": {
                    "benchmark_fingerprint": "sha256:fixture",
                    "cases": [
                        {"case_id": "timing-0"},
                        {"case_id": "timing-1"},
                    ]
                },
            }
        if path == "/preflight":
            return {
                "api_version": "v6.2",
                "status": "PASSED",
                "benchmark_fingerprint": "sha256:fixture",
                "num_cases": 2,
            }
        if path == "/evaluate":
            speedup = 0.7 if self.outcome == "slow" else 1.1
            return {
                "api_version": "v6.2",
                "server_backend": "metax",
                "status": "PASSED",
                "geo_mean": speedup,
                "min_speedup": speedup,
                "num_workloads": 3,
                "num_passed": 3,
                "is_hack": False,
                "per_workload": [
                    {
                        "uuid": "correctness-0",
                        "phase": "correctness",
                        "status": "PASSED",
                    },
                    {
                        "uuid": "timing-0",
                        "phase": "timing",
                        "status": "PASSED",
                        "speedup": speedup,
                        "latency_ms": 1.0,
                        "reference_latency_ms": speedup,
                    },
                    {
                        "uuid": "timing-1",
                        "phase": "timing",
                        "status": "PASSED",
                        "speedup": speedup,
                        "latency_ms": 2.0,
                        "reference_latency_ms": 2 * speedup,
                    },
                ],
            }
        raise AssertionError(path)


def _args(root, candidate, run_root):
    return portability._parse_args(
        [
            "--operator",
            "foo.out",
            "--candidate-code",
            str(candidate),
            "--target",
            "muxi=http://kgs.test",
            "--allow-partial-target-set",
            "--repo-root",
            str(root),
            "--run-root",
            str(run_root),
        ]
    )


def test_reusable_candidate_is_archived_and_updates_results(tmp_path):
    root, candidate = _repo(tmp_path)
    client = FakeClient("http://kgs.test")

    exit_code, report = portability.run(
        _args(root, candidate, root / "runs/kernel_todo_v2/run"),
        client_factory=lambda _: client,
    )

    assert exit_code == 0
    assert report["verdict_counts"] == {
        "BLOCKED": 0,
        "NEEDS_SPECIALIZATION": 0,
        "REUSABLE": 1,
    }
    copied = root / "kernel_todo_v2/muxi/codes/foo_out.py"
    assert copied.read_bytes() == candidate.read_bytes()
    results = (root / "kernel_todo_v2/muxi/results.md").read_text(encoding="utf-8")
    assert "| `foo.out` | 通过 | 通过 | 成功 | 1.100x |" in results
    assert "跨芯片验证" in results
    assert "| 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |" in results
    portability_results = (
        root / "kernel_todo_v2/nvidia/portability_results.md"
    ).read_text(encoding="utf-8")
    assert "| 已验证 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |" in portability_results
    assert "| 可直接复用 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |" in portability_results
    assert (
        "| NVIDIA 算子 | 沐曦 | 海光 | 摩尔线程 | 天数智芯 | "
        "平头哥 | 昇腾 | 昆仑芯 | 燧原 |"
        in portability_results
    )
    assert "| `foo.out`（`foo_out`） | 可直接复用 · 1.100x |" in portability_results
    assert "[证据]" not in portability_results
    assert "SHA" not in portability_results
    portability_details = json.loads(
        (root / "kernel_todo_v2/nvidia/portability_results.json").read_text(
            encoding="utf-8"
        )
    )
    detail = portability_details["operators"]["foo.out"]["targets"]["muxi"]
    assert detail["verdict"] == "REUSABLE"
    assert detail["geo_mean"] == 1.1
    assert detail["evidence"].endswith("targets/muxi/summary.json")
    assert "candidate_sha256" not in detail
    assert [call[1] for call in client.calls] == [
        "/status",
        "/inspect",
        "/preflight",
        "/evaluate",
        "/inspect",
        "/status",
    ]


def test_no_update_review_can_reuse_summary_and_then_update(tmp_path):
    root, candidate = _repo(tmp_path)
    args = _args(root, candidate, root / "runs/kernel_todo_v2/run")
    args.no_update_results = True

    exit_code, first = portability.run(
        args, client_factory=lambda url: FakeClient(url)
    )

    assert exit_code == 0
    assert first["updates"] == []
    assert first["nvidia_updates"] == []
    assert not (
        root / "kernel_todo_v2/nvidia/portability_results.md"
    ).exists()
    assert "| `foo.out` | 通过 | 通过 | 未跑 |" in (
        root / "kernel_todo_v2/muxi/results.md"
    ).read_text(encoding="utf-8")

    args.no_update_results = False

    def unexpected_client(_):
        raise AssertionError("terminal target summary should be reused")

    exit_code, second = portability.run(args, client_factory=unexpected_client)

    assert exit_code == 0
    assert second["updates"][0]["action"] == "reused"
    assert second["nvidia_updates"][0]["action"] == "created"
    assert "| `foo.out` | 通过 | 通过 | 成功 |" in (
        root / "kernel_todo_v2/muxi/results.md"
    ).read_text(encoding="utf-8")


def test_source_hack_note_and_other_busy_slot_are_not_gates(tmp_path):
    root, candidate = _repo(tmp_path, nvidia_hack="未检查")

    exit_code, report = portability.run(
        _args(root, candidate, root / "runs/kernel_todo_v2/run"),
        client_factory=lambda url: FakeClient(url, outcome="busy"),
    )

    assert exit_code == 0
    assert report["targets"]["muxi"]["verdict"] == "REUSABLE"


def test_slow_candidate_enters_nonterminal_specialization_queue(tmp_path):
    root, candidate = _repo(tmp_path)

    exit_code, report = portability.run(
        _args(root, candidate, root / "runs/kernel_todo_v2/run"),
        client_factory=lambda url: FakeClient(url, outcome="slow"),
    )

    assert exit_code == 2
    assert report["targets"]["muxi"]["verdict"] == "NEEDS_SPECIALIZATION"
    assert not (root / "kernel_todo_v2/muxi/codes/foo_out.py").exists()
    results = (root / "kernel_todo_v2/muxi/results.md").read_text(encoding="utf-8")
    assert "| `foo.out` | 通过 | 通过 | 需要特化 | 0.700x |" in results
    assert "| 1 | 1 | 1 | 0 | 0 | 0 | 0 |" in results
    portability_results = (
        root / "kernel_todo_v2/nvidia/portability_results.md"
    ).read_text(encoding="utf-8")
    assert "| `foo.out`（`foo_out`） | 需要特化 · 0.700x |" in portability_results


def test_flat_portability_ledger_is_migrated_to_multi_chip_matrix(tmp_path):
    root = tmp_path / "repo"
    nvidia_root = root / "kernel_todo_v2/nvidia"
    nvidia_root.mkdir(parents=True)
    path = nvidia_root / "portability_results.md"
    path.write_text(
        "| NVIDIA 算子 | Definition | 目标芯片 | 验证状态 | 加速比 | "
        "Hack 情况 | 原因 | candidate SHA256 | 证据 | 验证时间 |\n"
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |\n"
        "| `foo.out` | `foo_out` | muxi | 可直接复用 | 1.100x | "
        "未发现 | all passed | "
        "`1111111111111111111111111111111111111111111111111111111111111111` | "
        "[summary](muxi.json) | 2026-08-29T00:00:00+00:00 |\n",
        encoding="utf-8",
    )

    update = portability._sync_nvidia_portability_result(
        repo_root=root,
        source_operator="foo.out",
        definition="foo_out",
        run_root=root / "runs/kernel_todo_v2/run",
        summary={
            "chip": "haiguang",
            "verdict": "BLOCKED",
            "reason": "target unavailable",
            "geo_mean": None,
            "is_hack": False,
        },
    )

    assert update["action"] == "created"
    markdown = path.read_text(encoding="utf-8")
    assert "| 已验证 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |" in markdown
    assert "| 可直接复用 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |" in markdown
    assert "| 验证阻塞 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |" in markdown
    assert "| `foo.out`（`foo_out`） | 可直接复用 · 1.100x | 验证阻塞 |" in markdown
    details = json.loads(
        (nvidia_root / "portability_results.json").read_text(encoding="utf-8")
    )
    targets = details["operators"]["foo.out"]["targets"]
    assert targets["muxi"]["verdict"] == "REUSABLE"
    assert targets["haiguang"]["reason"] == "target unavailable"


def test_target_without_results_row_keeps_artifacts_without_adding_row(tmp_path):
    root, candidate = _repo(tmp_path)
    target_results = root / "kernel_todo_v2/muxi/results.md"
    target_results.write_text(_results("different.op"), encoding="utf-8")

    exit_code, report = portability.run(
        _args(root, candidate, root / "runs/kernel_todo_v2/run"),
        client_factory=lambda url: FakeClient(url),
    )

    assert exit_code == 0
    assert report["targets"]["muxi"]["has_results_row"] is False
    assert report["updates"] == [
        {"chip": "muxi", "action": "artifact_only", "reason": "operator row missing"}
    ]
    assert "foo.out" not in target_results.read_text(encoding="utf-8")


def test_baseline_block_does_not_become_candidate_failure(tmp_path):
    root, candidate = _repo(
        tmp_path, target_reference="未通过", target_timing="无法计时"
    )

    def unexpected_client(_):
        raise AssertionError("baseline-blocked target must not contact KGS")

    exit_code, report = portability.run(
        _args(root, candidate, root / "runs/kernel_todo_v2/run"),
        client_factory=unexpected_client,
    )

    assert exit_code == 3
    assert report["targets"]["muxi"]["verdict"] == "BLOCKED"
    results = (root / "kernel_todo_v2/muxi/results.md").read_text(encoding="utf-8")
    assert "| `foo.out` | 未通过 | 无法计时 | 未跑 |" in results
    assert "跨芯片验证阻塞" in results


def test_transport_failure_keeps_optimization_status(tmp_path):
    root, candidate = _repo(tmp_path)

    exit_code, report = portability.run(
        _args(root, candidate, root / "runs/kernel_todo_v2/run"),
        client_factory=lambda url: FakeClient(url, outcome="transport"),
    )

    assert exit_code == 3
    summary = report["targets"]["muxi"]
    assert summary["verdict"] == "BLOCKED"
    assert summary["stage"] == "transport/server"
    results = (root / "kernel_todo_v2/muxi/results.md").read_text(encoding="utf-8")
    assert "| `foo.out` | 通过 | 通过 | 未跑 |" in results
    assert "synthetic inspect failure" in results


def test_existing_different_target_code_requires_explicit_overwrite(tmp_path):
    root, candidate = _repo(tmp_path)
    target_code = root / "kernel_todo_v2/muxi/codes/foo_out.py"
    target_code.parent.mkdir()
    target_code.write_text("def run(x):\n    return x + 1\n", encoding="utf-8")

    exit_code, report = portability.run(
        _args(root, candidate, root / "runs/kernel_todo_v2/run"),
        client_factory=lambda url: FakeClient(url),
    )

    assert exit_code == 3
    assert report["updates"] == [
        {
            "chip": "muxi",
            "action": "update_blocked",
            "reason": "target codes file exists with different content",
        }
    ]
    assert target_code.read_text(encoding="utf-8") == "def run(x):\n    return x + 1\n"
    results = (root / "kernel_todo_v2/muxi/results.md").read_text(encoding="utf-8")
    assert "| `foo.out` | 通过 | 通过 | 未跑 |" in results
