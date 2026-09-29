import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "validate_flaggems_native_parity.py"
)
SPEC = importlib.util.spec_from_file_location("validate_flaggems_native_parity", SCRIPT)
assert SPEC and SPEC.loader
parity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parity
SPEC.loader.exec_module(parity)


class FakeClient:
    def __init__(self, server_url: str, scenario: str = "passed"):
        self.server_url = server_url
        self.scenario = scenario
        self.calls: list[tuple[str, str, dict | None]] = []
        self.status_calls = 0

    def request(self, method, path, payload=None, *, timeout):
        self.calls.append((method, path, payload))
        if path == "/status":
            self.status_calls += 1
            return {
                "api_version": "v6.2",
                "backend": "metax",
                "scheduler": {
                    "device_slots": 2,
                    "healthy": 2,
                    "available": 2,
                    "active": 0,
                    "waiting": 0,
                    "checking": 0,
                    "broken": 0,
                },
            }
        catalog = payload["binding"]["catalog_name"]
        if path == "/inspect":
            case_ids = ["timing-0", "timing-1"]
            if self.scenario == "inspect_mismatch" and catalog == "native":
                case_ids.reverse()
            signature = "(input)"
            if self.scenario == "signature_mismatch" and catalog == "native":
                signature = "(x)"
            fingerprint = f"sha256:{catalog}"
            return {
                "benchmark_fingerprint": fingerprint,
                "candidate_contract": {
                    "entrypoint": "run",
                    "signature": signature,
                },
                "case_list": {
                    "benchmark_fingerprint": fingerprint,
                    "cases": [{"case_id": case_id} for case_id in case_ids],
                },
            }
        if path == "/preflight":
            status = "PASSED"
            if self.scenario == "native_preflight_failed" and catalog == "native":
                status = "RUNTIME_ERROR"
            elif self.scenario == "both_preflight_failed":
                status = "RUNTIME_ERROR"
            elif self.scenario == "native_timeout" and catalog == "native":
                status = "TIMEOUT"
            return {
                "api_version": "v6.2",
                "status": status,
                "benchmark_fingerprint": f"sha256:{catalog}",
                "num_cases": 2,
            }
        if path == "/evaluate":
            rows = [
                {
                    "uuid": "correctness-0",
                    "phase": "correctness",
                    "status": "PASSED",
                },
                {
                    "uuid": "timing-0",
                    "phase": "timing",
                    "status": "PASSED",
                    "speedup": 1.1,
                    "latency_ms": 1.0,
                    "reference_latency_ms": 1.1,
                },
                {
                    "uuid": "timing-1",
                    "phase": "timing",
                    "status": "PASSED",
                    "speedup": 1.2,
                    "latency_ms": 2.0,
                    "reference_latency_ms": 2.4,
                },
            ]
            if catalog == "native":
                rows[0]["uuid"] = "native-correctness-0"
            if (
                self.scenario == "correctness_count_mismatch"
                and catalog == "native"
            ):
                rows.insert(
                    1,
                    {
                        "uuid": "native-correctness-1",
                        "phase": "correctness",
                        "status": "PASSED",
                    },
                )
            if self.scenario == "hack" and catalog == "native":
                is_hack = True
            else:
                is_hack = False
            return {
                "api_version": "v6.2",
                "status": "PASSED",
                "geo_mean": 1.15,
                "num_workloads": len(rows),
                "num_passed": len(rows),
                "is_hack": is_hack,
                "hack_reason": "framework fallback" if is_hack else "",
                "per_workload": rows,
            }
        raise AssertionError(path)


def _args(tmp_path: Path, operators: list[str]):
    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    for operator in operators:
        (candidate_root / f"{operator}.py").write_text(
            "def run(input):\n    return input\n",
            encoding="utf-8",
        )
    return parity._parse_args(
        [
            *[item for operator in operators for item in ("--operator", operator)],
            "--candidate-root",
            str(candidate_root),
            "--server-url",
            "http://kgs.test",
            "--output-root",
            str(tmp_path / "output"),
            "--adapter-catalog",
            "adapter",
            "--native-catalog",
            "native",
            "--expected-backend",
            "metax",
            "--max-workers",
            "2",
        ]
    )


def test_validate_one_passes_matching_candidate_and_workloads(tmp_path: Path):
    args = _args(tmp_path, ["gelu"])
    clients: list[FakeClient] = []

    def client_factory(url):
        client = FakeClient(url)
        clients.append(client)
        return client

    exit_code, report = parity.run(args, client_factory=client_factory)

    assert exit_code == 0
    assert report["counts"] == {"PASSED": 1}
    result = report["results"][0]
    assert result["verdict"] == "PASSED"
    assert result["case_ids"] == ["timing-0", "timing-1"]
    assert result["metrics"]["adapter"]["geo_mean"] == 1.15
    assert result["metrics"]["native"]["num_correctness"] == 1
    assert json.loads(
        (args.output_root / "report.json").read_text(encoding="utf-8")
    ) == report
    operation_paths = [path for client in clients for _, path, _ in client.calls]
    assert operation_paths.count("/inspect") == 2
    assert operation_paths.count("/preflight") == 2
    assert operation_paths.count("/evaluate") == 2


@pytest.mark.parametrize(
    ("scenario", "verdict", "stage"),
    [
        ("inspect_mismatch", "PARITY_FAILED", "inspect"),
        ("signature_mismatch", "PARITY_FAILED", "inspect"),
        ("native_preflight_failed", "PARITY_FAILED", "preflight"),
        ("both_preflight_failed", "CANDIDATE_FAILED", "preflight"),
        ("native_timeout", "BLOCKED", "preflight"),
        ("correctness_count_mismatch", "PARITY_FAILED", "evaluate"),
        ("hack", "CANDIDATE_FAILED", "evaluate"),
    ],
)
def test_gate_classifies_failures(
    tmp_path: Path,
    scenario: str,
    verdict: str,
    stage: str,
):
    args = _args(tmp_path, ["gelu"])

    exit_code, report = parity.run(
        args,
        client_factory=lambda url: FakeClient(url, scenario),
    )

    assert exit_code == 1
    result = report["results"][0]
    assert result["verdict"] == verdict
    assert result["stage"] == stage


def test_batch_preserves_operator_order_and_records_missing_candidate(tmp_path: Path):
    args = _args(tmp_path, ["gelu", "relu"])
    (args.candidate_root / "relu.py").unlink()

    exit_code, report = parity.run(
        args,
        client_factory=lambda url: FakeClient(url),
    )

    assert exit_code == 1
    assert [row["operator"] for row in report["results"]] == ["gelu", "relu"]
    assert [row["verdict"] for row in report["results"]] == [
        "PASSED",
        "BLOCKED",
    ]
    assert report["results"][1]["stage"] == "input"


def test_terminal_summary_is_reused_only_for_same_identity(tmp_path: Path):
    args = _args(tmp_path, ["gelu"])
    first_clients: list[FakeClient] = []

    def first_factory(url):
        client = FakeClient(url)
        first_clients.append(client)
        return client

    assert parity.run(args, client_factory=first_factory)[0] == 0
    second_clients: list[FakeClient] = []

    def second_factory(url):
        client = FakeClient(url)
        second_clients.append(client)
        return client

    assert parity.run(args, client_factory=second_factory)[0] == 0

    assert [path for _, path, _ in second_clients[0].calls] == [
        "/status",
        "/status",
    ]
    (args.candidate_root / "gelu.py").write_text(
        "def run(input):\n    return -input\n",
        encoding="utf-8",
    )
    with pytest.raises(parity.GateError, match="identity differs"):
        parity.validate_one(
            operator="gelu",
            candidate_path=args.candidate_root / "gelu.py",
            server_url=args.server_url,
            output_root=args.output_root,
            adapter_catalog=args.adapter_catalog,
            native_catalog=args.native_catalog,
            settings={},
            language="triton",
            transport_timeout=60,
            force=False,
            client_factory=lambda url: FakeClient(url),
        )


def test_read_operators_accepts_markdown_and_rejects_unsafe_names(tmp_path: Path):
    valid = tmp_path / "operators.md"
    valid.write_text(
        "| 算子 | 状态 |\n| --- | --- |\n| `gelu` | pending |\n",
        encoding="utf-8",
    )
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("gelu;touch_bad\n", encoding="utf-8")

    assert parity.read_operators(valid) == ["gelu"]
    with pytest.raises(parity.GateError, match="invalid operator"):
        parity.read_operators(invalid)
