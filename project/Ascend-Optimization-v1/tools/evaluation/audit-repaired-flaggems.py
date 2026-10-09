#!/usr/bin/env python3
"""核验三项修复交付：完整三轮、源码可重建、逐 case 延迟与环境快照。"""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote


EXPECTED = {"_prelu_kernel_backward": (21, 9),
            "batch_norm_backward": (30, 15),
            "smooth_l1_loss_backward": (291, 12)}
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(tool, *args):
    completed = subprocess.run([sys.executable, "-B", str(HERE / tool), *map(str, args)],
                               capture_output=True, text=True, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="含 validated-v5 的报告根目录")
    args = parser.parse_args()
    root = args.report.resolve()
    final = root / "validated-v5"
    historical = root.parent / "flaggems-comparison-20261009"
    project = root.parents[2]

    # 汇总器从原始请求和结果重新验证全套合同、源码、UUID、axes 和正确性。
    run("summarize-repaired-flaggems.py", final)
    rows = read(final / "comparison.json")
    assert {r["operator"] for r in rows} == set(EXPECTED) and len(rows) == 3
    for row in rows:
        assert len(row["repeats"]) == 3
        for repeat in row["repeats"]:
            assert (repeat["correctness_cases"], repeat["timing_cases"]) == EXPECTED[row["operator"]]
            assert repeat["logical_device"] == "npu:0"
    experiment = read(final / "measurements/experiment.json")
    assert int(experiment["physical_device"]) == 7
    assert all(int(e["physical_device"]) == 7 for e in experiment["batch_experiments"])

    with (final / "per-case.csv").open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))
    assert len(cases) == 3 * sum(n[1] for n in EXPECTED.values()) == 108
    for case in cases:
        folder = final / "measurements" / case["operator"]
        for variant, column in (("repaired-flaggems", "baseline_ms"), ("candidate", "candidate_ms")):
            result = read(folder / f'{variant}-{case["repeat"]}.result.json')
            workload = next(w for w in result["per_workload"] if w["uuid"] == case["uuid"])
            assert float(case[column]) == workload["latency_ms"]
        assert float(case["speedup_vs_repaired_flaggems"]) == float(case["baseline_ms"]) / float(case["candidate_ms"])

    original_sha = {}
    for row in read(final / "repairs/repairs.json"):
        op = row["operator"]
        original = (historical / "source-audit" / op / "flaggems-original-source.py").read_bytes()
        assert original == (final / "repairs" / op / "original.py").read_bytes()
        assert hashlib.sha256(original).hexdigest() == row["original_sha256"]
        original_sha[op] = row["original_sha256"]
        for name in ("original.py", "repaired.py"):
            ast.parse((final / "repairs" / op / name).read_text(encoding="utf-8-sig"))
        ast.parse((final / "inputs" / op / "candidate.py").read_text(encoding="utf-8-sig"))

    # 按 README 重建输入。早期 Windows 写入留下 CRLF/CRCRLF；仅归一化换行比较。
    with tempfile.TemporaryDirectory(prefix="flaggems-repaired-audit-") as directory:
        temp = Path(directory)
        run("repair-flaggems-baselines.py", "--historical-report", historical,
            "--output", temp / "repairs", "--prelu-reduction", "compensated")
        run("prepare-repaired-flaggems-inputs.py", "--historical-report", historical,
            "--repairs", temp / "repairs", "--output", temp / "inputs")
        run("repair-prelu-candidate.py", "--input", temp / "inputs/_prelu_kernel_backward/candidate.py",
            "--output", temp / "accurate.py")
        for op in EXPECTED:
            for name in ("repaired.py", "candidate.py"):
                rebuilt = temp / "inputs" / op / name
                if op == "_prelu_kernel_backward" and name == "candidate.py":
                    rebuilt = temp / "accurate.py"
                generated = rebuilt.read_bytes().replace(b"\r", b"")
                measured = (final / "inputs" / op / name).read_bytes().replace(b"\r", b"")
                assert generated == measured, (op, name)

    environment = read(root / "environment-current.json")
    before = read(root / "shared-checkout-verification.json")
    assert environment["flaggems_commit"] == before["flaggems_commit"]
    assert environment["source_sha256"] == before["source_sha256"]
    assert set(environment["source_sha256"].values()) == set(original_sha.values())
    assert len(environment["batches"]) == 5
    assert all(v["exit_code"] == "0" and not v["server_alive"] for v in environment["batches"].values())

    docs = [root / "README.md", root / "周报精简版.md", root / "交付核验.md", final / "README.md",
            root / "diagnostics/README.md", project / "archive/README.md",
            project / "tools/README.md", historical / "README.md"]
    docs.extend(project / "archive/operators" / op.lstrip("_") / "README.md" for op in EXPECTED)
    links = 0
    for doc in docs:
        for dest in re.findall(r"\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if dest.startswith(("http:", "https:", "#")):
                continue
            target = (doc.parent / unquote(dest.split("#")[0])).resolve()
            if target == final / "delivery-audit.json":
                continue  # 本次核验全部通过后生成的输出。
            assert target.exists(), (str(doc), dest)
            links += 1

    audit = {"operators": {r["operator"]: {"correctness": EXPECTED[r["operator"]][0],
                "timing": EXPECTED[r["operator"]][1],
                "speedup": r["median_speedup_vs_repaired_flaggems"]} for r in rows},
             "complete_rounds_per_variant": 3, "timing_rows": len(cases),
             "physical_device": 7, "all_sources_match_requests": True,
             "all_sources_rebuilt_identically_after_newline_normalization": True,
             "original_sources_match_historical_evidence": True,
             "all_own_servers_stopped_in_environment_snapshot": True,
             "shared_original_sources_unchanged_in_environment_snapshot": True,
             "local_markdown_links_checked": links}
    (final / "delivery-audit.json").write_bytes((json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
