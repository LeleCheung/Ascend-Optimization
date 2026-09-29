#!/usr/bin/env python3
"""Reconcile original low-speedup rows with archived timings; never run GPU work."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def geo(values):
    return math.exp(statistics.mean(math.log(x) for x in values)) if values else None


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def signature(dtype, shape):
    return json.dumps([str(dtype).removeprefix("torch."), shape], sort_keys=True, separators=(",", ":"))


def recover(row, ledger, native):
    best = next(r for r in ledger["rounds"] if r["round_num"] == ledger["best_round"])
    expected = row["candidate_sha256"]
    if sha(best["solution"]["code"]) != expected or native["candidate_sha256"] != expected:
        raise ValueError(f"Candidate identity mismatch: {row['id']}")
    for name in ("reference_ratio_geo", "candidate_ratio_geo", "paired_old_geo", "paired_new_geo", "speedup_ratio_geo"):
        row.pop(name, None)
    old = [w for w in best["evaluation"]["workloads"] if positive(w.get("latency_ms")) and positive(w.get("reference_latency_ms"))]
    new = (native.get("timing") or {}).get("cases", [])
    index = collections.defaultdict(list)
    for w in new:
        if positive(w.get("latency")) and positive(w.get("latency_base")):
            index[signature(w["dtype"], w["shape_detail"])].append(w)
    counts = collections.Counter(signature(w["axes"].get("dtype"), w["axes"].get("shape_detail")) for w in old)
    pairs = []
    spikes = []
    for w in old:
        key = signature(w["axes"].get("dtype"), w["axes"].get("shape_detail"))
        others = [v["reference_latency_ms"] for r in ledger["rounds"] if r["round_num"] != ledger["best_round"] for v in (r.get("evaluation") or {}).get("workloads", []) if v.get("reference_latency_ms") and signature(v["axes"].get("dtype"), v["axes"].get("shape_detail")) == key]
        if len(others) >= 3 and w["reference_latency_ms"] > 5 * statistics.median(others):
            spikes.append({"key": key, "reference_spike_ratio": w["reference_latency_ms"] / statistics.median(others)})
        if w["axes"].get("shape_detail") is None or counts[key] != 1 or len(index[key]) != 1:
            continue
        nw = index[key][0]
        pairs.append({"key": key, "old_uuid": w["uuid"], "old_ref_ms": w["reference_latency_ms"], "new_ref_ms": nw["latency_base"], "old_candidate_ms": w["latency_ms"], "new_candidate_ms": nw["latency"], "reference_ratio": nw["latency_base"] / w["reference_latency_ms"], "candidate_ratio": nw["latency"] / w["latency_ms"]})
    row.update(best_round=ledger["best_round"], historical_geo=best["evaluation"]["geo_mean"], historical_candidate_sha256=expected, historical_hash_matches=True, historical_num_timing_cases=len(old), native_num_timing_cases=len(new), matched_cases=len(pairs), paired=pairs, full_timing_signature_match=bool(pairs) and len(pairs) == len(old) == len(new), recovered_reference_spikes=spikes)
    if pairs:
        row.update(reference_ratio_geo=geo([x["reference_ratio"] for x in pairs]), candidate_ratio_geo=geo([x["candidate_ratio"] for x in pairs]), paired_old_geo=geo([x["old_ref_ms"] / x["old_candidate_ms"] for x in pairs]), paired_new_geo=geo([x["new_ref_ms"] / x["new_candidate_ms"] for x in pairs]), comparison_state="same_candidate_recorded_signature_pairs")
    else:
        row["comparison_state"] = "no_unambiguous_recorded_signature_overlap"


def build(mainline, audit_root, runs_root):
    audit = json.loads((audit_root / "performance-attribution/analysis.json").read_text())
    previous = {r["id"]: r for r in audit["rows"]}
    selected = [r for r in mainline["rows"] if isinstance(r["reported_speedup"], (int, float)) and 0 < r["reported_speedup"] < .8]
    rows = []
    for original in selected:
        row = dict(previous[original["id"]])
        row.update(operator=original["operator"], reported_speedup=original["reported_speedup"], reported_note=original["reported_note"], exact_original_failure_closed=False)
        if row["candidate_sha256"] != original["candidate_sha256"]:
            raise ValueError(f"Archive identity mismatch: {row['id']}")
        rows.append(row)
    by_sha = {r["candidate_sha256"]: r for r in rows}
    names = {r["operator"] for r in rows}
    for path in sorted((runs_root / "flaggems-adapter-version").glob("*/batch/*/workspace/definitions/*/.ledger.json")):
        if path.parent.name not in names:
            continue
        ledger = json.loads(path.read_text())
        row = by_sha.get(sha(ledger.get("best_code", "")))
        if row is None or row.get("ledger_path"):
            continue
        native = json.loads((audit_root / row["native_result_path"]).read_text())
        recover(row, ledger, native)
        row.update(ledger_path=str(path), historical_ledger_recovered=True)
    for row in rows:
        # Binding repro is evidence of this audit's invalid baseline, not of the intern's failure.
        row["baseline_identity_invalid_in_current_audit"] = row["id"] == "pingtouge:8"
        row["usable_for_recorded_latency_comparison"] = bool(row["paired"]) and not row["baseline_identity_invalid_in_current_audit"]
    dispatch = next(r for r in rows if r["id"] == "pingtouge:2")
    rest = [r for r in rows if r is not dispatch]
    usable = [r for r in rest if r["usable_for_recorded_latency_comparison"]]
    stable = [r["id"] for r in usable if r.get("full_timing_signature_match") and .94 <= r["reference_ratio_geo"] <= 1.06 and .94 <= r["candidate_ratio_geo"] <= 1.06]
    index_groups = json.loads((audit_root / "index-select-workload-mismatch.json").read_text())["groups"]["pingtouge"]
    return {"scope": "Original non-Ascend low-speedup rows, excluding dispatch from main counts; current pinned native/core is separate from original intern retest.", "limitations": "No new device measurements. Recorded shape/dtype/scalars and candidate SHA do not establish identical tensor values, strides, environment, invocation path or trustworthy baseline semantics. Usable pairs are descriptive, not acceptance or causal closure.", "source_mainline_sha256": sha(json.dumps(mainline, ensure_ascii=False, sort_keys=True)), "source_analysis_sha256": hashlib.sha256((audit_root / "performance-attribution/analysis.json").read_bytes()).hexdigest(), "summary": {"original_low_speedup_rows": len(rows), "excluding_dispatch": len(rest), "recovered_historical_ledgers": sum(r.get("historical_ledger_recovered", False) for r in rest), "usable_recorded_pairs_rows": len(usable), "invalid_baseline_rows": sum(r["baseline_identity_invalid_in_current_audit"] for r in rest), "without_recorded_pairs": sum(not r["paired"] for r in rest), "full_signature_and_both_geos_within_6_percent": stable}, "dispatch_record": dispatch, "rows": rest, "reference_outliers_in_scope": [x for x in audit["historical_reference_outliers"] if x["id"] in {r["id"] for r in rest}], "index_select_native_dim_groups": index_groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.mainline.read_text()), args.audit_root, args.runs_root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
