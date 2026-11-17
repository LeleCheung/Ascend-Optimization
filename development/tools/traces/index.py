"""Bind saved source occurrences, measurements and presentation annotations."""
import json
from pathlib import Path

from .importer import build_session, digest, read, ref


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_occurrence(session, sha):
    # A measured output is anchored to the producing invocation, including its
    # last literal edit. SHA equality alone must not merge rollback occurrences.
    matches = [call for call in session["calls"] if call["output_sha256"] == sha and call["edits"]]
    if len(matches) > 1:
        raise ValueError(f"Multiple producing occurrences for {sha}; an explicit evaluation anchor is required")
    if matches:
        call = matches[0]
        edits = [t for t in session["transitions"] if t["call_id"] == call["id"]]
        return dict(call_id=call["id"], transition_id=edits[-1]["id"])
    if sha == session["starter_sha256"]:
        return dict(call_id=None, transition_id=None, role="provided_starter")
    raise ValueError(f"No source occurrence for {sha}")


def measurement(path, session, root):
    result = read(path)
    operator_dir = path.parent.parent
    candidate = operator_dir / "candidate.py"
    sha = digest(candidate.read_bytes())
    request_path = operator_dir / "request.json"
    request = read(request_path)
    assert digest(request["implementation"]["sources"][0]["content"].encode("utf-8")) == sha
    state_path = operator_dir.parents[1] / "state.json"
    state = read(state_path)
    op_state = state["operators"][session["operator"]]
    assert op_state["source_sha256"] == sha
    assert op_state["evaluation"]["sha256"] == digest(path.read_bytes())
    supplements = {}
    for name in ("supplementary", "cann_supplementary"):
        p = operator_dir / (name + ".json")
        value = read(p)
        supplements[name] = dict(passed=value.get("passed"), total=value.get("total"), source=ref(p, root))
    return dict(id=f"{session['id']}:evaluation-{sha[:12]}", session_id=session["id"],
                source_sha256=sha, source=ref(candidate, root), result=ref(path, root),
                request=ref(request_path, root), state=ref(state_path, root),
                started_at=op_state["started_at"], finished_at=op_state["finished_at"],
                binding=request["binding"], settings=request["settings"],
                status=result["status"], score=result["geo_mean"],
                passed=result["num_passed"], total=result["num_workloads"],
                supplements=supplements, occurrence=source_occurrence(session, sha))


def observation(config, evaluation, run, root):
    paired_path, sim_path = run / config["paired"], run / config["simulator"]
    paired, sim = read(paired_path)["result"], read(sim_path)
    assert paired["spec"]["source_sha256"] == sim["source_sha256"] == evaluation["source_sha256"]
    assert paired["spec"]["case"] == sim["case"]
    pipelines = {p["pipeline"]: {k: p[k] for k in ("productive_ratio", "wait_only_ratio", "idle_ratio")}
                 for p in sim["timeline"]["pipelines"] if p["pipeline"] in ("MTE2", "MTE3", "VECTOR")}
    return dict(paired=ref(paired_path, root), simulator=ref(sim_path, root),
                source_sha256=sim["source_sha256"], case=sim["case"],
                device_task_us=float(paired["candidate"]["basic"]["Task Duration(us)"]),
                simulator_scope=dict(core=sim["timeline"]["core"], kernel=sim["timeline"]["kernel"],
                                     capture_duration=sim["timeline"]["capture_duration"],
                                     time_unit=sim["timeline"]["time_unit"]),
                pipelines=pipelines, qualification=sim["collection_qualification"],
                hotspot=sim["regions"][0])


def event_stream(session, evaluations, preflights):
    """A source-anchored partial order; never invent missing wall-clock times."""
    events = [dict(e, lane="model", ordering="original_log_order") for e in session["events"]]
    for item in [*preflights, *evaluations]:
        is_formal = "score" in item
        occurrence = item["occurrence"]
        call_id, transition_id = occurrence["call_id"], occurrence["transition_id"]
        anchors = [e["sequence"] for e in events if
                   (is_formal and e["kind"] == "model_call_end" and e["call_id"] == call_id) or
                   (not is_formal and e.get("id") == transition_id)]
        events.append(dict(kind="formal_evaluation" if is_formal else "preflight_result",
                           lane="controller", ordering="source_anchored_not_exact_interleaving",
                           anchor_sequence=max(anchors) if anchors else 0,
                           call_id=call_id, source_sha256=item["source_sha256"],
                           record_id=item["id"], recorded_at=item.get("finished_at"),
                           status=item["status"], score=item.get("score"), source=item["result"]))
    events.sort(key=lambda e: (e.get("anchor_sequence", e.get("sequence", 0)), 1 if e["lane"] == "controller" else 0))
    for i, e in enumerate(events, 1):
        e["stream_position"] = i
    return events


def build_index(spec_path, output, root):
    spec = read(spec_path)
    index = dict(schema="hgkda.optimization-trace/v1", operator=spec["operator"],
                 presentation=ref(spec_path, root), history=spec["history"],
                 scope="saved model/tool events and controller artifacts; external unsaved actions are outside scope",
                 ordering="source-anchored partial order; invocation/tool sequence and controller timestamps are retained separately",
                 sessions=[], evaluations=[], preflights=[], measured_edges=[], evidence=[], views=spec["views"])
    for config in spec["sessions"]:
        run = root / config["run"]
        session = build_session(run, root, config["id"])
        assert not session["gaps"], session["gaps"]
        paths = [*run.glob(config["result_glob"]), *(root / p for p in config.get("additional_results", []))]
        evaluations = sorted((measurement(p, session, root) for p in paths), key=lambda e: e["started_at"])
        assert len(config["methods"]) == len(evaluations)
        contracts = {json.dumps((e["binding"], e["settings"]), sort_keys=True) for e in evaluations}
        assert len(contracts) == 1, "Incompatible requests inside one trace"
        transitions = {t["id"]: n for n, t in enumerate(session["transitions"])}
        preflights = []
        for p in sorted(run.glob("attempts/*/preflight.json")):
            value = read(p)
            sha = digest((p.parent / "candidate.py").read_bytes())
            assert sha == value["source_sha256"] == p.parent.name
            preflights.append(dict(id=f"{session['id']}:preflight-{sha[:12]}", session_id=session["id"],
                                   source_sha256=sha, source=ref(p.parent / "candidate.py", root),
                                   result=ref(p, root), status=value["status"],
                                   occurrence=source_occurrence_or_edit(session, sha)))
        for n, evaluation in enumerate(evaluations):
            evaluation.update(number=n + 1, method=config["methods"][n],
                              method_source=ref(root / config["method_source"], root))
            if config.get("observations"):
                evaluation["observation"] = observation(config["observations"][n], evaluation, run, root)
            if n:
                before = evaluations[n-1]
                a, b = before["occurrence"]["transition_id"], evaluation["occurrence"]["transition_id"]
                start = transitions[a] if a else -1
                stop = transitions[b]
                assert stop > start
                selected = session["transitions"][start + 1:stop + 1]
                assert selected[0]["input_sha256"] == before["source_sha256"]
                index["measured_edges"].append(dict(session_id=session["id"], source=before["id"], target=evaluation["id"],
                    relation="verified_saved_edit_sequence", transition_ids=[t["id"] for t in selected],
                    intermediate_edit_count=max(0, len(selected)-1)))
        imported = output.parent / (session["id"] + ".json")
        save(imported, session)
        stream = output.parent / (session["id"] + ".events.jsonl")
        stream.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in event_stream(session, evaluations, preflights)), encoding="utf-8")
        config_source = read(run / "config.json")
        catalog = {name: ref(run / "reference-catalog" / name, root)
                   for name in ("definition.json", "oracle.py", "correctness.jsonl", "timing.jsonl")}
        index["sessions"].append(dict(id=session["id"], title=config["title"], run=config["run"],
            imported=ref(imported, root), event_stream=ref(stream, root), config=ref(run / "config.json", root),
            reference_kind=config["reference_kind"], catalog=catalog,
            environment={k: config_source.get(k) for k in ("device", "container", "sdk", "device_python")},
            counts=dict(model_calls=len(session["calls"]), source_transitions=len(session["transitions"]),
                        preflight_sources=len(preflights), formal_evaluations=len(evaluations),
                        failed_preflights=sum(p["status"] != "passed" for p in preflights),
                        tool_calls=sum(c["tool_calls"] for c in session["calls"]),
                        unsuccessful_tools=sum(c["unsuccessful_tools"] for c in session["calls"])),
            evaluations=[e["id"] for e in evaluations], gaps=session["gaps"]))
        index["evaluations"].extend(evaluations)
        index["preflights"].extend(preflights)
    for evidence in spec["evidence"]:
        row = {k: v for k, v in evidence.items() if k != "path"}
        row["source"] = ref(root / evidence["path"], root)
        index["evidence"].append(row)
    save(output, index)
    return index


def source_occurrence_or_edit(session, sha):
    """Preflights can exercise an intermediate source within an invocation."""
    matches = [t for t in session["transitions"] if t["output_sha256"] == sha]
    if not matches:
        raise ValueError(f"Preflight has no saved edit occurrence: {sha}")
    t = matches[-1]
    return dict(call_id=t["call_id"], transition_id=t["id"])


def verify(index_path, root):
    index = read(index_path)
    files = {}

    def walk(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path, expected = value["path"], value["sha256"]
                if path not in files:
                    files[path] = digest((root / path).read_bytes())
                assert files[path] == expected, path
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(index)
    for session_ref in index["sessions"]:
        session = read(root / session_ref["imported"]["path"])
        walk(session)
        assert not session["gaps"]
        previous = None
        previous_sha = None
        for transition in session["transitions"]:
            assert transition["parent_transition"] == previous
            if previous is not None:
                assert transition["input_sha256"] == previous_sha
            previous = transition["id"]
            previous_sha = transition["output_sha256"]
        for call in session["calls"]:
            assert call["input_snapshot_matches_replay"] is not False
            assert call["output_snapshot_matches_replay"] is not False
        events = [json.loads(line) for line in (root / session_ref["event_stream"]["path"]).read_text(encoding="utf8").splitlines()]
        assert len(events) == len(session["events"]) + sum(e["session_id"] == session["id"] for e in index["evaluations"] + index["preflights"])
    for view in index["views"]:
        points = [e for e in index["evaluations"] if e["session_id"] == view["session"]]
        assert all(1 <= a["point"] <= len(points) for a in view["annotations"])
        assert all(e["score"] > 0 and e["passed"] == e["total"] for e in points)
        for annotation in view["annotations"]:
            assert set(annotation["evidence"]) <= {e["id"] for e in index["evidence"]}
            assert annotation["color"] in {"paired", "sim", "knowledge", "probe", "ir"}
        if view.get("present", True):
            nodes = view.get("nodes", [])
            assert len(nodes) == len(points)
            assert {n["point"] for n in nodes} == set(range(1, len(points)+1))
            assert all(n["location"] in {"device", "host", "mixed"} for n in nodes)
    return dict(status="passed", files_verified=len(files), evaluations=len(index["evaluations"]),
                preflights=len(index["preflights"]), measured_edges=len(index["measured_edges"]),
                sessions={s["id"]: s["counts"] for s in index["sessions"]})
