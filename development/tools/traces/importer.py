"""Index frozen model/tool events and recover source transitions in memory.

No recorded shell, Python or candidate code is executed. Original files stay
unchanged; recovered code is used only to compute content hashes.
"""
import hashlib
import json
import re
from pathlib import Path

from development.tools.skill_validation.audit_generation import legacy_shell_replacement


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ref(path, root, **extra):
    path = path.resolve()
    return dict(path=path.relative_to(root).as_posix(), sha256=digest(path.read_bytes()), **extra)


def calls_in_order(run):
    def key(path):
        match = re.search(r"(\d+)$", path.name)
        return int(match[1]) if match else 0
    return sorted((run / "agent-calls").iterdir(), key=key)


def tool_records(path):
    """Collapse repeated streamed blocks by tool ID, retaining original lines."""
    tools, results = {}, {}
    for line_number, line in enumerate(path.open(encoding="utf-8-sig"), 1):
        row = json.loads(line)
        message = row.get("message", {})
        if not isinstance(message, dict):
            continue
        for block in message.get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tools[block["id"]] = dict(block, source_line=line_number)
            elif block.get("type") == "tool_result":
                results[block["tool_use_id"]] = dict(block, source_line=line_number)
    return sorted(tools.values(), key=lambda t: t["source_line"]), results


def is_candidate(path):
    return path.replace("\\", "/").split("/")[-1] == "candidate.py"


def source_mutation(tool, text):
    """Replay only literal writes/replacements, never general code."""
    payload = tool.get("input", {})
    name = tool["name"]
    if name in ("Edit", "Write") and is_candidate(payload.get("file_path", "")):
        if name == "Write":
            return payload["content"], "literal_write"
        before, after = payload["old_string"], payload["new_string"]
        if text is None or not before or before not in text:
            return None, "unresolved_edit"
        if not payload.get("replace_all", False) and text.count(before) != 1:
            return None, "ambiguous_edit"
        return text.replace(before, after, -1 if payload.get("replace_all") else 1), "literal_edit"
    if name != "Bash":
        return text, None
    command = payload.get("command", "")
    # A literal heredoc can be followed by read-only commands. We only extract
    # the literal bytes up to its delimiter and never run the command string.
    match = re.match(r"^cat\s+(>>?)\s+candidate\.py\s+<<\s*(['\"]?)([A-Za-z0-9_]+)\2\n", command)
    if match:
        delimiter = match[3]
        end = re.search(r"^" + re.escape(delimiter) + r"(?:\n|$)", command[match.end():], re.M)
        if not end:
            return None, "unresolved_heredoc"
        value = command[match.end():match.end()+end.start()]
        return ((text or "") + value if match[1] == ">>" else value), "literal_heredoc"
    if "candidate.py" in command and "s.replace" in command:
        value = legacy_shell_replacement(command, text)
        if value is not None:
            return value, "literal_legacy_replace"
        return None, "unresolved_shell_edit"
    # Detect potentially modifying commands that this bounded parser does not
    # understand, without treating reads/backups as writes to the candidate.
    if re.search(r"(?:>\s*candidate\.py|open\(['\"]candidate\.py['\"],\s*['\"]w|sed\s+-i.*candidate\.py)", command):
        return None, "unresolved_shell_edit"
    return text, None


def classify_tool(tool):
    payload = tool.get("input", {})
    material = " ".join(str(payload.get(k, "")) for k in ("file_path", "path", "command", "description"))
    lower = material.lower()
    if any(k in lower for k in ("simulator", "regions.json", "timeline.json")):
        return "simulator"
    if any(k in lower for k in ("paired", "comparison.json", "opprof")):
        return "paired_profiling"
    if any(k in lower for k in (".ttir", ".bcmlir", "actual-ir", "runtime-api", "compiler-comparison")):
        return "source_ir"
    if any(k in lower for k in ("probe", "host-profile", "diagnosis.json")):
        return "diagnostic"
    if any(k in lower for k in ("performance-feedback", "result.json", "summarize-final")):
        return "evaluation_feedback"
    if any(k in lower for k in ("knowledge", "skill.md", "context-brief")):
        return "method_context"
    if any(k in lower for k in ("preflight", "check --", "evaluate")):
        return "validation"
    if tool["name"] in ("Edit", "Write"):
        return "edit"
    return "other"


def build_session(run, root, session_id):
    config = read(run / "config.json") if (run / "config.json").exists() else {}
    starter = run / "workspace/starter.py"
    starter_sha = digest(starter.read_bytes()) if starter.exists() else None
    # The upstream source is a provided starting artifact, not assumed to be
    # the pre-existing contents of candidate.py before the first recorded write.
    text = None
    transitions, calls, events, snapshots, gaps = [], [], [], {}, []
    previous_transition = None
    for call in calls_in_order(run):
        provenance_path = call / "source-provenance.json"
        provenance = read(provenance_path) if provenance_path.exists() else None
        log_path = call / "agent-events.jsonl"
        log_ref = ref(log_path, root)
        input_recovered = digest(text.encode()) if text is not None else None
        input_verified = None
        if provenance:
            inp = (call / "input-source.py").read_bytes()
            assert digest(inp) == provenance["input_sha256"]
            input_verified = (input_recovered == provenance["input_sha256"]
                              if input_recovered is not None else None)
            if text is not None and not input_verified:
                gaps.append(dict(kind="between_call_source_gap", call=call.name,
                                 reconstructed=input_recovered, recorded=provenance["input_sha256"]))
            text = inp.decode("utf-8") if provenance["candidate_existed"] else None
            if provenance["candidate_existed"]:
                snapshots.setdefault(provenance["input_sha256"], dict(
                    sha256=provenance["input_sha256"],
                    source=ref(call / "input-source.py", root), availability="frozen_file"))
        call_start_sha = digest(text.encode()) if text is not None else None
        tools, results = tool_records(log_path)
        invocation = read(call / "host-invocation.json")
        call_id = f"{session_id}:{call.name}"
        events.append(dict(kind="model_call_start", call_id=call_id, source_sha256=call_start_sha,
                           recorded_at=invocation.get("started_at"), source=ref(call / "host-invocation.json", root)))
        count, failures = 0, 0
        for tool in tools:
            result = results.get(tool["id"])
            success = result is not None and not result.get("is_error", False)
            payload = tool.get("input", {})
            event = dict(kind="tool_call", call_id=call_id, tool_id=tool["id"],
                         tool=tool["name"], category=classify_tool(tool), success=success,
                         source=dict(log_ref, line=tool["source_line"],
                                     result_line=result.get("source_line") if result else None),
                         description=payload.get("description", "")[:220],
                         path=payload.get("file_path", payload.get("path", "")))
            events.append(event)
            if not success:
                failures += 1
                continue
            before = digest(text.encode()) if text is not None else None
            new_text, kind = source_mutation(tool, text)
            if kind is None:
                continue
            if kind.startswith("unresolved") or kind == "ambiguous_edit":
                gaps.append(dict(kind=kind, call=call.name, source=event["source"]))
                text = None
                continue
            text = new_text
            after = digest(text.encode()) if text is not None else None
            transition_id = f"{session_id}:change-{len(transitions)+1:04}"
            transition = dict(id=transition_id, parent_transition=previous_transition,
                              call_id=call_id, kind=kind, input_sha256=before,
                              output_sha256=after, source=event["source"],
                              relation_status="verified_literal_replay" if before else "write_with_unknown_file_input")
            transitions.append(transition)
            events.append(dict(transition, kind="candidate_change", edit_kind=kind))
            if after:
                snapshots.setdefault(after, dict(sha256=after, source=event["source"],
                                                 availability="reconstructable_from_literal_tool_record"))
            previous_transition = transition_id
            count += 1
        final_sha = digest(text.encode()) if text is not None else None
        output_verified = None
        if provenance and "output_sha256" in provenance:
            output_verified = final_sha == provenance["output_sha256"]
            if output_verified is not True:
                gaps.append(dict(kind="call_output_mismatch", call=call.name,
                                 recovered=final_sha, recorded=provenance["output_sha256"]))
            output = (call / "output-source.py").read_bytes()
            assert digest(output) == provenance["output_sha256"]
            text = output.decode("utf-8")
            final_sha = provenance["output_sha256"]
            snapshots[final_sha] = dict(sha256=final_sha, source=ref(call / "output-source.py", root), availability="frozen_file")
        calls.append(dict(id=call_id, invocation=call.name, model=invocation.get("model"),
                          started_at=invocation.get("started_at"),
                          input_sha256=call_start_sha, output_sha256=final_sha,
                          input_snapshot_matches_replay=input_verified,
                          output_snapshot_matches_replay=output_verified,
                          edits=count, tool_calls=len(tools), unsuccessful_tools=failures,
                          prompt=ref(call / "prompt.md", root), source=log_ref,
                          method_manifest=ref(call / "method-manifest.json", root) if (call / "method-manifest.json").exists() else None))
        events.append(dict(kind="model_call_end", call_id=call_id, source_sha256=final_sha,
                           edits=count, unsuccessful_tools=failures))
    if starter.exists():
        snapshots[starter_sha] = dict(sha256=starter_sha, source=ref(starter, root), availability="frozen_file")
    for i, event in enumerate(events, 1):
        event["sequence"] = i
    return dict(id=session_id, operator=config.get("operator", "log10_"),
                run=run.relative_to(root).as_posix(), starter_sha256=starter_sha,
                order_basis="model invocation number then original tool line; missing wall-clock timestamps remain null",
                calls=calls, transitions=transitions, events=events,
                snapshots=list(snapshots.values()), gaps=gaps)
