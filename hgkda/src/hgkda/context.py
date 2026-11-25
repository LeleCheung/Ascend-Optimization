"""Build model context from facts, heuristic hypotheses, and knowledge for one candidate."""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

from .artifacts import identity, read_json, unwrap
from .diagnostics import diagnose_case
from .hotspots import instruction_hotspots
from .regions import analyze as analyze_regions


def knowledge(operator, hypotheses=()):
    from .diagnostics import TYPES
    kind = TYPES.get(operator, "unknown")
    root = files("hgkda").joinpath("data", "knowledge")
    names = ["contract.md", "evidence.md", "operators.md"]
    if hypotheses:
        names.append("rules.md")
    library = json.loads(root.joinpath("mechanisms.json").read_text(encoding="utf8"))
    selected = []
    for record in library["records"]:
        exact = operator in record.get("operators", [])
        transferable = record.get("transferable_by_type") is True and kind in record.get("types", [])
        if exact or transferable:
            selected.append(dict(record, match="operator" if exact else "explicit_type_transfer",
                application='Observed method; new source, shapes, and compiler versions require independent validation'))
    return {"operator_type": kind, "mechanisms": selected, "source_checkpoint": library["source_checkpoint"], "documents": [
        {"id": name, "sha256": hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest(),
         "text": root.joinpath(name).read_text(encoding="utf8")} for name in names],
        "retrieval": {"exact": sum(r["match"] == "operator" for r in selected),
                      "type_transfer": sum(r["match"] == "explicit_type_transfer" for r in selected),
                      "empty": not selected,
                      "policy": 'Prefer exact operator records; supplement by type only for explicitly transferable records, retaining applicability constraints'},
        "update_policy": 'Treat new records as unverified hypotheses; bind source, contract, and complete results before review and conditional deduplication'}


def build_context(operator, source, comparison, *, listing=None, ir_manifest=None, roofline=None, regions=None):
    source = Path(source)
    source_id = identity(source, kind="candidate_source")
    compared = unwrap(read_json(comparison))
    spec = compared["spec"]
    if spec.get("operator") != operator or spec.get("source_sha256") != source_id["sha256"]:
        raise ValueError('Paired evidence has different operator or source identities')
    inputs = [source_id, identity(comparison, kind="paired_comparison")]
    records = compared["candidate"]["tables"]
    groups = {}
    for name, group in records.items():
        if name.startswith("PipeUtilization.csv/"):
            metrics = {key: value["mean"] for key, value in group["fields"].items()
                       if key.endswith("_ratio") and value.get("missing") == 0}
            groups[name] = diagnose_case({"operator": operator, "case_id": spec.get("case", {}).get("name"),
                                         "metrics": metrics}, levels=("metrics", "instruction"))
    hypotheses = [dict(item, group=group) for group, result in groups.items() for item in result["hypotheses"]]
    instruction, region_result, ir, roof = None, None, [], None
    if listing:
        content = read_json(listing)
        declared = content.get("source_sha256")
        if declared != source_id["sha256"]:
            raise ValueError('Instruction listing must declare a matching source_sha256; verify historical unbound listings before import')
        instruction = instruction_hotspots(content, source.read_text(encoding="utf8"))
        names = {k["kernel"] for k in instruction["kernels"]}
        filtered = dict(content, kernels=[k for k in content.get("kernels", []) if k["name"] in names])
        if names:
            region_result = analyze_regions(filtered)
        inputs.append(identity(listing, kind="instruction_listing"))
    if regions:
        region_result = unwrap(read_json(regions))
        if region_result.get('source_sha256') != source_id['sha256'] or region_result.get('case') != spec.get('case'):
            raise ValueError('Hotspot regions must bind to the same source and case')
        timeline = region_result.get('timeline', {})
        if timeline.get('status') == 'completed':
            names = {row['kernel'] for row in (instruction or {}).get('kernels', [])}
            if timeline.get('kernel') not in names or not timeline.get('core'):
                raise ValueError('Timeline must bind to a matched candidate kernel and specific core')
        inputs.append(identity(regions, kind='instruction_regions'))
    if ir_manifest:
        manifest_path = Path(ir_manifest)
        manifest = read_json(manifest_path)
        if manifest.get("source_sha256") != source_id["sha256"]:
            raise ValueError('IR manifest belongs to different candidate source')
        for item in manifest.get("files", []):
            path = manifest_path.parent / item["path"]
            binding = identity(path, kind="compiled_ir")
            if binding["sha256"] != item["sha256"]:
                raise ValueError('IR file hash mismatch: ' + item["path"])
            ir.append(dict(binding, text=path.read_text(encoding="utf8")))
        inputs.append(identity(ir_manifest, kind="ir_manifest"))
    if roofline:
        document = read_json(roofline)
        roof = unwrap(document)
        if roof.get("source_sha256") != source_id["sha256"] or roof.get("case") != spec.get("case"):
            raise ValueError('Roofline reference must bind to the same source and case')
        inputs.append(identity(roofline, kind="roofline_reference"))
    return {"operator": operator, "source": dict(source_id, text=source.read_text(encoding="utf8")),
            "case": spec.get("case"), "reference_kind": spec.get("reference_kind", "unspecified"),
            "facts": {"paired": compared, "compiled_ir": ir, "instruction_hotspots": instruction,
                      "regions": region_result, "roofline": roof},
            "hypotheses": hypotheses, "diagnostic_groups": groups,
            "knowledge": knowledge(operator, hypotheses), "inputs": inputs,
            "agent_task": {"output_fields": ["evidence", "mechanism_hypothesis", "proposed_change",
                                               "predicted_metric_change", "validation", "unresolved"],
                           "execution": 'The caller-hosted model interprets evidence and proposes changes; this command only builds context and does not call a model or run device tasks'}}
