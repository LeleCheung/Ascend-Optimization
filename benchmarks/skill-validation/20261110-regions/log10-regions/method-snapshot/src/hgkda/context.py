"""将同一候选的事实、启发式假设和知识组织为模型上下文。"""
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
                application="已观察到的方法；新源码、形状和编译版本仍需独立验证"))
    return {"operator_type": kind, "mechanisms": selected, "source_checkpoint": library["source_checkpoint"], "documents": [
        {"id": name, "sha256": hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest(),
         "text": root.joinpath(name).read_text(encoding="utf8")} for name in names],
        "retrieval": {"exact": sum(r["match"] == "operator" for r in selected),
                      "type_transfer": sum(r["match"] == "explicit_type_transfer" for r in selected),
                      "empty": not selected,
                      "policy": "精确算子记录优先；仅显式标记可迁移的记录按类型补充，保留原适用约束"},
        "update_policy": "新记录先作待验证假设；绑定源码、合同和完整结果后，经审查按条件去重更新"}


def build_context(operator, source, comparison, *, listing=None, ir_manifest=None, roofline=None, regions=None):
    source = Path(source)
    source_id = identity(source, kind="candidate_source")
    compared = unwrap(read_json(comparison))
    spec = compared["spec"]
    if spec.get("operator") != operator or spec.get("source_sha256") != source_id["sha256"]:
        raise ValueError("成对证据的算子或源码身份不同")
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
            raise ValueError("指令清单必须声明匹配的 source_sha256；历史无绑定清单需先核验后导入")
        instruction = instruction_hotspots(content, source.read_text(encoding="utf8"))
        names = {k["kernel"] for k in instruction["kernels"]}
        filtered = dict(content, kernels=[k for k in content.get("kernels", []) if k["name"] in names])
        if names:
            region_result = analyze_regions(filtered)
        inputs.append(identity(listing, kind="instruction_listing"))
    if regions:
        region_result = unwrap(read_json(regions))
        if region_result.get('source_sha256') != source_id['sha256'] or region_result.get('case') != spec.get('case'):
            raise ValueError('热点区域必须绑定相同源码与用例')
        timeline = region_result.get('timeline', {})
        if timeline.get('status') == 'completed':
            names = {row['kernel'] for row in (instruction or {}).get('kernels', [])}
            if timeline.get('kernel') not in names or not timeline.get('core'):
                raise ValueError('时间轴必须绑定已匹配的候选kernel和具体核')
        inputs.append(identity(regions, kind='instruction_regions'))
    if ir_manifest:
        manifest_path = Path(ir_manifest)
        manifest = read_json(manifest_path)
        if manifest.get("source_sha256") != source_id["sha256"]:
            raise ValueError("IR 清单与候选源码不同")
        for item in manifest.get("files", []):
            path = manifest_path.parent / item["path"]
            binding = identity(path, kind="compiled_ir")
            if binding["sha256"] != item["sha256"]:
                raise ValueError("IR 文件哈希不符：" + item["path"])
            ir.append(dict(binding, text=path.read_text(encoding="utf8")))
        inputs.append(identity(ir_manifest, kind="ir_manifest"))
    if roofline:
        document = read_json(roofline)
        roof = unwrap(document)
        if roof.get("source_sha256") != source_id["sha256"] or roof.get("case") != spec.get("case"):
            raise ValueError("Roofline 参考必须绑定相同源码与用例")
        inputs.append(identity(roofline, kind="roofline_reference"))
    return {"operator": operator, "source": dict(source_id, text=source.read_text(encoding="utf8")),
            "case": spec.get("case"), "reference_kind": spec.get("reference_kind", "unspecified"),
            "facts": {"paired": compared, "compiled_ir": ir, "instruction_hotspots": instruction,
                      "regions": region_result, "roofline": roof},
            "hypotheses": hypotheses, "diagnostic_groups": groups,
            "knowledge": knowledge(operator, hypotheses), "inputs": inputs,
            "agent_task": {"output_fields": ["evidence", "mechanism_hypothesis", "proposed_change",
                                               "predicted_metric_change", "validation", "unresolved"],
                           "execution": "由调用宿主的模型解释证据并提出修改；本命令只生成上下文，不调用模型或执行设备任务"}}
