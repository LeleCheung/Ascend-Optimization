"""HG-KDA 统一命令行；标准输出始终为 JSON。"""
from __future__ import annotations

import argparse
import ast
import json
import platform
import sys
from pathlib import Path

from . import __version__
from .artifacts import emit, envelope, identity, read_json, unwrap


def parser():
    root = argparse.ArgumentParser(description="HG-KDA：性能证据、机制诊断与评测适配")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ["doctor", "compare", "diagnose", "hotspots", "regions", "roofline", "context", "knowledge"]:
        p = commands.add_parser(name)
        p.add_argument("--output", type=Path, help="新 JSON 文件；已有文件不会覆盖")
        if name == "compare":
            p.add_argument("--directory", type=Path, required=True)
        if name in {"diagnose", "roofline"}:
            p.add_argument("--input", type=Path, required=True)
        if name in {"hotspots", "regions"}:
            p.add_argument("--listing", type=Path, required=True)
            p.add_argument("--source", type=Path, required=name == "hotspots")
            p.add_argument("--top", type=int, default=12)
        if name == "regions":
            p.add_argument("--kernel")
            p.add_argument("--core")
            p.add_argument("--trace", type=Path)
            p.add_argument("--trace-unit", choices=["us", "ns", "cycles"], default="us")
            p.add_argument("--source-map", type=Path)
        if name in {"context", "knowledge"}:
            p.add_argument("--operator", required=True)
        if name == "context":
            p.add_argument("--source", type=Path, required=True)
            p.add_argument("--comparison", type=Path, required=True)
            p.add_argument("--listing", type=Path)
            p.add_argument("--ir-manifest", type=Path)
            p.add_argument("--roofline", type=Path)
            p.add_argument("--regions", type=Path)
    p = commands.add_parser("gate")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--review", type=Path)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("feedback")
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--timing-cases", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--operator", required=True)
    p.add_argument("--threshold", type=float, default=.8)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("kgs")
    p.add_argument("action", choices=["status", "upload", "evaluate", "operation"])
    p.add_argument("--server", required=True)
    p.add_argument("--request", type=Path)
    p.add_argument("--review", type=Path)
    p.add_argument("--archive", type=Path)
    p.add_argument("--run-directory", type=Path)
    p.add_argument("--operation-id")
    p.add_argument("--output", type=Path)
    p = commands.add_parser("collect")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-directory", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("curate")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("retrieve")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--operator-type", required=True)
    p.add_argument("--scope", type=Path, required=True)
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--output", type=Path)
    return root


def dispatch(args):
    cmd = args.command
    if cmd == "doctor":
        return {"version": __version__, "python": platform.python_version(),
                "offline_dependencies": "Python standard library", "kg_required": False,
                "kgs_required": False, "device_access": "not_requested"}, []
    if cmd == "compare":
        from .compare import compare
        return compare(args.directory), [identity(args.directory / "spec.json", kind="paired_spec")]
    if cmd == "diagnose":
        from .diagnostics import diagnose_case
        data = unwrap(read_json(args.input))
        return diagnose_case(data, levels=("metrics", "instruction")), [identity(args.input)]
    if cmd == "hotspots":
        from .hotspots import instruction_hotspots
        if args.top < 1:
            raise ValueError("top 必须为正整数")
        listing = read_json(args.listing)
        source = identity(args.source, kind="candidate_source")
        if listing.get("source_sha256") != source["sha256"]:
            raise ValueError("指令清单与源码身份未绑定或不同")
        result = instruction_hotspots(listing, args.source.read_text(encoding="utf8"), count=args.top)
        return result, [identity(args.listing), source]
    if cmd == "regions":
        from .regions import analyze
        sources, inputs = {}, [identity(args.listing)]
        if args.source_map:
            inputs.append(identity(args.source_map))
            for file, entry in read_json(args.source_map).items():
                path = args.source_map.parent / entry["path"]
                entry_id = identity(path)
                if entry_id["sha256"] != entry["sha256"]:
                    raise ValueError("源码映射哈希不符：" + file)
                sources[file] = dict(text=path.read_bytes().decode("utf8"), sha256=entry["sha256"])
                inputs.append(entry_id)
        if args.trace:
            inputs.append(identity(args.trace))
            if not read_json(args.trace).get('hgkda_binding'):
                raise ValueError('导入时间轴须声明hgkda_binding，明确源码、case、kernel、core和时间单位')
        return analyze(read_json(args.listing), read_json(args.trace) if args.trace else None,
                       kernel=args.kernel, core=args.core, top=args.top,
                       sources=sources, trace_unit=args.trace_unit), inputs
    if cmd == "roofline":
        from .roofline import derive
        data = read_json(args.input)
        result = derive(data["facts"], data["shape"], data["dtype"], data["empirical"])
        result.update(source_sha256=data.get("source_sha256"), case=data.get("case"))
        return result, [identity(args.input)]
    if cmd == "context":
        from .context import build_context
        result = build_context(args.operator, args.source, args.comparison, listing=args.listing,
                               ir_manifest=args.ir_manifest, roofline=args.roofline, regions=args.regions)
        return result, result.pop("inputs")
    if cmd == "knowledge":
        from .context import knowledge
        return knowledge(args.operator, hypotheses=True), []
    if cmd == "gate":
        from .gate import admit, inspect_text
        text = args.source.read_bytes().decode("utf8")
        result = admit(text, read_json(args.review)) if args.review else inspect_text(text)
        return result, [identity(args.source)] + ([identity(args.review)] if args.review else [])
    if cmd == "feedback":
        from .feedback import performance_feedback
        source = identity(args.source)
        audit = read_json(args.audit)
        rows = [r for r in audit.get('rows', []) if r.get('operator') == args.operator]
        if not audit.get('evidence_verified') or len(rows) != 1 or rows[0]['source_sha256'] != source['sha256']:
            raise ValueError('正式审计与候选源码身份不同或未通过')
        result = read_json(args.result)
        if result.get('geo_mean') != rows[0].get('geo_mean'):
            raise ValueError('正式结果与审计性能不同')
        cases = [json.loads(line) for line in args.timing_cases.read_text(encoding='utf8').splitlines() if line.strip()]
        checks = rows[0].get('checks', {})
        checked = len(checks) >= 2 and all(r.get('status') == 'passed' for r in checks.values())
        feedback = performance_feedback(result, cases, threshold=args.threshold, checks_passed=checked)
        feedback.update(source_sha256=source['sha256'], operator=args.operator)
        return feedback, [source,identity(args.result),identity(args.audit),identity(args.timing_cases)]
    if cmd == "kgs":
        from .adapters.kgs import call, evaluate, upload_bundle
        if args.action == "status":
            return call(args.server, "/status"), []
        if args.action == "operation":
            if not args.operation_id or not args.operation_id.isalnum():
                raise ValueError("需要有效的 operation-id")
            return call(args.server, "/operations/" + args.operation_id), []
        if args.action == "upload":
            if not args.archive:
                raise ValueError("upload 需要 --archive")
            return upload_bundle(args.server, args.archive), [identity(args.archive)]
        if not all([args.request, args.review, args.run_directory]):
            raise ValueError("evaluate 需要 --request、--review 和新的 --run-directory")
        return evaluate(args.server, args.request, args.review, args.run_directory), [identity(args.request), identity(args.review)]
    if cmd == "collect":
        from .adapters.msprof import collect
        return collect(args.config, args.run_directory), [identity(args.config)]
    if cmd == "curate":
        from .curation import organize
        return organize(read_json(args.records), root=args.root), [identity(args.records)]
    if cmd == "retrieve":
        from .curation import retrieve
        if args.max_chars <= 0:
            raise ValueError("max-chars 必须为正整数")
        return retrieve(unwrap(read_json(args.index)), args.operator_type, read_json(args.scope),
                        max_chars=args.max_chars), [identity(args.index), identity(args.scope)]
    raise ValueError("不支持的命令")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.output and args.output.exists():
            raise FileExistsError("输出已存在：" + str(args.output))
        result, inputs = dispatch(args)
        text = emit(envelope(args.command, result, inputs), args.output)
        if args.output:
            print(json.dumps({"status": "completed", "output": str(args.output)}, ensure_ascii=False))
        else:
            print(text, end="")
        return 0
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        print(json.dumps(envelope(args.command, {"error_type": type(error).__name__, "message": str(error)},
                                  status="failed"), ensure_ascii=False), file=sys.stderr)
        return 2
