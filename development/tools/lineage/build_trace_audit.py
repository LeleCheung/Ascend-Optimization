"""Build the bounded historical evidence audit from frozen discovery extracts."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "development/reports"
DATE = "20261116"


def read(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    discovery = read(f"trace-coverage-discovery-{DATE}.json")
    expanded = read(f"trace-expanded-inventory-{DATE}.json")
    bindings = read(f"trace-shared-profile-bindings-{DATE}.json")["shared_profiles"]
    index_path = ROOT / "operators/index.json"
    accepted = [r for r in json.loads(index_path.read_text(encoding="utf-8-sig"))["rows"]
                if r.get("scope") == "main_30" and r.get("accepted")]
    assert len(accepted) == 26
    preliminary = {r["operator"]: r for r in discovery["rows"]}
    simulator = expanded["simulator_files"]
    for item in simulator:
        item["operators"] = list(item["operators"])
        for binding in bindings:
            op = binding.get("implementation_definition")
            if binding["profile_id"] in item["path"] and op not in item["operators"]:
                item["operators"].append(op)
    special = {
        "log10_": ("diagnostic_negative_result", "区域与时间轴进入模型输入；两次修改均未超过固定起点。可讲诊断和失败预测，不能讲净收益。"),
        "uniform": ("posthoc_capture", "attempt-1第4轮正式成绩先于模拟器返回约9.5分钟；后续批次尚无已核验的模拟器消歧链。"),
        "selu_": ("no_gain_in_linked_transition", "有第1轮模拟器分析；紧接修改0.130370→0.129648，未提升。后续成功批次不能据此自动归因。"),
        "rad2deg_": ("capture_without_verified_decision_link", "共享服务manifest确认旧模拟器属于本算子；缺少它驱动后续成功修改的已核验事件链。"),
        "native_group_norm": ("no_gain_in_linked_transition", "模拟器partial及源码区域材料存在；search-1三轮0.591746→0.582306→0.546922，未形成成功修改。"),
        "negative_": ("metrics_not_simulator_attribution", "成对模拟器原件存在；已核验搜索的改善来自启动与分块。第3轮分析是msprof指标，不能由通用simulator警告认定使用模拟器。"),
    }
    rows = []
    for delivered in accepted:
        op = delivered["operator"]
        files = [f for f in simulator if op in f["operators"]]
        ledgers = [f for f in expanded["ledgers"] if f["operator"] == op]
        unique = {f["sha256"]: f for f in ledgers}
        status, gap = special.get(op, ("no_simulator_chain_located",
            "在声明的检索范围内未定位可归因的模拟器原件／决策链；这不证明历史上从未使用模拟器。"))
        rows.append({
            "operator": op, "delivery_source": delivered["source"],
            "delivery_source_sha256": digest(ROOT / delivered["source"]),
            "delivery_manifest": "operators/" + delivered["manifest"],
            "delivery_cann_score": delivered["cann_score"],
            "review_status": status, "gap": gap,
            "qualified_simulator_disambiguation": False,
            "simulator_files": files,
            "ledger_files": [{k: f[k] for k in ["path", "sha256"]} for f in ledgers],
            "unique_ledgers": len(unique),
            "historical_rounds": [{"ledger_sha256": f["sha256"], **r}
                                  for f in unique.values() for r in f["rounds"]],
            "preliminary_other_evidence": preliminary[op]["examples"],
        })
    # The curve uses raw result extracts, never current delivery scores.
    extract = read(f"trace-ledger-extract-{DATE}.json")
    uniform = next(f for f in extract if f["operator"] == "uniform"
                   and "uniform-endurance-20261022-attempt-1/" in f["ledger_path"])
    points = [{k: r[k] for k in ["round", "source_sha256", "result_path",
                                "result_sha256", "score", "passed", "total"]}
              for r in uniform["rounds"]]
    assert len(points) == 4 and all(p["passed"] == p["total"] == 18 for p in points)
    output = {
        "schema": "hgkda.historical-attribution-audit/v1", "date": "2026-11-16",
        "scope": "26 accepted operators; bounded historical audit, not an ablation study",
        "delivery_index_sha256": digest(index_path),
        "roots": expanded["roots"], "excluded_directory_tokens": expanded["excluded_directory_tokens"],
        "visited_files": expanded["files_visited"], "scan_errors": expanded["errors"],
        "limitations": ["Search is bounded to listed roots and naming conventions; snapshots and backups were excluded.",
                        "A ledger or retrospective explanation alone does not establish causal attribution.",
                        "Historical scores use their own contracts and must not be concatenated with current CANN delivery scores.",
                        "No source edits, new candidates, or device experiments were performed for this audit."],
        "confirmed_positive_cases": 0, "rows": rows,
        "trace_example": {"operator": "uniform", "contract": "historical attempt-1 KGS contract, 18/18",
                          "interpretation": "post-hoc simulator capture; not a simulator-caused improvement",
                          "points": points, "event_evidence": f"trace-uniform-event-order-{DATE}.json"},
    }
    (REPORTS / f"trace-attribution-matrix-{DATE}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = ["# 26项历史trace证据覆盖", "", "本表回答模拟器归因证据是否闭合，不改变算子交付状态。",
             "原始文件数含重复副本，不代表实验数；完整路径、SHA和逐轮记录见同名JSON。", "",
             "| 算子 | 去重账本 | 模拟器文件 | 当前判断／缺口 |", "| --- | ---: | ---: | --- |"]
    for r in rows:
        table.append(f"| `{r['operator']}` | {r['unique_ledgers']} | {len(r['simulator_files'])} | {r['gap']} |")
    table += ["", "检索范围：远端runs、整理前campaigns归档、当前development/experiments。",
              "排除缓存、环境、snapshot、backup、method-package等目录；零计数表示本次未定位，不表示不存在。",
              "当前没有确认满足完整标准的模拟器消歧成功案例。已有交付性能不能替代方法归因证据。", ""]
    (REPORTS / f"trace-attribution-matrix-{DATE}.md").write_text("\n".join(table), encoding="utf-8")
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">',
           '<rect width="1600" height="900" fill="white"/>',
           '<g font-family="Microsoft YaHei, Arial, sans-serif" fill="#14233b">',
           '<text x="80" y="80" font-size="40" font-weight="700">uniform：真实性能轨迹与证据时序</text>',
           '<text x="80" y="133" font-size="26">历史KGS合同 · 每轮18/18 · 不与当前CANN交付分数拼接</text>',
           '<path d="M140 210 V650 H1420" fill="none" stroke="#14233b" stroke-width="3"/>']
    xs = [260, 580, 900, 1220]
    ys = [650 - p["score"] / 1.2 * 420 for p in points]
    for tick in [0, .4, .8, 1.2]:
        y = 650 - tick / 1.2 * 420
        svg.append(f'<text x="55" y="{y+9}" font-size="26">{tick:.1f}×</text>')
    path = f'M{xs[0]} {ys[0]}' + ''.join(f'H{x} V{y}' for x,y in zip(xs[1:],ys[1:]))
    svg.append(f'<path d="{path}" stroke="#2563eb" stroke-width="5" fill="none"/>')
    labels = ["初始版本", "随机数计算改写", "宿主计数器改写", "稳定整数参数类型"]
    for p,x,y,label in zip(points,xs,ys,labels):
        svg += [f'<circle cx="{x}" cy="{y}" r="9" fill="#2563eb"/>',
                f'<text x="{x}" y="{y-25}" text-anchor="middle" font-size="30" font-weight="700">{p["score"]:.3f}×</text>',
                f'<text x="{x}" y="700" text-anchor="middle" font-size="26">R{p["round"]} · {html.escape(label)}</text>']
    svg += ['<rect x="80" y="755" width="1440" height="105" rx="15" fill="#fff3dc"/>',
            '<text x="110" y="795" font-size="28" font-weight="700">14:38:39 正式成绩返回 → 14:48:07 模拟器材料返回（UTC）</text>',
            '<text x="110" y="835" font-size="27">此轮模拟器是事后诊断，不能归因为此前的性能提升；历史随机状态语义另有约束。</text>',
            '</g></svg>']
    (REPORTS / f"trace-uniform-evidence-{DATE}.svg").write_text("\n".join(svg), encoding="utf-8")
    print(f"Wrote {len(rows)} rows; simulator evidence located for {sum(bool(r['simulator_files']) for r in rows)} operators; no confirmed positive attribution.")


if __name__ == "__main__":
    main()
