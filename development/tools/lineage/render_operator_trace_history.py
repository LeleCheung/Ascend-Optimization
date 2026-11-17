"""Render discovered batches in the established search-trajectory visual style."""
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "development/reports"
OUT = REPORTS / "operator-traces-history-20261116"
OPS = ["rad2deg_", "uniform", "negative_", "selu_", "log10_", "native_group_norm"]
COLORS = {"P": "#2368b1", "S": "#c55a18", "I": "#7852a6", "E": "#168358", "?": "#455269"}
BANDS = {"P": "#e3eff8", "S": "#fbe8d6", "I": "#eee6f5", "E": "#e3f0df", "?": "#e7edf5"}
NAMES = {"P": "成对分析", "S": "模拟器", "I": "源码／IR", "E": "专项探测", "?": "待核验"}


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def text(x, y, s, size=26, weight=600, anchor="start", fill="#1b2434"):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{html.escape(str(s))}</text>'


def existing_series():
    spec = importlib.util.spec_from_file_location("episodes", Path(__file__).with_name("render_operator_traces.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.data()


def short_batch(path):
    for token, name, date in [
        ("strict-night-20261102", "严格约束搜索", "2026-11-02"),
        ("night-20261021-v3", "首批搜索", "2026-10-21"),
        ("paired-pointwise-20261022", "成对分析批次", "2026-10-22"),
        ("buffer-pointwise-20261022", "缓冲与分块批次", "2026-10-22"),
        ("endurance-20261022", "持续优化", "2026-10-22"),
        ("next25-20261029-v2", "新目标搜索 v2", "2026-10-29"),
        ("next25-20261029", "新目标搜索", "2026-10-29"),
    ]:
        if token in path:
            if "uniform" in path and "endurance" in path:
                name += " · 尝试2" if "attempt-2" in path else " · 尝试1"
            return name, date
    return "历史搜索", "日期待核验"


def label_from_plan(r):
    p = r.get("plan") or {}
    s = (p.get("strategy", "") + " " + p.get("code_changes", "")).lower()
    if p.get("kind") == "baseline":
        return "建立本批起点"
    if "stream" in s and ("persistent" in s or "single-wave" in s):
        return "固定program流式循环"
    if "two-pointer" in s:
        return "双指针组织预取"
    if "bf16" in s and "float16" in s:
        return "BF16中间精度与分块"
    if "counter" in s and "int32" in s:
        return "稳定计数器整数类型"
    if "philox" in s:
        return "改写随机数计算"
    if "quantize" in s or "dtype-bound" in s:
        return "修复随机数边界量化"
    if "multi-tile" in s or "unrolled" in s and "tiles" in s:
        return "每program处理多块"
    if "closure" in s or "wrapper" in s or "bookkeeping" in s:
        return "精简宿主调用路径"
    if "launch" in s:
        return "调整启动与编译缓存"
    if "tile" in s or "block" in s:
        return "调整分块与数据处理"
    return "修改方法待核验"


def load_history():
    known = existing_series()
    known_by_ledger = {s["ledger"]["sha256"]: s for s in known if "ledger" in s}
    inventory = read(REPORTS / "trace-expanded-inventory-20261116.json")
    result = {op: [] for op in OPS}
    seen = set()
    for ledger in inventory["ledgers"]:
        op = ledger["operator"]
        if op not in result or ledger["sha256"] in seen:
            continue
        seen.add(ledger["sha256"])
        title, date = short_batch(ledger["path"])
        batch = dict(operator=op, title=title, date=date, reference="本批KGS参考",
                     source=dict(path=ledger["path"], sha256=ledger["sha256"]), points=[])
        for r in ledger["rounds"]:
            e = r["evaluation"]
            point = dict(round=r["round"], score=e.get("geo_mean"), status=e.get("status"),
                         passed=e.get("num_passed"), total=e.get("num_workloads"),
                         source_sha256=r["source_sha256"], action=label_from_plan(r),
                         evidence="?", evidence_detail="具体信息来源待核验", plan=r.get("plan"),
                         profile=r.get("profile"), conclusion=r.get("conclusion"))
            if ledger["sha256"] in known_by_ledger:
                k = next(p for p in known_by_ledger[ledger["sha256"]]["points"] if p["round"] == r["round"])
                point["action"] = k["action"]
                point["evidence"] = {"probe": "E", "source": "I", "profile": "P", "sim": "S"}.get(k["kind"], "?")
                point["evidence_detail"] = k["information_source"]
                point["result"] = k.get("result")
            batch["points"].append(point)
        result[op].append(batch)

    # Local controllers use model calls, preflights and formal evaluations as
    # separate counters. Only actual formal summaries become performance dots.
    local = ROOT / "development/experiments/skill-validation"
    groups = [
        ("log10_", "独立工作流开发", "2026-11-10", "20261110-log10-development", [
            ("a4d38b964aa6", "合规Triton布局与数学", "I"),
            ("a73e15786d68", "有限网格循环", "?"),
            ("4eca306ca042", "缓存尚未实际命中", "E"),
            ("2f90b7925721", "修复冷缓存登记", "E"),
            ("0fefe838c58c", "缓存宿主发射计划", "E"),
            ("4c184cdc36f4", "减少重复模式查询", "E"),
            ("e62c6d636097", "双块预读尝试", "?"),
            ("b2a52734c432", "热路径使用整数地址", "I"),
            ("d519b459c915", "分离满块与尾块", "I"),
            ("b64e7d8351df", "无掩码成对循环", "I"),
            ("69699663ecc7", "相邻载荷与编译期循环", "I"),
            ("8600ce070318", "输入dtype数学", "I"),
        ]),
        ("log10_", "冻结方法重生成", "2026-11-10", "20261110-log10-frozen", [
            ("40cfe1a83343", "冻结方法生成首版", "?"),
            ("0261f02aee72", "按完整反馈继续优化", "?"),
            ("b6b10ceb10e3", "继续改进启动与布局", "?"),
            ("45c7380a8062", "分块与启动缓存命中", "E"),
        ]),
        ("selu_", "独立工作流恢复", "2026-11-09", "20261109-selu-recovery", [
            ("3cec1212c3b7", "独立生成起点", "P"),
            ("c6255855aa16", "缓存编译启动器", "E"),
            ("80bfefff4ff8", "raw stream与简化缓存键", "E"),
            ("0fed8d43398d", "NoWait接口与扩大分块", "I"),
            ("5e9e4d482620", "命中缓存后直接发射", "E"),
        ]),
    ]
    for op, title, date, folder, entries in groups:
        report = REPORTS / ("selu-skill-recovery-20261109.md" if op == "selu_" else "log10-method-transfer-20261110.md")
        batch = dict(operator=op, title=title, date=date, reference="CANN", points=[],
                     source=dict(path=str(report.relative_to(ROOT)), sha256=sha(report)))
        for i, (prefix, action, evidence) in enumerate(entries, 1):
            directory = local / folder
            if prefix == "3cec1212c3b7":
                directory = local / "20261109-selu"
            summary = directory / ("formal-" + prefix) / "summarize-final.json"
            row = read(summary)["rows"][0]
            batch["points"].append(dict(round=i, score=row["cann_score"],
                                        status=row["evaluation"]["status"],
                                        passed=row["evaluation"]["num_passed"], total=row["evaluation"]["num_workloads"],
                                        source_sha256=row["source_sha256"], action=action, evidence=evidence,
                                        evidence_detail=NAMES[evidence], result=row["evaluation"],
                                        summary_path=str(summary.relative_to(ROOT)), summary_sha256=sha(summary)))
        if folder == "20261110-log10-development":
            batch["activity"] = dict(model_calls=24, preflight_sources=17, formal_sources=12)
        if folder == "20261110-log10-frozen":
            batch["activity"] = dict(model_calls=7, preflight_sources=4, formal_sources=4)
        result[op].append(batch)
    old_regions = next(s for s in known if s["op"] == "log10_")
    batch = dict(operator="log10_", title="模拟器区域与等待修订", date="2026-11-10", reference="CANN",
                 source=old_regions["report"], points=[])
    for p in old_regions["points"]:
        batch["points"].append(dict(**p, evidence="S" if p["kind"] == "sim" else "P", status="PASSED"))
    result["log10_"].append(batch)
    for op, batches in result.items():
        batches.sort(key=lambda b: b["date"])
        for i, b in enumerate(batches, 1):
            b["id"] = f"{op}-batch-{i:02}"
            b["performance_nodes"] = sum(p["score"] is not None for p in b["points"])
    return result


def pill(x, y, key, size=22):
    color = COLORS[key]
    label = f"{key} {NAMES[key]}"
    width = len(label)*size*.75+28
    return (f'<rect x="{x}" y="{y-26}" width="{width}" height="38" rx="19" fill="#fffdf8" stroke="{color}" stroke-width="2.5"/>'
            +text(x+width/2, y+1, label, size, 700, "middle", color))


def plot(b, x, y, width, height, compact=False):
    pts = b["points"]
    n = len(pts)
    values = [p["score"] for p in pts if p["score"] is not None]
    ceiling = math.ceil(max(values, default=1)*1.15*10)/10
    # Reserve a separate strip at the top for method annotations.
    xs = [x+18+i*(width-36)/max(1,n-1) for i in range(n)]
    sy = lambda value: y+height-value/ceiling*(height-90)
    out=[]
    for i,p in enumerate(pts):
        left=x if i==0 else (xs[i-1]+xs[i])/2
        right=x+width if i==n-1 else (xs[i]+xs[i+1])/2
        out.append(f'<rect x="{left}" y="{y}" width="{right-left}" height="{height}" fill="{BANDS[p["evidence"]]}"/>')
    for v in [0,ceiling/2,ceiling]:
        yy=sy(v)
        out += [f'<path d="M{x} {yy} H{x+width}" stroke="#c7cbd1" stroke-width="1.3"/>', text(x-12,yy+8,f'{v:.2f}×',19 if compact else 24,700,"end")]
    out.append(f'<path d="M{x} {y} V{y+height} H{x+width}" fill="none" stroke="#455269" stroke-width="2"/>')
    previous=None
    for i,p in enumerate(pts):
        if p["score"] is None:
            yy=y+height-35
            out += [f'<path d="M{xs[i]-7} {yy-7} l14 14 m0 -14 l-14 14" stroke="#b14e24" stroke-width="3"/>',text(xs[i],yy-18,'未通过',18 if compact else 23,700,"middle","#b14e24")]
            previous=None
        else:
            yy=sy(p["score"])
            if previous:
                px,py=previous
                out.append(f'<path class="trajectory" d="M{px} {py} L{xs[i]} {yy}" fill="none" stroke="#168358" stroke-width="{5 if compact else 7}" stroke-linecap="round" stroke-linejoin="round"/>')
            out.append(f'<circle cx="{xs[i]}" cy="{yy}" r="{6 if compact else 8}" fill="#168358" stroke="#fffdf8" stroke-width="3"/>')
            if not compact or n<=5 or i in [0,n-1]:
                label_offset = 32 if i%2 and n>6 and not compact else -15
                label_anchor = "end" if i==n-1 else "middle"
                out.append(text(xs[i],yy+label_offset,f'{p["score"]:.3f}',18 if compact else 25,800,label_anchor))
            previous=(xs[i],yy)
        if not compact or n<=5 or i in [0,n-1]:
            out.append(text(xs[i],y+height+34,f'R{p["round"]}',18 if compact else 23,700,"middle"))
        out.append(f'<title>R{p["round"]}: {html.escape(p["action"])}</title>')
    if compact:
        milestone=pts[-1]
        out.append(pill(x+width/2-72,y+31,milestone["evidence"],20))
        out.append(text(x+width/2,y+75,milestone["action"],20,800,"middle"))
    else:
        # Keep annotations near the curve and only label a few spaced milestones.
        chosen=[n-1] if n<3 else ([1,n-1] if n<=4 else [int((n-1)*.27),int((n-1)*.63),n-1])
        for j,i in enumerate(chosen):
            p=pts[i];xx=xs[i];count=len(chosen)
            center=x+(j+.5)*width/count
            out.append(pill(center-78,y+32,p["evidence"]))
            out.append(text(center,y+76,p["action"],23,800,"middle"))
            if p["score"] is not None:
                out.append(f'<path d="M{center} {y+88} L{xx} {sy(p["score"])-28}" stroke="{COLORS[p["evidence"]]}" stroke-width="1.8" stroke-dasharray="5 5" fill="none"/>')
    return ''.join(out)


def svg_shell(parts):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900"><rect width="1600" height="900" fill="#fff7e8"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">'+''.join(parts)+'</g></svg>'


def single(b):
    pts=b['points'];n=len(pts)
    parts=[text(54,52,'SEARCH TRAJECTORIES',22,800),text(54,116,b['operator'],49,800),text(54,165,f'{b["date"]} · {b["title"]} · 相对{b["reference"]}',27)]
    for i,key in enumerate(['P','S','I','E','?']):parts.append(pill(54+i*220,219,key))
    parts.append(plot(b,118,290,1405,410))
    parts += [text(54,800,f'本批{n}条正式评测记录 · {b["performance_nodes"]}个性能点 · 失败与退步保留',27,800),
              text(54,850,'底色与标签表示分析信息来源；未确认的来源标“待核验”。',25)]
    return svg_shell(parts)


def primary(history,op):
    preferred={'rad2deg_':'持续优化','uniform':'持续优化 · 尝试2','negative_':'新目标搜索 v2',
               'selu_':'独立工作流恢复','log10_':'独立工作流开发','native_group_norm':'新目标搜索 v2'}
    return next(b for b in history[op] if b['title']==preferred[op])


def overview(history,ops,page):
    parts=[text(54,52,'SEARCH TRAJECTORIES',22,800),text(54,116,f'真实优化轨迹 · 第{page}页',49,800)]
    for i,key in enumerate(['P','S','I','E','?']):parts.append(pill(58+i*220,190,key,21))
    for j,op in enumerate(ops):
        b=primary(history,op);xx=55+j*520
        parts += [text(xx,272,op,27,800),text(xx,310,b['title'],23,700)]
        parts.append(plot(b,xx+50,346,435,320,True))
        end=b['points'][-1]
        total=sum(len(batch['points']) for batch in history[op])
        parts += [text(xx,739,f'本图{len(b["points"])}条 · 相对{b["reference"]}',20,700),
                  text(xx,780,f'已收集{len(history[op])}批、{total}条评测',22,800)]
    parts += [text(54,850,'所有批次可在HTML中切换；不同批次分别作图。',25)]
    return svg_shell(parts)


def main():
    OUT.mkdir(exist_ok=True)
    history=load_history()
    counts={op:dict(batches=len(bs),evaluation_records=sum(len(b['points']) for b in bs),performance_nodes=sum(b['performance_nodes'] for b in bs)) for op,bs in history.items()}
    data=dict(schema='hgkda.trace-history/v1',scope='Discovered ledgers plus local formal summaries; incomplete historical inventory',counts=counts,operators=history)
    (OUT/'traces.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cards=[];nav=[]
    for i,(op,bs) in enumerate(history.items()):
        count=counts[op]
        nav.append(f'<button onclick="showOp({i})">{op}</button>')
        options=[];figs=[];tables=[]
        default=primary(history,op)['id']
        for b in bs:
            file=b['id']+'.svg';(OUT/file).write_text(single(b),encoding='utf-8')
            selected=' selected' if b['id']==default else ''
            options.append(f'<option value="{b["id"]}"{selected}>{b["date"]} · {b["title"]} · {len(b["points"])}条</option>')
            rows=[]
            for p in b['points']:
                score='未通过／无有效分数' if p['score'] is None else f'{p["score"]:.6f}×'
                rows.append(f'<tr><td>R{p["round"]}</td><td>{score}</td><td>{html.escape(p["action"])}</td><td>{html.escape(p.get("information_source",p.get("evidence_detail",NAMES[p["evidence"]])))}</td></tr>')
            hide='' if b['id']==default else ' hidden'
            figs.append(f'<div class="batch" id="{b["id"]}"{hide}><img src="{file}"><p class="downloads"><a href="{file}" download>SVG</a><a href="{b["id"]}.png" download>PNG</a></p><details><summary>逐轮修改与信息来源</summary><table><tr><th>评测</th><th>成绩</th><th>修改</th><th>信息来源</th></tr>{"".join(rows)}</table></details></div>')
        cards.append(f'<section class="operator" id="op{i}" hidden><h2>{op}：已收集{count["batches"]}批、{count["evaluation_records"]}条正式评测记录</h2><select onchange="showBatch({i},this.value)">{"".join(options)}</select>{"".join(figs)}</section>')
    for page,ops in enumerate([['rad2deg_','uniform','log10_'],['selu_','negative_','native_group_norm']],1):
        (OUT/f'overview-{page}.svg').write_text(overview(history,ops,page),encoding='utf-8')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>六算子 · 多批次优化轨迹</title><style>
*{box-sizing:border-box}body{margin:0;background:#fff7e8;color:#1b2434;font-family:"Microsoft YaHei",sans-serif}header{padding:28px 40px}h1{font-size:34px;margin:0 0 16px}p{font-size:22px}nav{display:flex;flex-wrap:wrap;gap:10px}button,select,summary{font:inherit;font-size:20px;padding:10px 16px}button,select{background:#fffdf8;color:#1b2434;border:1px solid #aeb6be;border-radius:16px}button{cursor:pointer}main{max-width:1680px;margin:auto;padding:0 24px 40px}img{display:block;width:100%}.operator h2{font-size:28px}.operator select{margin-bottom:18px}.downloads{display:flex;gap:24px;margin:10px 30px}a{color:#2368b1}details{background:#fffdf8;margin:16px;padding:14px}table{border-collapse:collapse;width:100%;font-size:21px}td,th{border-bottom:1px solid #d9cdb9;padding:12px;text-align:left}summary{cursor:pointer}.cover p{margin:12px 28px} [hidden]{display:none!important}</style>
<header><h1>真实优化轨迹：多批次视图</h1><p>模型调用、设备预检、正式评测分别计数；图中每个点对应正式评测结果。</p><nav><button onclick="showOp(-1)">三栏总览</button>NAV</nav></header><main><section id="covers"><div class="cover"><img src="overview-1.svg"><p><a href="overview-1.png" download>下载第一页PNG</a>　<a href="overview-1.svg" download>SVG</a></p></div><div class="cover"><img src="overview-2.svg"><p><a href="overview-2.png" download>下载第二页PNG</a>　<a href="overview-2.svg" download>SVG</a></p></div></section>CARDS</main><script>function showOp(n){document.getElementById('covers').hidden=n>=0;document.querySelectorAll('.operator').forEach((s,i)=>s.hidden=i!==n);window.scrollTo(0,0)}function showBatch(n,id){document.getElementById('op'+n).querySelectorAll('.batch').forEach(s=>s.hidden=s.id!==id)}</script></html>'''
    (OUT/'index.html').write_text(template.replace('NAV',''.join(nav)).replace('CARDS',''.join(cards)),encoding='utf-8')
    lines=['# 六算子多批次轨迹','', '沿用原样式图的米色底、三栏总览、阶段底色、绿色轨迹和节点方法标签。新增本地独立工作流正式评测，不再仅截取2～4轮片段。', '', '[打开多批次页面](index.html)。按算子切换后，可选择所有已收集批次并展开逐轮表。', '', '| 算子 | 上版图节点 | 本版已收集批次 | 正式评测记录 | 有效性能点 |','| --- | ---: | ---: | ---: | ---: |']
    previous={s['op']:len(s['points']) for s in existing_series()}
    for op,c in counts.items():lines.append(f'| `{op}` | {previous[op]} | {c["batches"]} | {c["evaluation_records"]} | {c["performance_nodes"]} |')
    lines += ['', '以上为已收集记录，不声称穷尽全部历史；账本副本按SHA去重，重复代码在不同批次的正式评测保留。',
              '模型的一次调用可能含多次编辑、编译和预检，不能都画成有效性能点。log10独立开发有24次模型调用、17份设备预检源码、12份正式性能结果；冻结阶段另有7次调用、4份正式结果。',
              '不同参考、约束与阶段的批次各自成图；不跨批次连线。历史失败或不符合当前约束的版本仅用于过程展示，不替代当前交付记录。',
              '信息来源不明的节点标“待核验”，有源码计划仍可展示修改。原件路径、SHA及评测摘要见traces.json。',
              'SELU首轮目录还保留一份无有效分数的失败正式请求f3502fad5259；它没有被拼入后续恢复曲线，来源为development/experiments/skill-validation/20261109-selu/formal-f3502fad5259/summarize-final.json。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(counts,ensure_ascii=False))


if __name__=='__main__':
    main()
