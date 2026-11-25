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
NAMES = {"P": 'Paired analysis', "S": 'Simulator', "I": 'Source / IR', "E": 'Targeted probe', "?": 'Unverified'}


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
        ("strict-night-20261102", 'Strict-constraint search', "2026-11-02"),
        ("night-20261021-v3", 'First search', "2026-10-21"),
        ("paired-pointwise-20261022", 'Paired-analysis batch', "2026-10-22"),
        ("buffer-pointwise-20261022", 'Buffer/tiling batch', "2026-10-22"),
        ("endurance-20261022", 'Continued optimization', "2026-10-22"),
        ("next25-20261029-v2", 'New-target search v2', "2026-10-29"),
        ("next25-20261029", 'New-target search', "2026-10-29"),
    ]:
        if token in path:
            if "uniform" in path and "endurance" in path:
                name += ' | attempt 2' if "attempt-2" in path else ' | attempt 1'
            return name, date
    return 'Historical search', 'Date unverified'


def label_from_plan(r):
    p = r.get("plan") or {}
    s = (p.get("strategy", "") + " " + p.get("code_changes", "")).lower()
    if p.get("kind") == "baseline":
        return 'Establish batch start'
    if "stream" in s and ("persistent" in s or "single-wave" in s):
        return 'Fixed-program streaming loop'
    if "two-pointer" in s:
        return 'Two-pointer prefetch'
    if "bf16" in s and "float16" in s:
        return 'BF16 intermediates and tiling'
    if "counter" in s and "int32" in s:
        return 'Stabilize counter integer types'
    if "philox" in s:
        return 'Rewrite random computation'
    if "quantize" in s or "dtype-bound" in s:
        return 'Repair random-bound quantization'
    if "multi-tile" in s or "unrolled" in s and "tiles" in s:
        return 'Multiple tiles per program'
    if "closure" in s or "wrapper" in s or "bookkeeping" in s:
        return 'Simplify host calls'
    if "launch" in s:
        return 'Adjust launch/compilation caches'
    if "tile" in s or "block" in s:
        return 'Adjust tiling and data processing'
    return 'Change unverified'


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
        batch = dict(operator=op, title=title, date=date, reference='Batch KGS reference',
                     source=dict(path=ledger["path"], sha256=ledger["sha256"]), points=[])
        for r in ledger["rounds"]:
            e = r["evaluation"]
            point = dict(round=r["round"], score=e.get("geo_mean"), status=e.get("status"),
                         passed=e.get("num_passed"), total=e.get("num_workloads"),
                         source_sha256=r["source_sha256"], action=label_from_plan(r),
                         evidence="?", evidence_detail='Specific evidence source unverified', plan=r.get("plan"),
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
        ("log10_", 'Independent workflow development', "2026-11-10", "20261110-log10-development", [
            ("a4d38b964aa6", 'Compliant Triton layout and math', "I"),
            ("a73e15786d68", 'Bounded-grid loop', "?"),
            ("4eca306ca042", 'Cache not yet hit', "E"),
            ("2f90b7925721", 'Repair cold-cache registration', "E"),
            ("0fefe838c58c", 'Cache host dispatch plan', "E"),
            ("4c184cdc36f4", 'Reduce repeated mode queries', "E"),
            ("e62c6d636097", 'Double-prefetch experiment', "?"),
            ("b2a52734c432", 'Integer addresses in hot path', "I"),
            ("d519b459c915", 'Separate full/tail tiles', "I"),
            ("b64e7d8351df", 'Unmasked paired loop', "I"),
            ("69699663ecc7", 'Adjacent payloads + compile-time loops', "I"),
            ("8600ce070318", 'Input-dtype math', "I"),
        ]),
        ("log10_", 'Frozen-method regeneration', "2026-11-10", "20261110-log10-frozen", [
            ("40cfe1a83343", 'First frozen-method version', "?"),
            ("0261f02aee72", 'Continue using full feedback', "?"),
            ("b6b10ceb10e3", 'Improve launching and layout', "?"),
            ("45c7380a8062", 'Tiling and launch-cache hits', "E"),
        ]),
        ("selu_", 'Independent workflow recovery', "2026-11-09", "20261109-selu-recovery", [
            ("3cec1212c3b7", 'Independent generated start', "P"),
            ("c6255855aa16", 'Cache compiled launcher', "E"),
            ("80bfefff4ff8", 'Raw stream + simpler cache keys', "E"),
            ("0fed8d43398d", 'NoWait interface + larger tiles', "I"),
            ("5e9e4d482620", 'Direct dispatch on cache hit', "E"),
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
    batch = dict(operator="log10_", title='Simulator region/wait revision', date="2026-11-10", reference="CANN",
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
            out += [f'<path d="M{xs[i]-7} {yy-7} l14 14 m0 -14 l-14 14" stroke="#b14e24" stroke-width="3"/>',text(xs[i],yy-18,'Failed',18 if compact else 23,700,"middle","#b14e24")]
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
    parts=[text(54,52,'SEARCH TRAJECTORIES',22,800),text(54,116,b['operator'],49,800),text(54,165,f'{b["date"]} · {b["title"]} | relative to {b["reference"]}',27)]
    for i,key in enumerate(['P','S','I','E','?']):parts.append(pill(54+i*220,219,key))
    parts.append(plot(b,118,290,1405,410))
    parts += [text(54,800,f'Batch {n} formal evaluations | {b["performance_nodes"]} performance points; failures/regressions retained',27,800),
              text(54,850,'Background and labels show evidence sources; unresolved sources are marked unverified.',25)]
    return svg_shell(parts)


def primary(history,op):
    preferred={'rad2deg_':'Continued optimization','uniform':'Continued optimization | attempt 2','negative_':'New-target search v2',
               'selu_':'Independent workflow recovery','log10_':'Independent workflow development','native_group_norm':'New-target search v2'}
    return next(b for b in history[op] if b['title']==preferred[op])


def overview(history,ops,page):
    parts=[text(54,52,'SEARCH TRAJECTORIES',22,800),text(54,116,f' measured trace | page {page}',49,800)]
    for i,key in enumerate(['P','S','I','E','?']):parts.append(pill(58+i*220,190,key,21))
    for j,op in enumerate(ops):
        b=primary(history,op);xx=55+j*520
        parts += [text(xx,272,op,27,800),text(xx,310,b['title'],23,700)]
        parts.append(plot(b,xx+50,346,435,320,True))
        end=b['points'][-1]
        total=sum(len(batch['points']) for batch in history[op])
        parts += [text(xx,739,f'This chart: {len(b["points"])} records | relative to {b["reference"]}',20,700),
                  text(xx,780,f'Collected {len(history[op])} batches, {total} evaluations',22,800)]
    parts += [text(54,850,'Switch all batches in HTML; each batch has its own chart.',25)]
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
            options.append(f'<option value="{b["id"]}"{selected}>{b["date"]} · {b["title"]} · {len(b["points"])} records</option>')
            rows=[]
            for p in b['points']:
                score='Failed / no valid score' if p['score'] is None else f'{p["score"]:.6f}×'
                rows.append(f'<tr><td>R{p["round"]}</td><td>{score}</td><td>{html.escape(p["action"])}</td><td>{html.escape(p.get("information_source",p.get("evidence_detail",NAMES[p["evidence"]])))}</td></tr>')
            hide='' if b['id']==default else ' hidden'
            figs.append(f'<div class="batch" id="{b["id"]}"{hide}><img src="{file}"><p class="downloads"><a href="{file}" download>SVG</a><a href="{b["id"]}.png" download>PNG</a></p><details><summary>Per-round changes and evidence</summary><table><tr><th>Evaluation</th><th>Score</th><th>Change</th><th>Evidence</th></tr>{"".join(rows)}</table></details></div>')
        cards.append(f'<section class="operator" id="op{i}" hidden><h2>{op}: collected {count["batches"]} batches, {count["evaluation_records"]} formal evaluations</h2><select onchange="showBatch({i},this.value)">{"".join(options)}</select>{"".join(figs)}</section>')
    for page,ops in enumerate([['rad2deg_','uniform','log10_'],['selu_','negative_','native_group_norm']],1):
        (OUT/f'overview-{page}.svg').write_text(overview(history,ops,page),encoding='utf-8')
    template="""<!doctype html><html lang="en"><meta charset="utf-8"><title>Six operators · Multi-batch traces</title><style>
*{box-sizing:border-box}body{margin:0;background:#fff7e8;color:#1b2434;font-family:"Microsoft YaHei",sans-serif}header{padding:28px 40px}h1{font-size:34px;margin:0 0 16px}p{font-size:22px}nav{display:flex;flex-wrap:wrap;gap:10px}button,select,summary{font:inherit;font-size:20px;padding:10px 16px}button,select{background:#fffdf8;color:#1b2434;border:1px solid #aeb6be;border-radius:16px}button{cursor:pointer}main{max-width:1680px;margin:auto;padding:0 24px 40px}img{display:block;width:100%}.operator h2{font-size:28px}.operator select{margin-bottom:18px}.downloads{display:flex;gap:24px;margin:10px 30px}a{color:#2368b1}details{background:#fffdf8;margin:16px;padding:14px}table{border-collapse:collapse;width:100%;font-size:21px}td,th{border-bottom:1px solid #d9cdb9;padding:12px;text-align:left}summary{cursor:pointer}.cover p{margin:12px 28px} [hidden]{display:none!important}</style>
<header><h1>Measured traces：Multi-batch view</h1><p>Model calls、device preflights、and formal evaluations are counted separately；Each point is a formal result。</p><nav><button onclick="showOp(-1)">Three-column overview</button>NAV</nav></header><main><section id="covers"><div class="cover"><img src="overview-1.svg"><p><a href="overview-1.png" download>Download page 1 PNG</a>　<a href="overview-1.svg" download>SVG</a></p></div><div class="cover"><img src="overview-2.svg"><p><a href="overview-2.png" download>Download page 2 PNG</a>　<a href="overview-2.svg" download>SVG</a></p></div></section>CARDS</main><script>function showOp(n){document.getElementById('covers').hidden=n>=0;document.querySelectorAll('.operator').forEach((s,i)=>s.hidden=i!==n);window.scrollTo(0,0)}function showBatch(n,id){document.getElementById('op'+n).querySelectorAll('.batch').forEach(s=>s.hidden=s.id!==id)}</script></html>"""
    (OUT/'index.html').write_text(template.replace('NAV',''.join(nav)).replace('CARDS',''.join(cards)),encoding='utf-8')
    lines=['# Six-operator multi-batch traces','', 'Retains the original beige background, three-column overview, phase shading, green traces, and method labels. Adds local independent-workflow formal evaluations beyond short 2-4-round excerpts.', '', '[Open multi-batch view](index.html). Select an operator, choose any collected batch, and expand per-round details.', '', '| Operator | Previous chart nodes | Collected batches | Formal evaluations | Valid performance points |','| --- | ---: | ---: | ---: | ---: |']
    previous={s['op']:len(s['points']) for s in existing_series()}
    for op,c in counts.items():lines.append(f'| `{op}` | {previous[op]} | {c["batches"]} | {c["evaluation_records"]} | {c["performance_nodes"]} |')
    lines += ['', 'Collected records are not an exhaustive history; deduplicate ledger copies by SHA but retain same-source evaluations across batches.',
              'A model call can include multiple edits, compilations, and preflights; only formal outcomes become performance points. Independent log10 development has 24 calls, 17 preflight sources, and 12 formal results; frozen generation adds 7 calls and 4 results.',
              'Batches with different references, constraints, or stages use separate disconnected charts. Historical failures/out-of-contract versions do not replace current deliveries.',
              'Unresolved guidance is marked unverified; source plans may still identify changes. See traces.json for original paths, SHA, and summaries.',
              "SELU's first run also retains failed request f3502fad5259 without a valid score. It is separate from the recovery curve: development/experiments/skill-validation/20261109-selu/formal-f3502fad5259/summarize-final.json.",'']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(counts,ensure_ascii=False))


if __name__=='__main__':
    main()
