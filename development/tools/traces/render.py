"""Render data-selected optimization lines and their source-linked history."""
import html
import json
import os
from pathlib import Path

from .importer import read
from .selection import remeasurement_indices

INK, GREEN, BG = "#1b2434", "#168358", "#fff7e8"
COLORS = dict(paired="#2368b1", sim="#c55a18", knowledge="#7852a6", probe="#2368b1", ir="#596170", unknown="#596170")
EVIDENCE_LABELS = dict(paired="Profiling", sim='Simulator', knowledge='Knowledge', probe="Profiling", ir='Code analysis', unknown='Unverified')
LOCATION_LABELS = dict(device='Device kernel', host='Host launch', mixed='Mixed', ir='IR')


def node_shape(x, y, location, size=11):
    """Shapes encode the change location, independently of evidence color."""
    style = f'fill="{GREEN}" stroke="{BG}" stroke-width="3"'
    if location == "device":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size}" {style}/>'
    if location == "host":
        return f'<rect x="{x-size:.2f}" y="{y-size:.2f}" width="{size*2}" height="{size*2}" rx="1" {style}/>'
    if location == "mixed":
        return f'<path d="M {x:.2f} {y-size-3:.2f} L {x+size+3:.2f} {y:.2f} L {x:.2f} {y+size+3:.2f} L {x-size-3:.2f} {y:.2f} Z" {style}/>'
    if location == "ir":
        return f'<path class="ir-layer-node" d="M {x:.2f} {y-size-4:.2f} L {x+size+4:.2f} {y+size+2:.2f} L {x-size-4:.2f} {y+size+2:.2f} Z" {style}/>'
    raise ValueError(f"Unknown modification location: {location}")


def text(x, y, value, size=26, color=INK, anchor="start", weight=700):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(value))}</text>')


def draw(index, view):
    original = [p for p in index["evaluations"] if p["session_id"] == view["session"]]
    hidden = remeasurement_indices(original)
    numbers = {i + 1: number for number, i in enumerate(
        (i for i in range(len(original)) if i not in hidden), 1)}
    points = [p for i, p in enumerate(original) if i not in hidden]
    view = dict(view, **{key: [dict(item, point=numbers[item['point']])
        for item in view.get(key, []) if item['point'] in numbers]
        for key in ('annotations', 'nodes')})
    session = next(s for s in index["sessions"] if s["id"] == view["session"])
    detail = any("observation" in p for p in points)
    top, bottom = (335, 535) if detail else (350, 745)
    if not detail:bottom=view.get('plot_bottom',bottom)
    left, right = (360, 1340) if detail else (165, 1480)
    low, high = view["ylim"]
    y = lambda v: bottom - (v-low)/(high-low)*(bottom-top)
    xs = [left + i*(right-left)/(len(points)-1) for i in range(len(points))] if len(points)>1 else [(left+right)/2]
    coords = list(zip(xs, [y(p["score"]) if p.get("score") is not None else bottom+40 for p in points]))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc">',
             f'<title id="title">{html.escape(view["title"])}</title>',
             '<desc id="desc">Measured formal nodes connected by straight lines; arrows identify method/evidence. HTML preserves full changes and failures.</desc>',
             '<defs>']
    for key, color in COLORS.items():
        parts.append(f'<marker id="arrow-{key}" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 1 L11 6 L0 11 Z" fill="{color}"/></marker>')
    parts += ['</defs>', f'<rect width="1600" height="900" fill="{BG}"/>',
              '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">',
              text(60, 66, view["title"], min(46, 1480 / max(1, sum(1 if ord(c)>127 else .57 for c in view['title']))))]
    delivery = view.get('delivery')
    if delivery:
        parts += [text(62, 137, f"Delivered  {delivery['score']:.3f}× CANN", 32, GREEN),
                  text(62, 185, f"Full evaluation {delivery['passed']}/{delivery['total']} | acceptance threshold 0.8x", 25)]
        if view.get('result_only'):
            parts += [text(800, 410, f"{delivery['score']:.3f}×", 90, GREEN, anchor='middle'),
                      text(800, 490, 'Current delivery | relative to CANN', 36, anchor='middle'),
                      text(800, 620, 'Optimization history unavailable; showing acceptance', 32, anchor='middle'),
                      '</g>', '</svg>']
            return '\n'.join(parts)
        parts.append(text(62, 239, view.get('line_status', 'Below: historical optimization segment'), 25))
    # Only show evidence categories used in this plot. Preserve raw categories in JSON.
    used = {a['color'] for a in view['annotations']}
    legend = [k for k in ('paired', 'sim', 'knowledge', 'ir', 'unknown')
              if k in used or (k == 'paired' and 'probe' in used)]
    parts.append('<g class="plot-legend">')
    start = 1530 - 175 * len(legend)
    for i, key in enumerate(legend):
        lx = start + 175 * i
        parts += [f'<line x1="{lx}" y1="140" x2="{lx+26}" y2="140" stroke="{COLORS[key]}" stroke-width="4"/>',
                  text(lx+36, 148, EVIDENCE_LABELS[key], 24, COLORS[key])]
    locations = [key for key in LOCATION_LABELS
                 if key != 'ir' or any(node.get('location') == 'ir' for node in view.get('nodes', []))]
    positions = (845, 1035, 1225, 1415) if len(locations) == 4 else (1015, 1195, 1390)
    for lx, key in zip(positions, locations):
        parts += [node_shape(lx, 191, key, size=8), text(lx+23, 199, LOCATION_LABELS[key], 24)]
    parts.append('</g>')
    if not detail:
        for tick in [high*i/5 for i in range(1,6)]:
            yy = y(tick)
            parts += [f'<line x1="120" y1="{yy}" x2="1510" y2="{yy}" stroke="#d8d5c8" stroke-width="1"/>',
                      text(105, yy+10, f"{tick:.0f}×" if high>=10 else f"{tick:.1f}×", 30, anchor="end")]
        parts.append(text(62, 307, 'Speedup', 32))
    else:
        parts.append(text(62, 307, 'Speedup', 32))
    segments, segment = [], []
    for xy, point in zip(coords, points):
        if point.get("score") is None:
            if segment:segments.append(segment)
            segment=[]
        else:segment.append(xy)
    if segment:segments.append(segment)
    for segment in segments:
        path = "M " + " L ".join(f"{x:.2f} {yy:.2f}" for x, yy in segment)
        parts.append(f'<path class="performance-line" d="{path}" stroke="{GREEN}" stroke-width="6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    node_options = {n["point"]: n for n in view.get("nodes", [])}
    for n, ((x, yy), point) in enumerate(zip(coords, points), 1):
        options = node_options.get(n, {})
        location = options.get("location", "device")
        score_label = f"{point['score']:.6f}×" if point.get('score') is not None else 'No valid score'
        parts.append(f'<g class="performance-node" data-evaluation="{point["id"]}" data-location="{location}"><title>{html.escape(point["method"])} | {score_label} | {str(point.get("source_sha256") or 'Unknown')[:12]}</title>')
        parts.append(node_shape(x, yy, location))
        # Labels on alternating sides keep the shallow rises and regressions visible.
        label_y = yy + (43 if n % 2 == 0 and not detail else -23)
        if options.get("label_side") == "above":
            label_y = yy - 25
        if detail and n > 1:
            label_y = yy + 38
        parts.append(text(x, label_y, f"{point['score']:.3f}×" if point.get('score') is not None else 'Evaluation failed', 34 if detail else 30, anchor="middle"))
        if not detail:
            parts.append(text(x, bottom + 45, n, 30, anchor="middle"))
        parts.append('</g>')
    for annotation in view["annotations"]:
        x, yy = coords[annotation["point"]-1]
        color = COLORS[annotation["color"]]
        tx, ty = annotation["position"]
        ax, ay = annotation["arrow"]
        dx, dy = x-ax, yy-ay
        length = max((dx*dx+dy*dy)**.5, 1)
        endx, endy = x-dx/length*20, yy-dy/length*20
        parts.append(f'<g class="method-annotation" data-point="{annotation["point"]}" data-evidence="{annotation["color"]}">')
        parts += [text(tx, ty, annotation["method"], 38, color),
                  f'<path d="M {ax} {ay} L {endx:.2f} {endy:.2f}" fill="none" stroke="{color}" stroke-width="2.8" marker-end="url(#arrow-{annotation["color"]})"/>', '</g>']
    if detail:
        case = points[0]["observation"]["case"]["inputs"]["spec"]
        parts.append(text(62, 608, f"{case['dtype']} {case['shape']} | waits from a single-core simulator window", 32))
        rows = [('Hardware task time', lambda o: f"{o['device_task_us']:.2f} μs"),
                ('MTE2 wait-only', lambda o: f"{o['pipelines']['MTE2']['wait_only_ratio']:.2%}"),
                ('Vector wait-only', lambda o: f"{o['pipelines']['VECTOR']['wait_only_ratio']:.2%}")]
        for r, (label, getter) in enumerate(rows):
            yy = 660 + 46*r
            parts.append(text(62, yy, label, 32))
            for x, p in zip(xs, points):
                parts.append(text(x, yy, getter(p["observation"]), 34, anchor="middle"))
        parts.append(text(62, 840, 'Partial recovery remains below start; retain original.', 36))
    else:
        parts.append(text(810, 853, 'Optimization step' if len(points)>1 else 'Saved version', 30, anchor="middle"))
        if view.get("gap"):
            parts.append(text(810, 640, view["gap"], 36, anchor="middle"))
    parts += ['</g>', '</svg>']
    return "\n".join(parts)


CSS = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#fff7e8;color:#1b2434;font:18px/1.65 'Microsoft YaHei',sans-serif}main{max-width:1600px;margin:auto;padding:20px 28px 60px}h1{font-size:32px;margin:0}h2{font-size:27px;margin:20px 0 10px}p{margin:10px 0}nav{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}button,.button{font:inherit;font-weight:700;padding:10px 20px;border:2px solid #168358;background:#fff7e8;color:#168358;border-radius:8px;cursor:pointer;text-decoration:none}button.active{background:#168358;color:white}a{color:#2368b1}figure{margin:0}figure svg{display:block;width:100%;height:auto}section[hidden]{display:none}.exports{display:flex;gap:22px}details{margin:12px 0;padding:12px 16px;background:#fffaf0;border:1px solid #cec7b8;border-radius:8px}summary{cursor:pointer;font-weight:700}.badge{display:inline-block;background:#e3f0df;padding:3px 10px;border-radius:5px;margin-right:8px}table{width:100%;border-collapse:collapse;font-size:17px}td,th{text-align:left;border-bottom:1px solid #d8d5c8;padding:9px 10px;vertical-align:top}th{background:#eee9da}code{font-size:15px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;padding:12px;background:#eee9da;font-size:14px}ul{padding-left:24px}.facts{display:flex;flex-wrap:wrap;gap:12px}.note{padding:12px 16px;border-left:5px solid #c55a18;background:#fbe8d6}.events{max-height:660px;overflow:auto}li{margin:7px 0}.links a{margin-right:18px}.small-table{table-layout:fixed}.small-table td:nth-child(2){width:45%}
"""


def render(index_path, output, root):
    index = read(index_path)
    visible_views = [view for view in index["views"] if view.get("present", True)]
    output.mkdir(parents=True, exist_ok=True)
    def link(reference, label):
        if isinstance(reference, str):
            path = reference
        else:
            path = reference["path"]
        relative = Path(os.path.relpath(root / path, output)).as_posix()
        return f'<a href="{html.escape(relative, quote=True)}" target="_blank">{html.escape(label)}</a>'
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>log10_ continuous optimization trace</title>', f'<style>{CSS}</style><main>',
             f'<h1>{html.escape(index["operator"])} continuous optimization trace</h1>',
             '<p>Continuous traces retain evidence reads, edits, preflights, and formal results; charts show one experimental sequence.</p>',
             '<nav>']
    for n, view in enumerate(visible_views):
        parts.append(f'<button class="{"active" if n == 0 else ""}" data-tab="{view["id"]}">{'Mainline' if n == 0 else 'Simulator detail'}</button>')
    parts += ['<button data-tab="history">Full history and continuous records</button>', '</nav>']
    for n, view in enumerate(visible_views):
        svg = draw(index, view)
        (output / (view["id"] + ".svg")).write_text(svg, encoding="utf-8")
        parts += [f'<section id="{view["id"]}" {"hidden" if n else ""}><figure>{svg}</figure>',
                  f'<p class="exports"><a href="{view["id"]}.svg">SVG</a><a href="{view["id"]}.png">PNG</a><a href="#detail-{view["session"]}" class="jump-history">Expand sequence records</a></p>']
        if n == 1:
            parts.append('<p class="note">Separate frozen-generation region experiment, disconnected from development 0.851x. Shows prefetch regression, diagnosis, and recovery without claiming improvement over its start.</p>')
        parts.append('</section>')
    parts += ['<section id="history" hidden><h2>History</h2>',
              f'<p>{link(index["history"]["html"], 'All 28 formal evaluations (7 batches)')}　{link(index_path.relative_to(root).as_posix(), 'Unified trace index JSON')}</p>',
              '<p>Two continuous sequences are presented. Other batches remain in historical charts; 28 points do not form one uninterrupted source chain.</p>',
              '<p class="note">Order recovered from model-call numbers/tool lines; formal evaluations keep original UTC timestamps. Missing times remain missing; source anchors locate outer events without implying exact interleaving.</p>']
    evidence_by_id = {e["id"]: e for e in index["evidence"]}
    for session in index["sessions"]:
        imported = read(root / session["imported"]["path"])
        counts = session["counts"]
        points = [p for p in index["evaluations"] if p["session_id"] == session["id"]]
        parts += [f'<h2 id="detail-{session["id"]}">{html.escape(session["title"])}</h2>',
                  '<div class="facts">' + ''.join(f'<span class="badge">{v} {k}</span>' for k, v in
                      [(' model calls', counts["model_calls"]), (' source edits', counts["source_transitions"]),
                       (' device preflights', counts["preflight_sources"]), (' formal evaluations', counts["formal_evaluations"])]) + '</div>',
                  '<p class="links">' + link(session["event_stream"], 'Continuous events JSONL') + link(session["imported"], 'Version/tool index JSON') + link(session["config"], 'Original experiment configuration') + '</p>',
                  '<details open><summary>Formal nodes: methods, ancestry, and artifacts</summary><table><thead><tr><th>Node</th><th>Method / result</th><th>Ancestry</th><th>Evidence</th></tr></thead><tbody>']
        for point in points:
            edge = next((e for e in index["measured_edges"] if e["target"] == point["id"]), None)
            relation = (f"Since the previous measured version, {len(edge['transition_ids'])} saved source edits" if edge else 'First formal point in this experiment')
            parts.append(f'<tr><td>{point["number"]}<br><code>{point["source_sha256"][:12]}</code></td>'
                         f'<td>{html.escape(point["method"])}<br><b>{point["score"]:.6f}×</b> · {point["passed"]}/{point["total"]}<br>UTC {point["started_at"]}</td>'
                         f'<td>{relation}<br>{html.escape(str(point["occurrence"]["call_id"] or 'Supplied frozen start'))}</td>'
                         f'<td>{link(point["source"], 'Source')} · {link(point["result"], 'KGS result')} · {link(point["request"], 'Contract request')}</td></tr>')
        parts += ['</tbody></table></details>', '<details><summary>Arrows: pre-change guidance separate from post-change verification</summary><ul>']
        for view in index["views"]:
            if view["session"] != session["id"]:
                continue
            for annotation in view["annotations"]:
                parts.append(f'<li><b>Node {annotation["point"]} · {html.escape(annotation["method"])}</b><ul>')
                for eid in annotation["evidence"]:
                    e = evidence_by_id[eid]
                    parts.append(f'<li>{link(e["source"],e["category"])} · <code>{e["stage"]}</code>：{html.escape(e["note"])}</li>')
                parts.append('</ul></li>')
        parts += ['</ul></details>', '<details><summary>Model calls: tool reads, edits, and failures</summary>']
        for call in imported["calls"]:
            parts += [f'<details><summary>{call["invocation"]} · {call["edits"]} edits | {call["tool_calls"]} tool calls | {call["unsuccessful_tools"]} errors or missing results</summary>',
                      f'<p><code>{str(call["input_sha256"] or 'Input file unknown')[:24]} → {str(call["output_sha256"])[:24]}</code></p>',
                      f'<p>{link(call["prompt"], 'Model prompt')} · {link(call["source"], 'Original tool log (with results)')}</p>',
                      '<div class="events"><table><thead><tr><th>Line</th><th>Action</th><th>Object / description</th></tr></thead><tbody>']
            for event in imported["events"]:
                if event.get("call_id") != call["id"] or event["kind"] not in ("tool_call", "candidate_change"):
                    continue
                line = event["source"]["line"]
                if event["kind"] == "candidate_change":
                    action = 'Source edit'
                    desc = str(event["input_sha256"] or 'Unknown')[:12] + " → " + event["output_sha256"][:12]
                else:
                    action = event["tool"] + (' | error / no result' if not event["success"] else "")
                    desc = event.get("path") or event.get("description") or event["category"]
                parts.append(f'<tr><td>{line}</td><td>{html.escape(action)}</td><td>{html.escape(desc)}</td></tr>')
            parts.append('</tbody></table></div></details>')
        parts += ['</details>', '<details><summary>Device preflight, including failures without formal scores</summary><table><tr><th>Source</th><th>Status</th><th>Artifact</th></tr>']
        for p in index["preflights"]:
            if p["session_id"] == session["id"]:
                parts.append(f'<tr><td><code>{p["source_sha256"][:12]}</code></td><td>{p["status"]}</td><td>{link(p["source"],'Source')} · {link(p["result"],'Preflight result')}</td></tr>')
        parts.append('</table></details>')
        if "observation" in points[0]:
            parts.append('<details><summary>Simulator originals and scope</summary><ul>')
            for p in points:
                o = p["observation"]
                parts.append(f'<li>Node {p["number"]}：{link(o["paired"],'Paired hardware metrics')} · {link(o["simulator"],'Hotspot regions / pipeline windows')}<pre>{html.escape(json.dumps(o["qualification"],ensure_ascii=False,indent=2))}</pre></li>')
            parts.append('</ul><p>Single-core wait fractions are separate from a chip-wide critical path. Original reports preserve capture errors and hardware/simulator core-count differences.</p></details>')
    parts += ['</section></main>', '''<script>
function show(id){document.querySelectorAll('main>section').forEach(s=>s.hidden=s.id!==id);document.querySelectorAll('button[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===id));}
document.querySelectorAll('button[data-tab]').forEach(b=>b.addEventListener('click',()=>show(b.dataset.tab)));
document.querySelectorAll('.jump-history').forEach(a=>a.addEventListener('click',()=>{show('history');setTimeout(()=>document.querySelector(a.hash).scrollIntoView(),0)}));
</script></html>''']
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def export_png(index_path, output):
    """Optional browser export; the core index and SVG need no browser package."""
    from playwright.sync_api import sync_playwright

    index = read(index_path)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
        try:
            for view in index["views"]:
                if not view.get("present", True):
                    continue
                page.goto((output / (view["id"] + ".svg")).as_uri())
                page.screenshot(path=str(output / (view["id"] + ".png")))
        finally:
            browser.close()
