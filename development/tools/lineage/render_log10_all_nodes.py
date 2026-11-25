"""Draw all 28 recorded log10 evaluations with method/evidence arrows."""
import html
import json
import math
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "development/reports/operator-traces-history-20261116"
WIDTH, HEIGHT = 3200, 2000
INK, GREEN = "#1b2434", "#168358"
COLORS = {"P": "#2368b1", "S": "#c55a18", "I": "#7852a6", "E": "#168358", "?": "#455269"}
LABELS = {"P": 'Paired profiling', "S": 'Simulator + IR', "I": 'Source / IR', "E": 'Targeted probe', "?": 'Source unverified'}


def text(x, y, value, size=30, weight=700, anchor="start", color=INK, extra=""):
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}" {extra}>{html.escape(str(value))}</text>'


def split_label(value, width=19):
    """Wrap labels in display columns, keeping Chinese and ASCII readable."""
    lines, line, columns = [], "", 0
    for char in value:
        size = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if columns + size > width and line:
            lines.append(line)
            line, columns = "", 0
        line += char
        columns += size
    if line:
        lines.append(line)
    return lines


def main():
    doc = json.loads((BASE / "traces.json").read_text(encoding="utf-8"))
    batches = doc["operators"]["log10_"]
    points = []
    for bi, batch in enumerate(batches):
        for pi, point in enumerate(batch["points"]):
            points.append(dict(point, batch_index=bi, local_index=pi,
                               batch_title=batch["title"], global_index=len(points)+1))
    assert len(points) == 28 and all(p["score"] is not None for p in points)
    # Replace coarse keyword summaries with the actual recorded changes.
    overrides = {
        2: 'Direct launch + larger tiles',
        4: 'Prebind launch and simplify calls',
        6: 'Cache launch arguments by address',
        7: 'Multiple tiles per program',
        9: 'Larger payload + double prefetch',
        12: 'Add launch cache (miss)',
    }
    for p in points:
        p["action"] = overrides.get(p["global_index"], p["action"])
    x0, x1, top, bottom = 135, 3060, 790, 1250
    xs = [x0+i*(x1-x0)/27 for i in range(28)]
    sy = lambda value: bottom - value/1.0*(bottom-top)
    short_names = ['10.21 first batch', '10.22 paired', '10.22 tiling', '10.29 new target',
                   '11.10 independent development', '11.10 frozen regeneration', '11.10 simulator revision']
    fills = ["#e7edf5", "#e3eff8", "#e3f0df", "#eee6f5", "#e3eff8", "#e3f0df", "#fbe8d6"]
    defs = ['<defs>']
    for key, color in COLORS.items():
        safe_key = "unknown" if key == "?" else key
        defs.append(f'<marker id="arrow-{safe_key}" markerWidth="12" markerHeight="10" refX="11" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L12 5 L0 10 Z" fill="{color}"/></marker>')
    defs.append('</defs>')
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="3200" height="2000" viewBox="0 0 3200 2000" role="img" aria-labelledby="title description">',
             '<title id="title">log10_: all 28 evaluations with method/evidence arrows</title>',
             '<desc id="description">All 28 formal evaluations from seven batches; 21 change arrows show methods and evidence categories. Separate batches are disconnected.</desc>',
             ''.join(defs), '<rect width="3200" height="2000" fill="#fff7e8"/>',
             '<g font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif">',
             text(70, 65, "SEARCH TRAJECTORIES", 30, 800),
             text(70, 145, 'log10_: all 28 evaluations', 58, 800),
             text(70, 198, 'Arrows: method + evidence category', 34)]
    for i, key in enumerate(["P", "S", "I", "E", "?"]):
        x = 1430 + i*340
        label = f"{key} {LABELS[key]}"
        parts.append(text(x, 135, label, 29, 800, color=COLORS[key]))
    # Batch partitions retain each evaluation's own reference.
    start = 0
    headings=[]
    for bi, batch in enumerate(batches):
        count = len(batch["points"])
        left = 105 if start == 0 else (xs[start-1]+xs[start])/2
        right = 3115 if start+count == 28 else (xs[start+count-1]+xs[start+count])/2
        parts.append(f'<rect x="{left:.2f}" y="740" width="{right-left:.2f}" height="530" fill="{fills[bi]}"/>')
        headings.append(f'<rect x="{left:.2f}" y="685" width="{right-left:.2f}" height="45" fill="#fff7e8"/>')
        headings.append(text((left+right)/2, 716, short_names[bi], 28, 800, "middle"))
        reference = "CANN" if batch["reference"] == "CANN" else 'Batch KGS reference'
        headings.append(f'<rect x="{left+4:.2f}" y="740" width="{right-left-8:.2f}" height="35" fill="{fills[bi]}"/>')
        headings.append(text((left+right)/2, 766, reference, 25, 700, "middle"))
        if start:
            parts.append(f'<path d="M{left:.2f} 740 V1270" stroke="#9da5ad" stroke-width="2" stroke-dasharray="8 8"/>')
        start += count
    for tick in [0, .25, .5, .75, 1.0]:
        yy = sy(tick)
        parts.append(f'<path d="M105 {yy:.2f} H3115" stroke="#c7cbd1" stroke-width="1.5"/>')
        parts.append(text(88, yy+10, f"{tick:.2f}×", 27, 700, "end"))
    parts.append('<path d="M105 780 V1270 H3115" stroke="#455269" stroke-width="2.5" fill="none"/>')
    # Direct arrows for every non-baseline change. Six label lanes leave the
    # complete curve visible, rather than moving methods into separate cards.
    annotation_records = []
    annotation_labels = []
    occupied = [[] for _ in range(6)]
    lane_y = [255, 405, 555, 1410, 1580, 1750]
    for i, p in enumerate(points):
        if p["local_index"] == 0:
            continue
        key = p["evidence"]
        color = COLORS[key]
        center = min(2990, max(230, xs[i]))
        preferred = i % 6
        lane = next((j for j in [(preferred+k)%6 for k in range(6)]
                     if all(abs(center-other)>410 for other in occupied[j])), None)
        assert lane is not None, p["global_index"]
        occupied[lane].append(center)
        yy = lane_y[lane]
        lines = split_label(p["action"])
        assert len(lines) <= 2, (p["action"], lines)
        block_height = len(lines)*37+38
        safe_key = "unknown" if key == "?" else key
        node_y = sy(p["score"])
        source_y = yy+block_height-9
        arrow_start = yy+block_height+4 if lane < 3 else yy-15
        arrow_end = node_y-15 if lane < 3 else node_y+15
        parts.append(f'<path class="evidence-arrow" data-target="eval-{i+1:02}" d="M{center:.2f} {arrow_start:.2f} L{xs[i]:.2f} {arrow_end:.2f}" stroke="{color}" stroke-width="2.5" marker-end="url(#arrow-{safe_key})" fill="none"/>')
        annotation_labels.append(f'<rect x="{center-205:.2f}" y="{yy-35:.2f}" width="410" height="{block_height+38:.2f}" fill="#fff7e8"/>')
        for li, line in enumerate(lines):
            prefix = f"{i+1:02}  " if li == 0 else ""
            annotation_labels.append(text(center, yy+li*37, prefix+line, 31, 800, "middle"))
        annotation_labels.append(text(center, source_y, f"{key} · {LABELS[key]}", 28, 800, "middle", color))
        annotation_records.append(dict(target=f"eval-{i+1:02}", method=p["action"],
                                       evidence_category=key, evidence_label=LABELS[key], lane=lane))
    # Draw lines independently inside batches. No invented cross-contract gain.
    previous = None
    for i,p in enumerate(points):
        xx, yy = xs[i], sy(p["score"])
        if p["local_index"] == 0:
            previous = None
        if previous:
            px,py = previous
            parts.append(f'<path class="trajectory" d="M{px:.2f} {py:.2f} L{xx:.2f} {yy:.2f}" fill="none" stroke="{GREEN}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>')
        parts.append(f'<circle class="evaluation-point" id="eval-{i+1:02}" data-score="{p["score"]}" data-batch="{p["batch_index"]+1}" cx="{xx:.2f}" cy="{yy:.2f}" r="10" fill="{GREEN}" stroke="#fffdf8" stroke-width="3"><title>{i+1:02}: {html.escape(p["action"])} · {p["score"]:.6f}×</title></circle>')
        parts.append(text(xx, yy+38 if i%2 else yy-24, f"{p['score']:.3f}", 28, 800, "middle"))
        parts.append(text(xx, 1308, f"{i+1:02}", 29, 800, "middle"))
        if p["local_index"] == 0:
            parts.append(text(xx, 1340, 'Start', 23, 700, "middle"))
        previous = (xx,yy)
    parts.extend(headings)
    parts.extend(annotation_labels)
    parts += [text(70, 1960, 'X-axis: collected evaluations 01-28; disconnected batches use their own references.', 30), '</g></svg>']
    svg = BASE / "log10-all-28.svg"
    svg.write_text("\n".join(parts), encoding="utf-8")
    (BASE / "log10-all-28.json").write_text(json.dumps(dict(points=points, annotations=annotation_records,
                                                          point_count=28, arrow_count=len(annotation_records)),
                                                     ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>log10_ · 28-node optimization trace</title><style>
body{margin:0;background:#fff7e8;font-family:"Microsoft YaHei",sans-serif;color:#1b2434}nav{display:flex;gap:20px;align-items:center;padding:16px 28px;position:sticky;top:0;background:#fff7e8;z-index:2}button,a{font:inherit;font-size:20px}button{padding:8px 20px;background:#fffdf8;border:1px solid #9da5ad;border-radius:10px;cursor:pointer}a{color:#2368b1}main{overflow:auto}img{display:block;width:100%;max-width:none}</style>
<nav><strong>28 nodes · 21 method/evidence ／evidence arrows</strong><button onclick="zoom(1)">Fit width</button><button onclick="zoom(2)">Zoom 2x</button><a href="log10-all-28.svg" download>SVG</a><a href="log10-all-28.png" download>PNG</a></nav><main><img id="trace" src="log10-all-28.svg" alt="log10 all 28 evaluations with method/evidence arrows"></main><script>function zoom(n){document.getElementById('trace').style.width=(100*n)+'%'}</script></html>"""
    (BASE / "log10-all-28.html").write_text(page,encoding="utf-8")
    gallery=BASE / "index.html"
    if gallery.exists():
        content=gallery.read_text(encoding="utf-8")
        marker='<!-- all-log10-link -->'
        if marker not in content:
            content=content.replace('<nav>', '<p>'+marker+'<a href="log10-all-28.html">log10_: all 28 nodes and method/evidence arrows in one view</a></p><nav>', 1)
            gallery.write_text(content,encoding="utf-8")
    index=BASE / "README.md"
    if index.exists():
        content=index.read_text(encoding="utf-8")
        if 'log10-all-28.html' not in content:
            content=content.replace('\n\n', '\n\n[log10_ complete 28-node view with method/evidence arrows](log10-all-28.html) | [PNG](log10-all-28.png) | [SVG](log10-all-28.svg)\n\n', 1)
            index.write_text(content,encoding="utf-8")
    print(f"Wrote {svg}: {len(points)} points, {len(annotation_records)} method/evidence arrows.")


if __name__ == "__main__":
    main()
