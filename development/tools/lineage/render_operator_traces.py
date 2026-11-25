"""Render six real optimization episodes, preserving setbacks and source labels."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "development/reports"
OUT = REPORTS / "operator-traces-20261116"
INK = "#17263a"
GREEN = "#168358"
ORANGE = "#b35b13"
BLUE = "#225fa4"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Labels summarize contemporary plans and conclusions. They are information
# sources, not estimates of each tool's independent causal contribution.
SPECS = [
    dict(op="rad2deg_", key="rad2deg_-endurance-20261022-attempt-1/", date="2026-10-22",
         title='Larger tiles after launch-path repair', note='Reduce fixed call cost before large-input device time.',
         stages=[
             ('Repair in-place and layout semantics', 'Start / source review', 'Establish a correct complete start', "base"),
             ('Reuse compiled launch entry', 'Evidence: host-time probe', 'Small inputs: about 65 -> 14 us', "probe"),
             ('Large-input tiles: 32768', 'Evidence: tile comparison', 'Large inputs: about 77 -> 28-31 us', "probe"),
             ('Reduce intermediate conversions', 'Evidence: source + type comparison', 'Small gain verified by full evaluation', "source"),
         ]),
    dict(op="uniform", key="uniform-endurance-20261022-attempt-2/", date="2026-10-22",
         title='Optimize random computation, then launch cost', note='Bottleneck shifts from device computation to host launching.',
         stages=[
             ('Measure original random implementation', 'Start / full evaluation', 'Separate compute and state-read costs', "base"),
             ('Vectorize Philox + state reads', 'Evidence: source + targeted probe', 'Also repair numerical bounds and layout', "probe"),
             ('Cache compiled launch entry', 'Evidence: host-time probe', 'Preserve each random-state update', "probe"),
         ]),
    dict(op="negative_", key="negative_-next25-20261029-v2-attempt-2/", date="2026-10-29",
         title='Separate small-input launch and large-input movement', note='Changes target different sizes; retain full results.',
         stages=[
             ('Fix contiguous and strided versions', 'Start / full evaluation', 'Small inputs about 55 us; large about 152 us', "base"),
             ('Directly launch compiled kernel', 'Evidence: empty-kernel launch probe', 'Small inputs: about 31-35 us', "probe"),
             ('64 KiB per program', 'Evidence: tile comparison', 'Large inputs: about 30-41 us', "probe"),
         ]),
    dict(op="selu_", key="selu_-paired-pointwise-20261022-paired-hgkda/", date="2026-10-22",
         title='From call cost to host-path rewrite', note='Historical cache experiments are distinct from current strict deliveries.',
         stages=[
             ('Fix existing tiled implementation', 'Start / paired-profiling background', 'Inspect small and large inputs separately', "profile"),
             ('Prebind arguments + object cache', 'Evidence: host probe + source', 'Small inputs: about 16-18 -> 9-10 us', "probe"),
         ]),
    dict(op="log10_", key=None, date="2026-11-10",
         title='Revise prefetch regression using waits and IR', note='Second change recovers performance; retain original start for this segment.',
         stages=[
             ('Fix generated version', 'Evidence: paired profiling', 'Inspect movement granularity and large-input gap', "profile"),
             ('64 programs + double prefetch', 'Evidence: simulator regions + IR', 'Shifted waits and fragmented movement regress', "sim"),
             ('Use loop-carried buffers', 'Evidence: simulator waits + IR', 'Fewer waits; partial recovery', "sim"),
         ]),
    dict(op="native_group_norm", key="native_group_norm-next25-20261029-v2-attempt-1/", date="2026-10-29",
         title='Two host-overhead experiments', note='Neither improved; retain regressions and analysis.',
         stages=[
             ('Measure fused normalization', 'Start / paired-profiling background', 'Full call far exceeds small-case device time', "profile"),
             ('Combine allocation and launch preparation', 'Evidence: host probe + source', 'Expected lower overhead; no measured gain', "probe"),
             ('Fewer arguments + launch selection', 'Evidence: argument-overhead probe', 'Further regression; simulator role unverified', "probe"),
         ]),
]


def data():
    inventory_path = REPORTS / "trace-expanded-inventory-20261116.json"
    ledgers = read(inventory_path)["ledgers"]
    extract = read(REPORTS / "trace-ledger-extract-20261116.json")
    series = []
    for spec in SPECS:
        s = dict(spec)
        s["reference"] = "CANN" if not s["key"] else 'PyTorch / ATen (batch KGS reference)'
        s["points"] = []
        if s["key"]:
            ledger = next(f for f in ledgers if s["key"] in f["path"] and "/runs/" in f["path"])
            s["ledger"] = {k: ledger[k] for k in ["path", "sha256"]}
            raw_extract = next((f for f in extract if f["ledger_path"] == ledger["path"]), None)
            for r in ledger["rounds"]:
                ev = r["evaluation"]
                point = dict(round=r["round"], score=ev["geo_mean"],
                             passed=ev["num_passed"], total=ev["num_workloads"],
                             source_sha256=r["source_sha256"], recorded_at=ev["evaluated_at"],
                             plan=r["plan"], conclusion=r["conclusion"], profile=r["profile"])
                if raw_extract:
                    raw = next(x for x in raw_extract["rounds"] if x["round"] == r["round"])
                    assert raw["score"] == point["score"]
                    point["result"] = {k: raw[k] for k in ["result_path", "result_sha256"]}
                s["points"].append(point)
        else:
            report = REPORTS / "regions-method-20261110.md"
            s["report"] = {"path": str(report.relative_to(ROOT)), "sha256": sha(report)}
            s["points"].append(dict(round=0, score=0.915250, passed=36, total=36,
                                    source_sha_prefix="45c7380a8062", precision="six decimal places in frozen report"))
            base = ROOT / "development/experiments/skill-validation/20261110-regions/log10-regions"
            for i, prefix in enumerate(["25e8b11f2c2f", "cbb329b76639"], 1):
                path = base / ("formal-" + prefix) / "summarize-final.json"
                row = read(path)["rows"][0]
                s["points"].append(dict(round=i, score=row["cann_score"],
                                        passed=row["evaluation"]["num_passed"], total=36,
                                        source_sha256=row["source_sha256"], result=row["evaluation"],
                                        summary_path=str(path.relative_to(ROOT)), summary_sha256=sha(path)))
        assert len(s["points"]) == len(s["stages"])
        for point, stage in zip(s["points"], s["stages"]):
            point.update(action=stage[0], information_source=stage[1], observation=stage[2], kind=stage[3])
            point["source_label_scope"] = "Recorded information used or experimental background; independent causal contribution not quantified"
        s.pop("stages")
        series.append(s)
    return series


def text(x, y, value, size=28, weight=400, anchor="start", fill=INK):
    return f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(value)}</text>'


def svg(s):
    pts = s["points"]
    n = len(pts)
    xs = [200 + i * 1190 / (n - 1) for i in range(n)]
    top, bottom = 205, 515
    limit = max(p["score"] for p in pts) * 1.3
    y = lambda value: bottom - value / limit * (bottom-top)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">',
             '<rect width="1600" height="900" fill="#fffaf0"/>',
             '<g font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif">',
             text(60, 66, s["op"], 45, 700), text(60, 118, s["title"], 32, 600),
             text(60, 160, f'{s["date"]}  |  relative to {s["reference"]}  |  each round {pts[0]["passed"]}/{pts[0]["total"]}', 25),
             f'<path d="M120 {top} V{bottom} H1490" stroke="{INK}" stroke-width="2.5" fill="none"/>']
    for value in [0, limit / 2, limit]:
        yy = y(value)
        parts += [f'<path d="M120 {yy} H1490" stroke="#d8d6ce" stroke-width="1"/>',
                  text(101, yy+9, f'{value:.2f}×', 24, anchor="end")]
    path = f'M{xs[0]} {y(pts[0]["score"])}'
    for xx, p in zip(xs[1:], pts[1:]):
        path += f'H{xx} V{y(p["score"])}'
    parts.append(f'<path d="{path}" stroke="{GREEN}" stroke-width="6" stroke-linejoin="round" fill="none"/>')
    for i, (xx, p) in enumerate(zip(xs, pts)):
        yy = y(p["score"])
        label_y = yy + 44 if i and p["score"] < pts[i-1]["score"] else yy - 24
        parts += [f'<circle cx="{xx}" cy="{yy}" r="10" fill="{GREEN}" stroke="white" stroke-width="3"/>',
                  text(xx, label_y, f'{p["score"]:.3f}×', 34, 700, "middle"),
                  text(xx, 558, f'R{p["round"]}', 26, 700, "middle")]
        if i:
            prev = pts[i-1]["score"]
            delta = (p["score"] / prev - 1) * 100
            parts.append(text(xx, 599, f'{delta:+.1f}%', 27, 700, "middle", GREEN if delta >= 0 else ORANGE))
    width = 1480/n
    for i, p in enumerate(pts):
        xx = 60 + width*i
        color = ORANGE if p["kind"] == "sim" else BLUE
        parts += [f'<rect x="{xx}" y="638" width="{width-16}" height="145" rx="12" fill="white" stroke="#d9d8d1" stroke-width="1.5"/>',
                  text(xx+16, 674, p["information_source"], 24, 600, fill=color),
                  text(xx+16, 716, p["action"], 27, 700),
                  text(xx+16, 757, p["observation"], 22)]
    parts += [text(60, 840, s["note"], 28, 600),
              text(60, 879, 'Y-axis: reference speedup (higher is better); percentages: speedup change from previous round', 23), '</g></svg>']
    return "\n".join(parts)


def main():
    OUT.mkdir(exist_ok=True)
    series = data()
    (OUT / "traces.json").write_text(json.dumps({"schema": "hgkda.optimization-traces/v1", "series": series}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    for s in series:
        (OUT / (s["op"]+".svg")).write_text(svg(s), encoding="utf-8")
    cards = []
    buttons = []
    for i, s in enumerate(series):
        op = html.escape(s["op"])
        buttons.append(f'<button onclick="show({i})">{op}</button>')
        cards.append(f'<section class="chart" id="c{i}"><img src="{op}.svg" alt="{op} trace"><div class="links"><a href="{op}.svg" download>Download SVG</a><a href="{op}.png" download>Download PNG</a></div></section>')
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>HG-KDA · Six operator traces</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f0e6;color:#17263a;font-family:"Microsoft YaHei",sans-serif}header{padding:28px 36px 18px}h1{font-size:32px;margin:0 0 12px}p{font-size:20px;margin:0 0 18px}nav{display:flex;flex-wrap:wrap;gap:8px}button,a{font:inherit;font-size:18px;color:#17263a}button{padding:10px 16px;border:1px solid #bcc2c6;background:white;border-radius:8px;cursor:pointer}main{padding:0 24px 24px}.chart{max-width:1600px;margin:0 auto 24px;background:white;border-radius:12px;overflow:hidden}.chart img{width:100%;display:block}.links{display:flex;gap:24px;padding:12px 24px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.grid .chart{margin:0}.grid .links{display:none}@media(max-width:1000px){.grid{grid-template-columns:1fr}}@media print{header,.links{display:none}.chart{break-after:page}}</style>
<header><h1>Evidence → Method → Performance</h1><p>First 6 measured segments。Retain gains、regressions and recovery；Each segment has its own reference and contract。</p><nav><button onclick="show(-1)">Six-chart overview</button>BUTTONS</nav></header>
<main class="grid">CARDS</main><script>function show(n){document.querySelector('main').classList.toggle('grid',n<0);document.querySelectorAll('.chart').forEach((c,i)=>c.style.display=n<0||i===n?'':'none');window.scrollTo(0,0)}</script></html>""".replace('BUTTONS', ''.join(buttons)).replace('CARDS',''.join(cards))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    md = ['# First six measured operator traces', "", 'Show methods, analysis, and outcomes; independent causal attribution or a historical-best score is not a plotting prerequisite.', "",
          'Open [index.html](index.html) for overview or individual charts with SVG/PNG. traces.json retains ledgers, source identity, plans, and raw evaluation data.', "",
          "Each chart is one continuous experiment, not an operator's complete history. Percentages compare consecutive speedups. Historical candidates are not automatically current strict deliveries.", "",
          '| Operator | Segment scores | Chart |', "| --- | --- | --- |"]
    for s in series:
        scores = " → ".join(f'{p["score"]:.3f}×' for p in s["points"])
        md.append(f'| `{s["op"]}` | {scores} | [SVG]({s["op"]}.svg)／[PNG]({s["op"]}.png) |')
    md += ["", 'Analysis sources summarize frozen plans/conclusions and region reports; historical model hypotheses are not automatically established causes.',
           'SELU paired profiling is batch background; host probes directly guided these changes. New capture failures do not remove prior measured nodes.',
           'Partial native_group_norm simulation exists; its contribution to these two changes remains unverified.',
           'log10 combines paired metrics, simulation, and IR; double-prefetch failure and recovery do not establish net improvement.', ""]
    (OUT / "README.md").write_text("\n".join(md),encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
