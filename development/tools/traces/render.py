"""Render data-selected optimization lines and their source-linked history."""
import html
import json
import os
from pathlib import Path

from .importer import read
from .selection import remeasurement_indices

INK, GREEN, BG = "#1b2434", "#168358", "#fff7e8"
COLORS = dict(paired="#2368b1", sim="#c55a18", knowledge="#7852a6", probe="#2368b1", ir="#596170", unknown="#596170")
EVIDENCE_LABELS = dict(paired="Profiling", sim="模拟器", knowledge="知识库", probe="Profiling", ir="代码分析", unknown="待核验")
LOCATION_LABELS = dict(device="kernel 内", host="宿主启动", mixed="混合", ir="IR层")


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
             '<desc id="desc">真实正式评测节点以直线连接，方法与信息来源通过箭头指向相应版本；完整修改与失败记录见HTML。</desc>',
             '<defs>']
    for key, color in COLORS.items():
        parts.append(f'<marker id="arrow-{key}" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 1 L11 6 L0 11 Z" fill="{color}"/></marker>')
    parts += ['</defs>', f'<rect width="1600" height="900" fill="{BG}"/>',
              '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">',
              text(60, 66, view["title"], min(46, 1480 / max(1, sum(1 if ord(c)>127 else .57 for c in view['title']))))]
    delivery = view.get('delivery')
    if delivery:
        parts += [text(62, 137, f"已交付  {delivery['score']:.3f}× CANN", 32, GREEN),
                  text(62, 185, f"完整评测 {delivery['passed']}/{delivery['total']} · 性能验收线 0.8×", 25)]
        if view.get('result_only'):
            parts += [text(800, 410, f"{delivery['score']:.3f}×", 90, GREEN, anchor='middle'),
                      text(800, 490, '当前交付版本 · 相对 CANN', 36, anchor='middle'),
                      text(800, 620, '优化过程尚未恢复，暂展示验收结果', 32, anchor='middle'),
                      '</g>', '</svg>']
            return '\n'.join(parts)
        parts.append(text(62, 239, view.get('line_status', '下图：历史优化片段'), 25))
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
        parts.append(text(62, 307, "加速比", 32))
    else:
        parts.append(text(62, 307, "加速比", 32))
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
        score_label = f"{point['score']:.6f}×" if point.get('score') is not None else "无有效分数"
        parts.append(f'<g class="performance-node" data-evaluation="{point["id"]}" data-location="{location}"><title>{html.escape(point["method"])} | {score_label} | {str(point.get("source_sha256") or "未知")[:12]}</title>')
        parts.append(node_shape(x, yy, location))
        # Labels on alternating sides keep the shallow rises and regressions visible.
        label_y = yy + (43 if n % 2 == 0 and not detail else -23)
        if options.get("label_side") == "above":
            label_y = yy - 25
        if detail and n > 1:
            label_y = yy + 38
        parts.append(text(x, label_y, f"{point['score']:.3f}×" if point.get('score') is not None else "评测失败", 34 if detail else 30, anchor="middle"))
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
        parts.append(text(62, 608, f"{case['dtype']} {case['shape']} · 流水等待取自模拟器单核窗口", 32))
        rows = [("真机任务耗时", lambda o: f"{o['device_task_us']:.2f} μs"),
                ("MTE2 仅等待", lambda o: f"{o['pipelines']['MTE2']['wait_only_ratio']:.2%}"),
                ("Vector 仅等待", lambda o: f"{o['pipelines']['VECTOR']['wait_only_ratio']:.2%}")]
        for r, (label, getter) in enumerate(rows):
            yy = 660 + 46*r
            parts.append(text(62, yy, label, 32))
            for x, p in zip(xs, points):
                parts.append(text(x, yy, getter(p["observation"]), 34, anchor="middle"))
        parts.append(text(62, 840, "修订后部分恢复，仍低于起点；保留原版本。", 36))
    else:
        parts.append(text(810, 853, "优化步骤" if len(points)>1 else "已保存版本", 30, anchor="middle"))
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
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>log10_ 连续优化 trace</title>', f'<style>{CSS}</style><main>',
             f'<h1>{html.escape(index["operator"])} 连续优化 trace</h1>',
             '<p>连续 trace 保存证据读取、源码变化、预检与正式结果；图是其中一条实验线的性能视图。</p>',
             '<nav>']
    for n, view in enumerate(visible_views):
        parts.append(f'<button class="{"active" if n == 0 else ""}" data-tab="{view["id"]}">{"主线图" if n == 0 else "模拟器放大图"}</button>')
    parts += ['<button data-tab="history">完整历史与连续记录</button>', '</nav>']
    for n, view in enumerate(visible_views):
        svg = draw(index, view)
        (output / (view["id"] + ".svg")).write_text(svg, encoding="utf-8")
        parts += [f'<section id="{view["id"]}" {"hidden" if n else ""}><figure>{svg}</figure>',
                  f'<p class="exports"><a href="{view["id"]}.svg">SVG</a><a href="{view["id"]}.png">PNG</a><a href="#detail-{view["session"]}" class="jump-history">展开本线记录</a></p>']
        if n == 1:
            parts.append('<p class="note">这是另一条冻结生成版区域实验线，不能接到开发线 0.851× 之后。它展示预取退步、诊断与恢复；不宣称超过该实验起点。</p>')
        parts.append('</section>')
    parts += ['<section id="history" hidden><h2>历史入口</h2>',
              f'<p>{link(index["history"]["html"], "打开全部28条正式评测图（7个批次）")}　{link(index_path.relative_to(root).as_posix(), "统一 trace 索引 JSON")}</p>',
              '<p>本页先完成两条连续实验线。其他批次仍在原历史图中保留；没有把28个性能点伪装成一条连续源码继承线。</p>',
              '<p class="note">顺序按模型调用编号和原始工具行号恢复；正式评测保留原UTC时间。缺失时间不补造，外层控制器事件按源码锚点展示，不声称已恢复所有动作的精确交错时刻。</p>']
    evidence_by_id = {e["id"]: e for e in index["evidence"]}
    for session in index["sessions"]:
        imported = read(root / session["imported"]["path"])
        counts = session["counts"]
        points = [p for p in index["evaluations"] if p["session_id"] == session["id"]]
        parts += [f'<h2 id="detail-{session["id"]}">{html.escape(session["title"])}</h2>',
                  '<div class="facts">' + ''.join(f'<span class="badge">{v} {k}</span>' for k, v in
                      [("次模型调用", counts["model_calls"]), ("次源码修改", counts["source_transitions"]),
                       ("份设备预检", counts["preflight_sources"]), ("次正式评测", counts["formal_evaluations"])]) + '</div>',
                  '<p class="links">' + link(session["event_stream"], "连续事件 JSONL") + link(session["imported"], "版本与工具索引 JSON") + link(session["config"], "原始实验配置") + '</p>',
                  '<details open><summary>正式性能节点：方法、版本关系与原件</summary><table><thead><tr><th>节点</th><th>方法／结果</th><th>版本关系</th><th>证据入口</th></tr></thead><tbody>']
        for point in points:
            edge = next((e for e in index["measured_edges"] if e["target"] == point["id"]), None)
            relation = (f"与上个已测版间有 {len(edge['transition_ids'])} 次保存的源码修改" if edge else "本实验首个正式点")
            parts.append(f'<tr><td>{point["number"]}<br><code>{point["source_sha256"][:12]}</code></td>'
                         f'<td>{html.escape(point["method"])}<br><b>{point["score"]:.6f}×</b> · {point["passed"]}/{point["total"]}<br>UTC {point["started_at"]}</td>'
                         f'<td>{relation}<br>{html.escape(str(point["occurrence"]["call_id"] or "已提供的冻结起点"))}</td>'
                         f'<td>{link(point["source"], "源码")} · {link(point["result"], "KGS结果")} · {link(point["request"], "合同请求")}</td></tr>')
        parts += ['</tbody></table></details>', '<details><summary>关键箭头：修改所参考的信息与事后验证分开</summary><ul>']
        for view in index["views"]:
            if view["session"] != session["id"]:
                continue
            for annotation in view["annotations"]:
                parts.append(f'<li><b>节点{annotation["point"]} · {html.escape(annotation["method"])}</b><ul>')
                for eid in annotation["evidence"]:
                    e = evidence_by_id[eid]
                    parts.append(f'<li>{link(e["source"],e["category"])} · <code>{e["stage"]}</code>：{html.escape(e["note"])}</li>')
                parts.append('</ul></li>')
        parts += ['</ul></details>', '<details><summary>逐次模型调用：展开工具读取、修改与失败</summary>']
        for call in imported["calls"]:
            parts += [f'<details><summary>{call["invocation"]} · {call["edits"]} 次修改 · {call["tool_calls"]} 次工具调用 · {call["unsuccessful_tools"]} 次返回错误或无结果</summary>',
                      f'<p><code>{str(call["input_sha256"] or "输入文件未知")[:24]} → {str(call["output_sha256"])[:24]}</code></p>',
                      f'<p>{link(call["prompt"], "模型提示词")} · {link(call["source"], "原始工具日志（含结果）")}</p>',
                      '<div class="events"><table><thead><tr><th>行号</th><th>动作</th><th>对象／描述</th></tr></thead><tbody>']
            for event in imported["events"]:
                if event.get("call_id") != call["id"] or event["kind"] not in ("tool_call", "candidate_change"):
                    continue
                line = event["source"]["line"]
                if event["kind"] == "candidate_change":
                    action = "源码修改"
                    desc = str(event["input_sha256"] or "未知")[:12] + " → " + event["output_sha256"][:12]
                else:
                    action = event["tool"] + (" · 返回错误／无结果" if not event["success"] else "")
                    desc = event.get("path") or event.get("description") or event["category"]
                parts.append(f'<tr><td>{line}</td><td>{html.escape(action)}</td><td>{html.escape(desc)}</td></tr>')
            parts.append('</tbody></table></div></details>')
        parts += ['</details>', '<details><summary>设备预检，包括没有正式分数的失败版本</summary><table><tr><th>源码</th><th>预检状态</th><th>原件</th></tr>']
        for p in index["preflights"]:
            if p["session_id"] == session["id"]:
                parts.append(f'<tr><td><code>{p["source_sha256"][:12]}</code></td><td>{p["status"]}</td><td>{link(p["source"],"源码")} · {link(p["result"],"预检结果")}</td></tr>')
        parts.append('</table></details>')
        if "observation" in points[0]:
            parts.append('<details><summary>模拟器原件与口径限制</summary><ul>')
            for p in points:
                o = p["observation"]
                parts.append(f'<li>节点{p["number"]}：{link(o["paired"],"真机成对指标")} · {link(o["simulator"],"热点区域／流水窗口")}<pre>{html.escape(json.dumps(o["qualification"],ensure_ascii=False,indent=2))}</pre></li>')
            parts.append('</ul><p>各自单核窗口的等待比例不代表整卡关键路径；采集异常、真机与模拟器核数差异等限制保留在原报告。</p></details>')
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
