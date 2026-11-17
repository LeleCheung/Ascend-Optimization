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
         title="启动路径修复后，继续扩大分块", note="先降低固定调用成本，再处理大输入的设备耗时。",
         stages=[
             ("修复原地与布局语义", "起点／源码检查", "建立完整正确的起点", "base"),
             ("复用已编译启动入口", "来源：宿主耗时探测", "小输入从约65降至14 μs", "probe"),
             ("大输入分块增至32768", "来源：分块对比实验", "大输入从约77降至28–31 μs", "probe"),
             ("减少中间精度转换", "来源：源码＋类型对比", "完整评测确认小幅改善", "source"),
         ]),
    dict(op="uniform", key="uniform-endurance-20261022-attempt-2/", date="2026-10-22",
         title="先优化随机数计算，再降低启动开销", note="性能瓶颈随优化迁移：设备计算 → 宿主启动。",
         stages=[
             ("测量原始随机数实现", "起点／完整评测", "区分计算与状态读取成本", "base"),
             ("向量化Philox＋状态读取", "来源：源码＋专项探测", "同时修复数值边界与布局", "probe"),
             ("缓存已编译启动入口", "来源：宿主耗时探测", "保留每次随机状态更新", "probe"),
         ]),
    dict(op="negative_", key="negative_-next25-20261029-v2-attempt-2/", date="2026-10-29",
         title="分开解决小输入启动与大输入搬运", note="两次修改针对不同尺寸区间；保留完整评测结果。",
         stages=[
             ("固定连续与跨步版本", "起点／完整评测", "小输入约55 μs，大输入约152 μs", "base"),
             ("直接启动已编译kernel", "来源：空kernel启动探测", "小输入降至约31–35 μs", "probe"),
             ("扩大到每program 64 KiB", "来源：分块对比实验", "大输入降至约30–41 μs", "probe"),
         ]),
    dict(op="selu_", key="selu_-paired-pointwise-20261022-paired-hgkda/", date="2026-10-22",
         title="从调用开销定位到宿主路径改写", note="该图保留当时的缓存实验，不等同于当前严格交付版本。",
         stages=[
             ("固定已有分块实现", "起点／成对profiling背景", "分别观察小输入与大输入", "profile"),
             ("预绑定参数＋对象缓存", "来源：宿主探测＋源码", "小输入约16–18降至9–10 μs", "probe"),
         ]),
    dict(op="log10_", key=None, date="2026-11-10",
         title="预取退步后，结合等待与IR修订", note="第二次修改恢复性能；本段最终仍保留原起点。",
         stages=[
             ("固定已生成版本", "来源：成对profiling", "检查搬运粒度与大输入差距", "profile"),
             ("64个program＋双块预取", "来源：模拟器区域＋IR", "等待转移、搬运切碎，出现退步", "sim"),
             ("改为循环携带缓冲", "来源：模拟器等待＋IR", "等待减少，性能部分恢复", "sim"),
         ]),
    dict(op="native_group_norm", key="native_group_norm-next25-20261029-v2-attempt-1/", date="2026-10-29",
         title="宿主开销假设下的两次尝试", note="两个提案均未改善；退步节点与分析过程一起保留。",
         stages=[
             ("测量融合归一化版本", "起点／成对profiling背景", "完整调用远高于小case设备耗时", "profile"),
             ("合并分配与启动准备", "来源：宿主探测＋源码", "预期减少开销，实测未改善", "probe"),
             ("减少参数＋选择启动入口", "来源：参数开销探测", "再次退步；模拟器贡献待核验", "probe"),
         ]),
]


def data():
    inventory_path = REPORTS / "trace-expanded-inventory-20261116.json"
    ledgers = read(inventory_path)["ledgers"]
    extract = read(REPORTS / "trace-ledger-extract-20261116.json")
    series = []
    for spec in SPECS:
        s = dict(spec)
        s["reference"] = "CANN" if not s["key"] else "PyTorch／ATen（本批KGS参考）"
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
             text(60, 160, f'{s["date"]}  ·  相对{s["reference"]}  ·  每轮{pts[0]["passed"]}/{pts[0]["total"]}', 25),
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
              text(60, 879, "纵轴：相对参考加速比（越高越好）  ·  百分比：较上一轮加速比的变化", 23), '</g></svg>']
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
        cards.append(f'<section class="chart" id="c{i}"><img src="{op}.svg" alt="{op} 优化轨迹"><div class="links"><a href="{op}.svg" download>下载 SVG</a><a href="{op}.png" download>下载 PNG</a></div></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HG-KDA · 六算子优化轨迹</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f0e6;color:#17263a;font-family:"Microsoft YaHei",sans-serif}header{padding:28px 36px 18px}h1{font-size:32px;margin:0 0 12px}p{font-size:20px;margin:0 0 18px}nav{display:flex;flex-wrap:wrap;gap:8px}button,a{font:inherit;font-size:18px;color:#17263a}button{padding:10px 16px;border:1px solid #bcc2c6;background:white;border-radius:8px;cursor:pointer}main{padding:0 24px 24px}.chart{max-width:1600px;margin:0 auto 24px;background:white;border-radius:12px;overflow:hidden}.chart img{width:100%;display:block}.links{display:flex;gap:24px;padding:12px 24px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.grid .chart{margin:0}.grid .links{display:none}@media(max-width:1000px){.grid{grid-template-columns:1fr}}@media print{header,.links{display:none}.chart{break-after:page}}</style>
<header><h1>分析信息 → 修改方法 → 性能变化</h1><p>首批6段真实实验。保留提升、退步与恢复；每段使用自己的参考与测试合同。</p><nav><button onclick="show(-1)">六图总览</button>BUTTONS</nav></header>
<main class="grid">CARDS</main><script>function show(n){document.querySelector('main').classList.toggle('grid',n<0);document.querySelectorAll('.chart').forEach((c,i)=>c.style.display=n<0||i===n?'':'none');window.scrollTo(0,0)}</script></html>'''.replace('BUTTONS', ''.join(buttons)).replace('CARDS',''.join(cards))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    md = ["# 首批六算子真实优化轨迹", "", "先展示方法、分析信息与结果；不以独立因果贡献或超过历史最佳作为画图门槛。", "",
          "打开[index.html](index.html)查看总览或单张；每张提供SVG、PNG。traces.json保存对应账本、源码身份、修改计划和原始评测信息。", "",
          "每图是一段连续实验，不是该算子从首次优化到当前交付的完整生涯。百分比表示相邻两轮加速比的变化；历史候选不自动等同当前严格交付版本。", "",
          "| 算子 | 本段成绩 | 图 |", "| --- | --- | --- |"]
    for s in series:
        scores = " → ".join(f'{p["score"]:.3f}×' for p in s["points"])
        md.append(f'| `{s["op"]}` | {scores} | [SVG]({s["op"]}.svg)／[PNG]({s["op"]}.png) |')
    md += ["", "分析信息来源依据冻结的plan／conclusion及既有区域报告归纳；未将历史模型的机制判断自动升级为已证明因果。",
           "SELU的成对profiling属于批次背景，本图修改方向直接来自宿主探测；本轮新的采集失败不改变已有测量节点的展示。",
           "native_group_norm有partial模拟器材料，参与当前两次修改的具体贡献待核验。",
           "log10结合成对指标、模拟器及IR；图显示双块预取失败和后续恢复，没有声称该轮净收益。", ""]
    (OUT / "README.md").write_text("\n".join(md),encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
