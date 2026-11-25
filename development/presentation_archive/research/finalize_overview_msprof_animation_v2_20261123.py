"""Validate and publish the navy cross-fade revision of the current overview."""
from base64 import b64encode
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from lxml import etree

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "development/reports/overview-msprof-animation-v2-20261123"
WORK = Path("E:/0_XSAI/00_KernelAgent/0_HGKDA_docs/BAAI-reports/06_work")
DELIVERY = WORK / "material/01_开场与总览"
STEM = "HG-KDA-overview-msprof-深蓝淡入版"
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}

with ZipFile(DELIVERY / f"{STEM}.pptx") as archive:
    slide = etree.fromstring(archive.read("ppt/slides/slide1.xml"))
    timing = slide.find("p:timing", NS)
    (REPORT / "animation-timing.xml").write_bytes(etree.tostring(timing, pretty_print=True, encoding="utf-8", xml_declaration=True))
    effects = slide.xpath('.//p:cTn[@presetClass]', namespaces=NS)
    checks = [{"preset_id": effect.get("presetID"), "class": effect.get("presetClass"), "trigger": effect.get("nodeType"), "targets": effect.xpath(".//p:spTgt/@spid", namespaces=NS)} for effect in effects]
    assert len(effects) == 4, checks
    assert all(effect.get("presetID") == "10" for effect in effects), checks
    assert [effect.get("presetClass") for effect in effects] == ["exit", "entr", "exit", "entr"], checks
    assert [effect.get("nodeType") for effect in effects] == ["clickEffect", "withEffect", "clickEffect", "withEffect"], checks
    assert [effect.xpath(".//p:spTgt/@spid", namespaces=NS)[0] for effect in effects] == ["22", "35", "24", "36"], checks
    for shape_id in ("35", "36"):
        shape = slide.xpath(f'.//p:sp[p:nvSpPr/p:cNvPr[@id="{shape_id}"]]', namespaces=NS)[0]
        assert shape.xpath('./p:spPr/a:solidFill/a:srgbClr/@val', namespaces=NS) == ["133F6D"]
        assert shape.xpath('.//a:rPr/a:solidFill/a:srgbClr/@val', namespaces=NS) == ["FFFFFF"]
        size = shape.find("p:spPr/a:xfrm/a:ext", NS)
        assert int(size.get("cx")) == 236 * 12700 and int(size.get("cy")) == 72 * 12700

validation = json.loads((REPORT / "validation.json").read_text(encoding="utf-8-sig"))
assert hashlib.sha256((WORK / "智源20261124汇报.pptx").read_bytes()).hexdigest().upper() == validation["source_sha256_before"]
result = {"clicks": 2, "effects": checks, "navy_white_larger_boxes": True, "main_unchanged": True, "pass": True}
(REPORT / "animation-xml-validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

images = ["data:image/png;base64," + b64encode((REPORT / f"state-{i}.png").read_bytes()).decode("ascii") for i in range(3)]
html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HG-KDA / 深蓝淡入预览</title><style>
body{margin:0;padding:22px;background:#eef2f6;color:#133f6d;font:16px 'Microsoft YaHei',sans-serif}main{max-width:1400px;margin:auto}
nav{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap}button{font:inherit;padding:10px 18px;border:1px solid #cad5e2;border-radius:7px;background:white;color:#133f6d;cursor:pointer}button.active{background:#133f6d;color:white}
#stage{position:relative;cursor:pointer;box-shadow:0 2px 10px #0001}#stage img{display:block;width:100%;pointer-events:none;transition:opacity .45s ease}#stage img+img{position:absolute;top:0;left:0;opacity:0}p{line-height:1.7}a{color:#133f6d}
</style><main><nav><button onclick="show(0)">进入本页</button><button onclick="show(1)">第一次点击 / msprof op</button><button onclick="show(2)">第二次点击 / msprof op simulator</button></nav>
<div id="stage" onclick="show(Math.min(2,state+1))"></div><p>点击图片或按空格、右箭头逐步预览。原标签淡出，更大的深蓝白字框同时淡入，每次0.45秒。</p>
<p><a href="HG-KDA-overview-msprof-深蓝淡入版.pptx">打开原生动画 PPT</a> / 主汇报文件保持原样。</p></main><script>
const images=__IMAGES__;let state=0;
images.forEach((src,i)=>{const img=document.createElement('img');img.src=src;img.alt=['原总图','msprof op 深蓝覆盖框','msprof op simulator 深蓝覆盖框'][i];document.getElementById('stage').appendChild(img)});
function show(n){state=n;document.querySelectorAll('#stage img').forEach((img,i)=>img.style.opacity=i<=n?1:0);document.querySelectorAll('button').forEach((b,i)=>b.classList.toggle('active',i===n))}
document.addEventListener('keydown',e=>{if([' ','ArrowRight','ArrowLeft'].includes(e.key)){e.preventDefault();show(Math.max(0,Math.min(2,state+(e.key==='ArrowLeft'?-1:1))))}});show(0);
</script></html>""".replace("__IMAGES__", json.dumps(images))
(DELIVERY / "HG-KDA-overview-msprof-深蓝淡入预览.html").write_text(html, encoding="utf-8")

index = WORK / "material/README.md"
text = index.read_text(encoding="utf-8-sig")
heading = "## 总图 / msprof 深蓝淡入动画（2026-11-23 修订）"
addition = heading + "\n\n[新版动画PPT](01_开场与总览/HG-KDA-overview-msprof-深蓝淡入版.pptx) / [交互预览](01_开场与总览/HG-KDA-overview-msprof-深蓝淡入预览.html)\n\n每次点击，原标签淡出，同时更大的深蓝白字框淡入；第一次显示 `msprof op`，第二次显示 `msprof op simulator`，过渡均为0.45秒。覆盖框236×72pt，白字18pt。主汇报未修改。复制整张页面到主汇报并选择“保留源格式”即可保留动画。以下初版覆盖动画保留供对照。\n\n"
if heading not in text:
    text = text.replace("## 总图 / msprof 覆盖动画（2026-11-23）", addition + "## 总图 / msprof 覆盖动画（2026-11-23 初版）", 1)
    index.write_text(text, encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
