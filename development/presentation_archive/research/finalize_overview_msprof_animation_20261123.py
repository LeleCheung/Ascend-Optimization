"""Check native click animations and publish a three-state visual preview."""

from base64 import b64encode
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "development/reports/overview-msprof-animation-20261123"
WORK = Path("E:/0_XSAI/00_KernelAgent/0_HGKDA_docs/BAAI-reports/06_work")
DELIVERY = WORK / "material/01_开场与总览"
STEM = "HG-KDA-overview-msprof-覆盖动画"
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}

with ZipFile(DELIVERY / f"{STEM}.pptx") as archive:
    slide = etree.fromstring(archive.read("ppt/slides/slide1.xml"))
    timing = slide.find("p:timing", NS)
    (REPORT / "animation-timing.xml").write_bytes(
        etree.tostring(timing, pretty_print=True, encoding="utf-8", xml_declaration=True)
    )
    effects = slide.xpath('.//p:cTn[@nodeType="clickEffect"]', namespaces=NS)
    checks = {
        "click_effects": len(effects),
        "effects": [
            {
                "preset_id": effect.get("presetID"),
                "preset_class": effect.get("presetClass"),
                "targets": effect.xpath(".//p:spTgt/@spid", namespaces=NS),
                "visibility_values": effect.xpath(".//p:strVal/@val", namespaces=NS),
            }
            for effect in effects
        ],
    }
    assert len(effects) == 2, checks
    assert all(effect.get("presetID") == "1" and effect.get("presetClass") == "entr" for effect in effects), checks
    assert [effect.xpath(".//p:spTgt/@spid", namespaces=NS)[0] for effect in effects] == ["35", "36"], checks
    slide_files = [name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
    assert len(slide_files) == 1, slide_files
    checks["slides"] = len(slide_files)

validation = json.loads((REPORT / "validation.json").read_text(encoding="utf-8-sig"))
checks["main_unchanged"] = hashlib.sha256((WORK / "智源20261124汇报.pptx").read_bytes()).hexdigest().upper() == validation["source_sha256_before"]
assert checks["main_unchanged"]
checks["pass"] = True
(REPORT / "animation-xml-validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")

images = ["data:image/png;base64," + b64encode((REPORT / f"state-{i}.png").read_bytes()).decode("ascii") for i in range(3)]
html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HG-KDA / msprof 覆盖动画预览</title>
<style>
body{margin:0;padding:22px;background:#eef2f6;color:#133f6d;font:16px "Microsoft YaHei",sans-serif}
main{max-width:1400px;margin:auto}nav{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap}
button{font:inherit;padding:10px 18px;border:1px solid #cad5e2;border-radius:7px;background:white;color:#133f6d;cursor:pointer}
button.active{background:#133f6d;color:white}img{display:block;width:100%;box-shadow:0 2px 10px #0001;cursor:pointer}
p{margin:14px 0 0;line-height:1.7}a{color:#133f6d}
</style><main><nav>
<button onclick="show(0)">进入本页</button>
<button onclick="show(1)">第一次点击 / msprof op</button>
<button onclick="show(2)">第二次点击 / msprof op simulator</button>
</nav><img id="slide" alt="总图动画状态" onclick="show(Math.min(2,state+1))">
<p>点击图片或按空格、右箭头，逐步预览。原生 PowerPoint 使用两次单击“出现”动画。</p>
<p><a href="HG-KDA-overview-msprof-覆盖动画.pptx">下载动画 PPT</a> / 主汇报文件保持原样。</p>
</main><script>
const images=__IMAGES__;let state=0;
function show(n){state=n;document.getElementById('slide').src=images[n];document.querySelectorAll('button').forEach((b,i)=>b.classList.toggle('active',i===n))}
document.addEventListener('keydown',e=>{if([' ','ArrowRight','ArrowLeft'].includes(e.key)){e.preventDefault();show(Math.max(0,Math.min(2,state+(e.key==='ArrowLeft'?-1:1))))}});show(0);
</script></html>""".replace("__IMAGES__", json.dumps(images))
(DELIVERY / "HG-KDA-overview-msprof-动画预览.html").write_text(html, encoding="utf-8")
(DELIVERY / "HG-KDA-overview-msprof-动画说明.md").write_text(
    "# 总图 / msprof 覆盖动画\n\n"
    "从主汇报第 4 页提取，保留当前排版、主题与所有原始图形。主汇报文件未改动。\n\n"
    "1. 进入本页：显示原总图。\n"
    "2. 第一次点击：`msprof op` 原位覆盖“成对 profiling”。\n"
    "3. 第二次点击：`msprof op simulator` 原位覆盖“模拟器分析”。\n\n"
    "两项均为 PowerPoint 原生“出现”入场，无移动、无淡入。文字与图形均可编辑。\n\n"
    "合入主汇报时复制整张幻灯片并选择“保留源格式”，替换原第 4 页，动画随页面保留。"
    "在普通编辑视图中会看到两个覆盖框；放映时按上述顺序出现。\n\n"
    "已检查：两项原生点击动画及目标、初始画面与主汇报一致、PowerPoint 原生导出无文字溢出、主汇报文件未变。"
    "HTML 用三个静态状态预览覆盖顺序。\n",
    encoding="utf-8",
)
index = WORK / "material/README.md"
text = index.read_text(encoding="utf-8-sig")
heading = "## 总图 / msprof 覆盖动画（2026-11-23）"
if heading not in text:
    addition = (
        heading + "\n\n"
        "[单页动画PPT](01_开场与总览/HG-KDA-overview-msprof-覆盖动画.pptx) / "
        "[三状态交互预览](01_开场与总览/HG-KDA-overview-msprof-动画预览.html) / "
        "[动画说明](01_开场与总览/HG-KDA-overview-msprof-动画说明.md)\n\n"
        "直接提取用户当前主汇报第4页；第一次单击用 `msprof op` 覆盖“成对 profiling”，"
        "第二次单击用 `msprof op simulator` 覆盖“模拟器分析”。保留现有排版，主汇报文件未修改。\n\n"
    )
    text = text.replace("## 最新三页定稿（2026-11-23）", addition + "## 最新三页定稿（2026-11-23）", 1)
    text = text.replace("主汇报原件尚未合入本次三页。", "用户已将三页融入主汇报，当前排版以主汇报为准。")
    index.write_text(text, encoding="utf-8")
print(json.dumps(checks, ensure_ascii=False, indent=2))
