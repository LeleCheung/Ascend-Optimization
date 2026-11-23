"""Draw the evidence-driven optimization routes as an offline SVG and PNG."""
from html import escape
from pathlib import Path


def drawing(ir_cases=0):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1700" height="970" viewBox="0 0 1700 970">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#596170"/></marker></defs>',
        '<rect width="1700" height="970" fill="#fff7e8"/>',
        '<g font-family="Microsoft YaHei, sans-serif" fill="#1b2434">']

    def label(value, x, y, size=26, color='#1b2434', weight=400):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{escape(value)}</text>')

    def box(x, y, w, h, title, detail, color='#2368b1', status=None):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="#fffaf0" stroke="{color}" stroke-width="2"/>')
        label(title, x + 22, y + 41, 30, color, 600)
        label(detail, x + 22, y + 80, 23)
        if status:
            label(status, x + 22, y + 115, 22, color)

    def arrow(x1, y1, x2, y2):
        parts.append(f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="#596170" stroke-width="3" marker-end="url(#arrow)"/>')

    label('持续模拟器重放：依据热点与等待选择修改层', 55, 65, 42, weight=600)
    status = (f'四项源码层案例已验证；另有 {ir_cases} 项固定源码的 IR 收益完成验收'
              if ir_cases else '当前四项已有完整验证的收益来自 Triton 源码；后续按具体问题进入下一层')
    label(status, 58, 110, 27, '#596170')
    box(60, 150, 480, 110, '成对真机 Profiling', '定位 kernel、流水与输入差距')
    box(610, 150, 480, 110, '继续模拟器重放', '当前二进制：热点、流水与等待', '#c55a18')
    box(1160, 150, 480, 110, '关联源码与实际 IR', '绑定源码、case、编译配置与产物')
    arrow(850, 267, 850, 300)
    label('机制假设 → 判断表达能力与编译结果 → 选择路线', 425, 337, 29, weight=600)

    label('编译优化线', 65, 395, 31, '#168358', 600)
    label('分块、布局、访存、融合、同步；依据缺陷选择具体修改层', 305, 395, 25, '#596170')
    box(60, 420, 480, 135, 'Triton 源码', '已有语言和编译链可表达的改法', '#168358', '当前：四项完整验证案例')
    box(610, 420, 480, 135, '单算子 IR', '固定源码，修改后编译并继续重放', '#c55a18',
        f'当前：{ir_cases} 项完整验收IR收益' if ir_cases else '下一步：注入验证与局部实验')
    box(1160, 420, 480, 135, '通用编译 pass', '多个算子反复出现共同缺陷', '#596170', '后续：验证适用范围与回归')
    arrow(548, 485, 600, 485)
    arrow(1098, 485, 1150, 485)
    label('各层共用重放闭环：新产物 → 重放热点／等待 → 真机验收 → 保留最佳 → 下一轮', 100, 588, 24, '#c55a18')

    label('能力扩展线', 65, 620, 31, '#2368b1', 600)
    label('先确认现有 Triton 及编译 pass 的表达边界，再选择扩展入口', 305, 620, 25, '#596170')
    box(60, 645, 480, 135, 'Triton kernel', '定位具体缺失的硬件操作', '#2368b1', '入口：先核对现有 builtin／pass')
    box(610, 645, 480, 135, 'custom op', '建立独立、可追溯的调用边界', '#596170', '条件：确认现有表达能力缺口')
    box(1160, 645, 480, 135, 'Ascend C', '实现底层指令或特殊流水能力', '#596170', '结果：单列能力扩展研究')
    arrow(548, 710, 600, 710)
    arrow(1098, 710, 1150, 710)

    arrow(850, 788, 850, 826)
    parts.append('<rect x="60" y="840" width="1580" height="98" rx="16" fill="#e8f2e8" stroke="#168358" stroke-width="2"/>')
    label('两条路线共同验证', 85, 881, 29, '#168358', 600)
    label('修改后重放 + 完整正确性／专项检查 + 真机性能与 Profiling → 保留最佳', 390, 881, 25)
    label('按日志保留重放与评测顺序、窗口及哈希；结果回到上方重放入口，能力扩展成果单列', 85, 918, 23, '#596170')
    parts.append('<path d="M1640 889H1675V130H850V142" fill="none" stroke="#c55a18" stroke-width="3" marker-end="url(#arrow)"/>')
    parts.append('</g></svg>')
    return ''.join(parts)


def write_routes(output: Path, ir_cases=0):
    output.mkdir(parents=True, exist_ok=True)
    svg = output / 'research-routes.svg'
    svg.write_text(drawing(ir_cases), encoding='utf8')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=dict(width=1700, height=970), device_scale_factor=1)
        page.goto(svg.resolve().as_uri())
        page.locator('svg').screenshot(path=str(output / 'research-routes.png'))
        browser.close()
    return svg


if __name__ == '__main__':
    print(write_routes(Path.cwd() / 'development/reports/group-meeting-20261119'))
