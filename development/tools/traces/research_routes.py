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

    label('Continuous simulator replay: choose layer from regions and waits', 55, 65, 42, weight=600)
    status = (f'Four source-level cases validated; {ir_cases} fixed-source IR gains also accepted'
              if ir_cases else 'Four validated gains currently use Triton source; choose deeper layers for concrete issues')
    label(status, 58, 110, 27, '#596170')
    box(60, 150, 480, 110, 'Paired hardware profiling', 'Locate kernel, pipeline, and input gaps')
    box(610, 150, 480, 110, 'Continue simulator replay', 'Current binary: regions, pipelines, waits', '#c55a18')
    box(1160, 150, 480, 110, 'Associate source and actual IR', 'Bind source, case, compile configuration, and artifacts')
    arrow(850, 267, 850, 300)
    label('Mechanism -> expressibility/generated code -> route', 425, 337, 29, weight=600)

    label('Compiler optimization', 65, 395, 31, '#168358', 600)
    label('Tiling, layout, movement, fusion, synchronization; choose layer by observed defect', 305, 395, 25, '#596170')
    box(60, 420, 480, 135, 'Triton source', 'Changes expressible by current language/compiler', '#168358', 'Current: four fully validated cases')
    box(610, 420, 480, 135, 'Single-operator IR', 'Fixed source; compile changes and continue replay', '#c55a18',
        f'Current: {ir_cases} fully accepted IR gains' if ir_cases else 'Next: injection validation/local experiments')
    box(1160, 420, 480, 135, 'General compiler pass', 'Recurring common defects across operators', '#596170', 'Later: scope validation/regression')
    arrow(548, 485, 600, 485)
    arrow(1098, 485, 1150, 485)
    label('Shared replay loop: new artifacts -> regions/waits -> hardware acceptance -> retain best -> next round', 100, 588, 24, '#c55a18')

    label('Capability extension', 65, 620, 31, '#2368b1', 600)
    label('Verify current Triton/pass expressibility before choosing extensions', 305, 620, 25, '#596170')
    box(60, 645, 480, 135, 'Triton kernel', 'Identify a missing hardware operation', '#2368b1', 'Entry: inspect existing builtin/pass first')
    box(610, 645, 480, 135, 'custom op', 'Establish an independent traceable boundary', '#596170', 'Condition: confirmed expressibility gap')
    box(1160, 645, 480, 135, 'Ascend C', 'Implement low-level instructions/special pipelines', '#596170', 'Result: separately reported extension research')
    arrow(548, 710, 600, 710)
    arrow(1098, 710, 1150, 710)

    arrow(850, 788, 850, 826)
    parts.append('<rect x="60" y="840" width="1580" height="98" rx="16" fill="#e8f2e8" stroke="#168358" stroke-width="2"/>')
    label('Both routes validate', 85, 881, 29, '#168358', 600)
    label('Post-change replay + full correctness/supplementary checks + hardware performance/profiling -> retain best', 390, 881, 25)
    label('Preserve replay/evaluation order, windows, and hashes; return results to the replay entry and report extensions separately', 85, 918, 23, '#596170')
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
