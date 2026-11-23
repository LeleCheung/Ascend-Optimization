"""Combine the latest comparison matrix with the local Ascend evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

import xlsxwriter

sys.path.insert(0, str(Path(os.environ['TEMP']) / 'ascend-stats-20261120-libs'))
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / '指标比较-1122.xlsx'
STATS = ROOT / '统计数据.xlsx'
OUTPUT = ROOT / '指标比较-综合最终版.xlsx'
REPORT = ROOT / 'development/reports/indicator-comparison-20261121'
EVIDENCE = ROOT / 'development/reports/ascend-statistics-20261120'
PUBLIC_SPECS = ROOT / 'development/reports/ascend-spec-public-20261121'
SPEC_REVIEW = ROOT / 'development/reports/indicator-spec-review-20261122'
MACHINE_REVIEW = ROOT / 'development/reports/machine-indicator-audit-20261121'
MSOPPROF_COMMIT = 'a74e105fcfad06c3faa4be308834b3158e221299'
MSOPPROF_SOURCE = f'https://github.com/Ascend/msopprof/blob/{MSOPPROF_COMMIT}/csrc/op_profiling/profiling/device/data_visualize/'
YES, NO, PART = '√', '×', '部分'
TIMING_LABEL = '微架构层 时序指标\n（模拟器/真机采样）'
METRIC_NAMES = {9: '寄存器数量/大小',
                10: 'L1/SPM 容量/带宽/访问粒度',
                11: 'L2 cache 容量/带宽/cache line size',
                12: 'DRAM 容量/带宽',
                13: '4/8/16/32 bit 矩阵算力',
                14: '4/8/16/32 bit 向量算力',
                17: 'L1/SPM 访存统计',
                18: 'L2 缓存命中率'}
USER_CONCLUSIONS = {(20, 3): PART, (21, 3): PART, (17, 4): YES}


def display_text(value):
    """Use ASCII slashes without adjacent spaces in displayed workbook text."""
    if isinstance(value, str):
        return re.sub(r'[ \t]*[/／][ \t]*', '/', value)
    return value


def expanded(sheet):
    values = {cell.coordinate: cell.value for row in sheet for cell in row}
    for area in sheet.merged_cells.ranges:
        value = sheet.cell(area.min_row, area.min_col).value
        for row in sheet.iter_rows(min_row=area.min_row, max_row=area.max_row,
                                   min_col=area.min_col, max_col=area.max_col):
            for cell in row:
                values[cell.coordinate] = value
    return values


def normalize(value):
    text = str(value or '').strip()
    if text.startswith(('✔', '√')):
        return YES
    if text in ('❌', '×', '没找到'):
        return NO
    if text in ('部分', '半对'):
        return PART
    raise ValueError(f'Unexpected source conclusion: {value!r}')


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE, STATS)}
    source = load_workbook(BASE)
    original = source.worksheets[0]
    values = expanded(original)
    stats = load_workbook(STATS)
    proof = list(stats['昇腾依据'].iter_rows(min_row=2, values_only=True))
    assert len(proof) == 20

    # Preserve the team's 1122 category-level judgments. Device-specific coverage
    # and missing subfields stay explicit in the evidence sheet.
    ascend = {row: USER_CONCLUSIONS.get((row, 4),
              normalize(values[original.cell(row, 4).coordinate])) for row in range(2, 22)}
    matrix = []
    for row in range(2, 22):
        matrix.append([USER_CONCLUSIONS[(row, col)] if (row, col) in USER_CONCLUSIONS else
                       ascend[row] if col == 4 else
                       normalize(values[original.cell(row, col).coordinate])
                       for col in range(3, 10)])

    book = xlsxwriter.Workbook(OUTPUT)
    book.set_properties({'title': '硬件规格与性能分析指标比较',
                         'subject': '2026-11-22 对齐小组1122版', 'author': 'HG-KDA'})
    default = {'font_name': 'Microsoft YaHei', 'font_size': 11,
               'valign': 'vcenter', 'border': 1, 'border_color': '#A9BBB3'}
    header = book.add_format({**default, 'bold': True, 'font_color': '#FFFFFF',
                              'bg_color': '#244B41', 'align': 'center'})
    ascend_header = book.add_format({**default, 'bold': True, 'font_color': '#FFFFFF',
                                     'bg_color': '#176244', 'align': 'center'})
    metric = book.add_format({**default, 'text_wrap': True})
    group_format = book.add_format({**default, 'bold': True, 'text_wrap': True,
                                    'bg_color': '#CBD7D2', 'align': 'center'})
    timing_group_format = book.add_format({**default, 'font_size': 10,
                                           'bold': True, 'text_wrap': True,
                                           'bg_color': '#CBD7D2', 'align': 'center'})
    palettes = {YES: ('#165235', '#C4DECC'), NO: ('#973531', '#F0CBC8'),
                PART: ('#785111', '#EED69A')}
    states = {s: book.add_format({**default, 'align': 'center', 'font_size': 12,
                                  'bold': s != PART, 'font_color': color,
                                  'bg_color': background})
              for s, (color, background) in palettes.items()}
    sheet = book.add_worksheet('指标比较')
    sheet.hide_gridlines(2)
    sheet.set_zoom(90)
    sheet.freeze_panes(1, 2)
    sheet.set_tab_color('#1C7255')
    sheet.set_column('A:A', 18)
    sheet.set_column('B:B', 36)
    sheet.set_column('C:C', 12)
    sheet.set_column('D:D', 14)
    sheet.set_column('E:F', 14)
    sheet.set_column('G:I', 12)
    sheet.set_row(0, 32)
    for col in range(1, 10):
        sheet.write(0, col - 1, display_text(original.cell(1, col).value),
                    ascend_header if col == 4 else header)
    for row, conclusions in zip(range(2, 22), matrix):
        sheet.set_row(row - 1, 27)
        sheet.write(row - 1, 1, display_text(METRIC_NAMES.get(row, original.cell(row, 2).value)), metric)
        for col, state in enumerate(conclusions, 2):
            sheet.write(row - 1, col, state, states[state])
    for area in original.merged_cells.ranges:
        if area.min_col == 1:
            value = (TIMING_LABEL if str(area) == 'A19:A21' else
                     original.cell(area.min_row, area.min_col).value)
            fmt = timing_group_format if str(area) == 'A19:A21' else group_format
        else:
            covered = {matrix[row - 2][col - 3]
                       for row in range(area.min_row, area.max_row + 1)
                       for col in range(area.min_col, area.max_col + 1)}
            assert len(covered) == 1, str(area)
            value = covered.pop()
            fmt = states[value]
        sheet.merge_range(str(area), display_text(value), fmt)
    sheet.set_landscape()
    sheet.set_paper(9)
    sheet.fit_to_pages(1, 1)
    sheet.set_margins(0.25, 0.25, 0.25, 0.25)
    sheet.print_area('A1:I21')
    sheet.center_horizontally()
    sheet.activate()
    sheet.set_first_sheet()

    wrap = book.add_format({**default, 'valign': 'top', 'text_wrap': True})
    link = book.add_format({**default, 'font_color': '#16665A', 'underline': True,
                            'valign': 'top', 'text_wrap': True})
    details = book.add_worksheet('昇腾依据')
    details.hide_gridlines(2)
    details.freeze_panes(1, 3)
    details.set_tab_color('#759789')
    details.set_column('A:A', 7)
    details.set_column('B:B', 33)
    details.set_column('C:C', 9)
    details.set_column('D:D', 60)
    details.set_column('E:E', 65)
    details.set_column('F:F', 65)
    details.set_column('G:G', 38)
    for col, name in enumerate(['序号', '指标', '主表判定', '数据／可用接口',
                                '统计口径', '依据文件', '设备／版本']):
        details.write(0, col, display_text(name), header)
    details.set_row(0, 28)
    records = []
    for index, record in enumerate(proof, 1):
        row, name, old, interface, observed, boundary, files, scope = record
        name = METRIC_NAMES.get(row, name)
        interface = str(interface)
        if row == 7:
            interface = ('msprof提供cube_utilization(%)；msOpProf提供Cube／Vector'
                         '及各dtype周期占比。按工具可提供利用率指标记√。')
        elif row == 9:
            interface = 'msOpProf simulator／MindStudio Insight提供GPR Count；950 SIMD文档列出寄存器数量及大小。'
            observed = ('Atlas A2可查看GPR使用，官方指南说明最大使用数量32；'
                        '950PR/950DT矢量寄存器256 B、上限32，掩码寄存器32 B、上限8。')
            boundary = ('主表沿用小组1122的厂商类别判定√。950 SIMD参数限该系列；'
                        '910B4-1各类寄存器完整物理规格仍有缺项。')
            files = ('official/msopprof-msopprof_simulator_user_guide.md#算子代码热点图; '
                     '../indicator-spec-review-20261122/RegisterTypes.json')
            scope = 'Atlas A2模拟器GPR使用信息；950PR/950DT SIMD规格分别记录。'
        elif row == 10:
            interface = 'L1/SPM容量、通路理论带宽模型与显式访问／搬运粒度。'
            observed = ('910B4-1：L1 Buffer 512 KiB、UB 192 KiB；SDK列出UB block粒度32 B。'
                        '官方msopprof 910B Roofline模型：L1每核读256 B/cycle、写256 B/cycle。')
            boundary = ('SPM按具体指令的访问／搬运粒度记录，32 B为UB block粒度；'
                        'L1带宽按通路、频率和核数换算。Cache架构使用cache line粒度。')
            files = ('device-query.json; ../ascend-spec-public-20261121/msopprof-roofline.h:182; '
                     '../ascend-spec-public-20261121/msopprof-roofline.cpp:234')
            scope = f'Ascend910B4-1 SDK配置；官方msopprof 910B模型；commit {MSOPPROF_COMMIT}。'
        elif row == 11:
            interface = 'L2 Cache容量、官方理论带宽模型与cache line／sector规格。'
            observed = ('910B4-1：L2容量168 MiB；官方默认／SK／SS带宽模型为4 TB/s，CJ分支为6.3 TB/s。'
                        '本轮显式910B4-1 SDK配置查询得到L2带宽参数110 B/cycle。'
                        '950白皮书另列L2最高128 MB、512 B cache line及4×128 B sector。')
            boundary = ('主表沿用小组1122的厂商类别判定√；各型号参数分别记录。'
                        '910B4-1的cache line映射及整卡带宽模型分支仍需匹配。')
            files = ('runtime-properties.json; ../ascend-spec-public-20261121/msopprof-data_visualize_const.h:163; '
                     '../ascend-spec-public-20261121/ascend950-architecture-whitepaper.pdf#page=15; '
                     '../indicator-spec-review-20261122/sdk-bandwidth-api-result-linked.json')
            scope = f'910B4-1实测容量与官方模型；950架构白皮书表3-1、4.3.2；msopprof commit {MSOPPROF_COMMIT}。'
        elif row == 12:
            interface = '官方公开主存容量与理论带宽；板卡／运行时观测另列。'
            observed = ('950PR白皮书：128 GB、1.6 TB/s，另一配置112 GB、1.4 TB/s。'
                        '300I Pro官网：24 GB、204.8 GB/s。'
                        '910B4-1板卡查询HBM Capacity(MB)=65536，运行时可见62420 MB；'
                        '本轮SDK配置查询HBM带宽参数32 B/cycle、容量32 GiB。')
            boundary = ('主表沿用小组1122的厂商类别判定√。各型号独立取值；'
                        '910B4-1 SDK配置与实际板卡容量存在差异，整卡峰值带宽及其范围仍需核实。')
            files = ('../ascend-spec-public-20261121/ascend950-architecture-whitepaper.pdf#page=14; '
                     'device-query.json; runtime-properties.json; '
                     '../indicator-spec-review-20261122/sdk-bandwidth-api-result-linked.json; '
                     '../indicator-spec-review-20261122/Atlas300.json')
            scope = '950PR、300I Pro官方规格；910B4-1板卡、运行时与SDK配置分别记录。'
        elif row == 13:
            interface = '官方公开Cube各精度理论峰值，按数据格式及配置记录。'
            observed = ('950PR较高配置：MXFP4 1730 TFLOPS、INT8 865 TOPS、'
                        'FP16/BF16 432 TFLOPS、TF32 216 TFLOPS。')
            boundary = ('主表沿用小组1122的厂商类别判定√；910B4-1完整精度峰值表仍缺项。'
                        '数值采用950PR较高配置的Cube独立表项，Cube＋Vector总峰值另列。'
                        '32bit相关表项为TF32；TF32输入有效精度与完整IEEE FP32分别理解，'
                        'FP32累加也单独区分。各精度格式逐项保留。')
            files = '../ascend-spec-public-20261121/ascend950-architecture-whitepaper.pdf#page=13'
            scope = '950PR架构白皮书表3-1，PDF第13—14页；较高配置Cube峰值。'
        elif row == 14:
            interface = '官方公开Vector FP16/BF16、FP32及INT8/16/32/64峰值；4bit条目仍缺依据。'
            observed = ('950PR较高配置：FP16/BF16 54 TFLOPS、FP32 27 TFLOPS、'
                        'INT8 54 TOPS、INT16 27 TOPS、INT32 13 TOPS、INT64 6 TOPS。')
            boundary = ('主表沿用小组1122的厂商类别判定√。950表已有部分精度峰值，'
                        '4bit Vector峰值及910B4-1完整精度峰值表仍缺项。')
            files = '../ascend-spec-public-20261121/ascend950-architecture-whitepaper.pdf#page=14'
            scope = '950PR架构白皮书表3-1，PDF第14页；较高配置Vector峰值。'
        elif row == 17:
            interface = 'L1缓存使用命中／未命中统计；软件管理SPM使用搬运、带宽及Bank冲突统计。'
            observed = ('昇腾L1 Buffer与UB分别记录。msOpProf提供L1读写带宽与GM↔UB通路统计；'
                        '官方模拟器热点图提供UB Bank冲突信息。')
            boundary = ('L1 Buffer、UB按软件管理存储的访存口径分析；L2 Cache单独记录容量与命中率。')
            files = ('valid-field-samples.json; official/msopprof-msopprof_performance_data.md; '
                     'official/msopprof-msopprof_simulator_user_guide.md#算子代码热点图')
        elif row == 20:
            interface = '官方模拟器记录指令时序与同步事件；MindStudio Insight展示同步连线及SET_FLAG／WAIT_FLAG等待。'
            observed = ('GN与GQA已有逐流水时序及等待事件。HG-KDA进一步脚本化整理、关联源码／IR与优化前后真机结果。')
            files = ('official/msopprof-msopprof_simulator_user_guide.md#指令流水图; simulator-observations.json')
        elif row == 21:
            interface = '官方逐流水指令时序可观察空闲区间；HG-KDA在已观测窗口内自动量化执行、等待与空闲。'
            observed = '官方工具提供原始时序和流水图；现有脚本输出active、wait_only及observable idle并进行前后比较。'
            files = ('official/msopprof-msopprof_simulator_user_guide.md#指令流水图; '
                     'simulator-observations.json; ../regions-method-20261110.md')
        name, interface, observed, boundary, files, scope = map(
            display_text, (name, interface, observed, boundary, files, scope))
        payload = [index, name, ascend[row], interface + '\n' + str(observed),
                   boundary, files, scope]
        details.set_row(index, 112)
        for col, value in enumerate(payload):
            details.write(index, col, value, states[value] if col == 2 else wrap)
        records.append({'source_row': row, 'metric': name, 'state': ascend[row],
                        'source_1122': normalize(values[original.cell(row, 4).coordinate]),
                        'state_origin': 'team_1122' if row != 17 else 'user_spm_metric_definition',
                        'previous_statistics_state': old, 'interface': interface,
                        'observed': observed, 'scope': boundary, 'files': files,
                        'device_version': scope})
    details.autofilter(0, 0, 20, 6)

    sources = book.add_worksheet('资料来源')
    sources.hide_gridlines(2)
    sources.freeze_panes(1, 0)
    sources.set_column('A:A', 29)
    sources.set_column('B:B', 64)
    sources.set_column('C:C', 75)
    for col, value in enumerate(['资料', '入口', '内容']):
        sources.write(0, col, value, header)
    sources.set_row(0, 28)
    links = [
        ('比较表底稿', BASE, '主表采用小组1122版；保留全部43组合并。仅调整NVIDIA依赖/空泡及L1/SPM访存统计判定。'),
        ('主表与依据的口径', REPORT / 'README.md', '主表沿用小组的厂商能力类别判定；依据页分别标明型号、实际取得的数据和缺项。'),
        ('昇腾统计底稿', STATS, '合入20项昇腾指标的数值、接口与来源。'),
        ('昇腾统计调研', EVIDENCE / 'README.md', '硬件规格、采集能力与各项依据。'),
        ('设备查询', EVIDENCE / 'device-query.json', '板卡信息、HBM及匹配SDK配置。'),
        ('运行时属性', EVIDENCE / 'runtime-properties.json', '实际型号、核数、内存与L2。'),
        ('有效字段样例', EVIDENCE / 'valid-field-samples.json', 'profiling字段与实际观测。'),
        ('模拟器观察', EVIDENCE / 'simulator-observations.json', '指令热点、流水等待与空闲。'),
        ('沐曦原表依据', BASE, '1122版并行度、功能部件流水利用率：mcProfiler官方界面显示有。'),
        ('实际机器与工具复核', MACHINE_REVIEW / 'README.md', '910B4-1、S5000型号与工具记录；保留S5000能力及导出问题的审计依据。'),
        ('昇腾补充规格复核', SPEC_REVIEW / 'README.md', '新增寄存器文档、GetCoreMemBw实查与各型号适用范围。'),
        ('昇腾公开规格复核', PUBLIC_SPECS / 'README.md', '官网、官方白皮书及msopprof公开理论带宽模型。'),
        ('950架构白皮书', PUBLIC_SPECS / 'ascend950-architecture-whitepaper.pdf',
         '表3-1：Cube／Vector各精度峰值、主存容量／带宽；PDF第15及26页：L2 cache line／sector。'),
    ]
    for name, path, description in links:
        row = sources.dim_rowmax + 1
        sources.set_row(row, 42)
        sources.write(row, 0, display_text(name), wrap)
        sources.write_url(row, 1, 'external:' + path.relative_to(ROOT).as_posix(),
                          link, path.name)
        sources.write(row, 2, display_text(description), wrap)
    for item in json.loads((EVIDENCE / 'official-sources.json').read_text(encoding='utf-8')):
        row = sources.dim_rowmax + 1
        sources.set_row(row, 48)
        sources.write(row, 0, '官方文档', wrap)
        sources.write_url(row, 1, item['url'], link, Path(item['local']).name)
        sources.write(row, 2, '固定版本：' + item['commit'], wrap)
    public_sources = [
        ('950 SIMD寄存器定义', 'https://asc.gitcode.com/api/SIMD-API/c_api/defs/type/data_type_definition.html',
         '950PR/950DT寄存器数量、字节大小；适用型号明确。'),
        ('GetCoreMemBw', 'https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1035.html',
         'L2、HBM带宽参数查询，单位byte/cycle；910B4-1配置查询已保存。'),
        ('950处理器官方入口', 'https://www.hiascend.com/hardware/processor',
         '官网提供《昇腾950 NPU架构白皮书》下载；厂商信息可获得性采用公开型号实例。'),
        ('L1／UB理论带宽', MSOPPROF_SOURCE + 'roofline.h#L182',
         '910B L1每核读256 B/cycle、写256 B/cycle；带宽按频率和核数换算。'),
        ('L2／主存理论带宽', MSOPPROF_SOURCE + 'data_visualize_const.h#L139',
         '910B各SoC、内存类型的GM与L2 Roofline理论带宽表。'),
    ]
    for name, url, description in public_sources:
        row = sources.dim_rowmax + 1
        sources.set_row(row, 65)
        sources.write(row, 0, display_text(name), wrap)
        sources.write_url(row, 1, url, link, '昇腾官方公开资料')
        sources.write(row, 2, display_text(description), wrap)
    nvidia_sources = [
        ('NVIDIA指令热点：√',
         'https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-sampling',
         'NCU通过PC采样定位热点SASS指令，并关联源码。'),
        ('NVIDIA流水线依赖：部分',
         'https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#instructions-dependencies-table',
         'NCU提供寄存器／scoreboard依赖和采样等待归因，覆盖部分依赖关系。'),
        ('NVIDIA流水线空泡：部分',
         'https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#pm-sampling',
         'NCU通过发射统计、等待采样及利用率时间线分析空闲，流水内部的详细空泡时序覆盖有限。'),
    ]
    for name, url, description in nvidia_sources:
        row = sources.dim_rowmax + 1
        sources.set_row(row, 65)
        sources.write(row, 0, display_text(name), wrap)
        sources.write_url(row, 1, url, link, 'Nsight Compute官方说明')
        sources.write(row, 2, display_text(description), wrap)
    book.close()

    # Check the delivered workbook itself and its correspondence with both inputs.
    result = load_workbook(OUTPUT)
    delivered = result['指标比较']
    delivered_values = expanded(delivered)
    assert result.sheetnames == ['指标比较', '昇腾依据', '资料来源']
    original_merges = set(map(str, original.merged_cells.ranges))
    assert set(map(str, delivered.merged_cells.ranges)) == original_merges
    assert delivered['A19'].value == TIMING_LABEL
    assert all(delivered.cell(row, 2).value == name for row, name in METRIC_NAMES.items())
    assert all(state == YES for state in ascend.values())
    assert all(delivered_values[delivered.cell(row, col).coordinate] in (YES, NO, PART)
               for row in range(2, 22) for col in range(3, 10))
    for source_row in range(2, 22):
        for col in (3, 5, 6, 7, 8, 9):
            expected = USER_CONCLUSIONS.get((source_row, col),
                       normalize(values[original.cell(source_row, col).coordinate]))
            assert delivered_values[delivered.cell(source_row, col).coordinate] == expected
        assert delivered_values[delivered.cell(source_row, 4).coordinate] == ascend[source_row]
        assert result['昇腾依据'].cell(source_row, 3).value == ascend[source_row]
    assert all(cell.comment is None for ws in result for row in ws for cell in row)
    assert all('／' not in cell.value and not re.search(r'[ \t]/|/[ \t]', cell.value)
               for row in delivered for cell in row if isinstance(cell.value, str))
    assert result['昇腾依据'].max_row == 21
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE, STATS)}
    differences = []
    for row in range(2, 22):
        for col in range(3, 10):
            cell = delivered.cell(row, col).coordinate
            previous = normalize(values[original.cell(row, col).coordinate])
            current = delivered_values[cell]
            if previous != current:
                differences.append({'cell': cell, 'metric': delivered.cell(row, 2).value,
                                    'source_1122': previous, 'delivered': current})
    assert {item['cell'] for item in differences} == {'C20', 'C21', 'D17'}
    validation = {'status': 'passed', 'output': OUTPUT.name, 'metrics': 20,
                  'vendors': 7, 'main_conclusion_cells': 140,
                  'allowed_states': [YES, NO, PART], 'ascend_evidence_rows': 20,
                  'original_merged_ranges_preserved': True,
                  'source_workbook': BASE.name,
                  'user_added_merged_ranges': [],
                  'merged_range_count': len(delivered.merged_cells.ranges),
                  'visible_conclusion_cells': sum(cell.value in (YES, NO, PART)
                       for row in delivered.iter_rows(min_row=2, max_row=21, min_col=3, max_col=9) for cell in row),
                  'other_vendor_conclusions_match_1122_except_user_changes': True,
                  'five_domestic_vendor_conclusions_match_1122': True,
                  'five_domestic_vendor_metrics_checked': 100,
                  'conclusion_differences_from_1122': differences,
                  'display_slash': 'ASCII /, no adjacent spaces',
                  'user_conclusion_changes': {'C20': PART, 'C21': PART, 'D17': YES},
                  'user_metric_names': METRIC_NAMES,
                  'timing_label': TIMING_LABEL,
                  'input_files_preserved': True, 'cell_comments': 0,
                  'ascend_counts': {s: list(ascend.values()).count(s) for s in (YES, NO, PART)}}
    (REPORT / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    (REPORT / 'ascend-conclusions.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    (REPORT / 'README.md').write_text(
        '# 指标比较综合最终版\n\n'
        '2026-11-22。按用户最新要求，以小组《指标比较-1122.xlsx》为主表底稿，'
        '统一汇报口径；根目录`指标比较-综合最终版.xlsx`与本目录`指标比较-主表.pdf`同步。\n\n'
        '## 与小组1122版的区别\n\n'
        '- NVIDIA流水线依赖、流水线空泡：×改为部分，共2项；指令热点保持√。\n'
        '- L1硬件规格名称使用“L1/SPM 容量/带宽/访问粒度”；'
        '统计名称使用“L1/SPM 访存统计”，昇腾D17按搬运、带宽及Bank冲突信息记√。\n'
        '- 时序层标题使用“微架构层 时序指标（模拟器/真机采样）”。\n'
        '- 其余137项判定采用1122底稿，符号统一为√、×、部分；'
        '摩尔线程及其他4家国产厂商共100项判定逐项保持。\n\n'
        '## 表格与具体型号依据\n\n'
        '主表沿用小组对厂商能力类别及公开信息可获得性的判断，昇腾硬件区D9:D15整体√。'
        '具体型号的覆盖范围及缺项写入“昇腾依据”页；'
        '主表√表达该类别存在可获得信息，具体型号的全部子规格需继续按依据页核实。'
        '950寄存器与峰值参数、300I Pro规格和910B4-1实测/SDK参数分别记录。'
        '同事原始Excel保持原件；前版交付与生成器归档于`archive-before-1122-alignment-20261122/`。\n\n'
        '910B优化已经使用已知的L1/UB/L0容量、核数及分块约束，'
        '结合profiling、模拟器指令时序与源码/IR关联形成改法，再以正确性和真机耗时验证。'
        '完整各精度峰值表的缺项主要影响理论上限及利用率分母的核算；'
        '已取得的实际执行证据继续支撑现有优化结论。\n\n'
        '## 格式与检查\n\n'
        '五层、20项指标、7家厂商；保留1122原有43组合并，昇腾系统层D4:D8、'
        '硬件区D9:D15及三项时序D19:D21保持整块。深色底与字体沿用，'
        '半角斜杠/前后无空格，主表无批注。交付检查展开全部合并单元格，'
        '确认相对1122仅C20、C21、D17三项判定变化，并验证输入原件未改。\n', encoding='utf-8')
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == '__main__':
    main()
