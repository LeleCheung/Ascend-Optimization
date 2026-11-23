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
BASE = ROOT / '指标比较-1121.xlsx'
STATS = ROOT / '统计数据.xlsx'
OUTPUT = ROOT / '指标比较-综合最终版.xlsx'
REPORT = ROOT / 'development/reports/indicator-comparison-20261121'
EVIDENCE = ROOT / 'development/reports/ascend-statistics-20261120'
PUBLIC_SPECS = ROOT / 'development/reports/ascend-spec-public-20261121'
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
USER_CONCLUSIONS = {(20, 3): PART, (21, 3): PART}


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

    # Capability availability is the comparison criterion for profiler rows.
    # Hardware information availability includes the reviewed public model examples.
    ascend = {row: YES for row in range(2, 22)}
    ascend.update({9: PART, 14: PART})
    matrix = []
    for row in range(2, 22):
        matrix.append([USER_CONCLUSIONS[(row, col)] if (row, col) in USER_CONCLUSIONS else
                       ascend[row] if col == 4 else
                       normalize(values[original.cell(row, col).coordinate])
                       for col in range(3, 10)])

    book = xlsxwriter.Workbook(OUTPUT)
    book.set_properties({'title': '硬件规格与性能分析指标比较',
                         'subject': '2026-11-21 综合最终版', 'author': 'HG-KDA'})
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
    sheet.merge_range('D4:D8', YES, states[YES])
    sheet.merge_range('D10:D13', YES, states[YES])
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
    for col, name in enumerate(['序号', '指标', '结论', '数据／可用接口',
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
            interface = 'msOpProf simulator／MindStudio Insight提供GPR Count与寄存器使用情况。'
            observed = ('官方模拟器用户指南说明通用寄存器最大使用数量为32，并列出Atlas A2支持GPR Count。')
            boundary = ('已取得工具所述通用寄存器数量与使用信息；位宽、各类寄存器的完整物理数量与容量仍需型号匹配依据。')
            files = 'official/msopprof-msopprof_simulator_user_guide.md#算子代码热点图'
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
                        '950白皮书另列L2最高128 MB、512 B cache line及4×128 B sector。')
            boundary = ('本行判断厂商公开信息的可获得性；容量、带宽与粒度逐项标明型号。'
                        '950的cache line规格应用于950；910B带宽值按具体内存类型选择。')
            files = ('runtime-properties.json; ../ascend-spec-public-20261121/msopprof-data_visualize_const.h:163; '
                     '../ascend-spec-public-20261121/ascend950-architecture-whitepaper.pdf#page=15')
            scope = f'910B4-1实测容量与官方模型；950架构白皮书表3-1、4.3.2；msopprof commit {MSOPPROF_COMMIT}。'
        elif row == 12:
            interface = '官方公开主存容量与理论带宽；板卡／运行时观测另列。'
            observed = ('950PR白皮书：128 GB、1.6 TB/s，另一配置112 GB、1.4 TB/s。'
                        '910B4-1板卡查询HBM Capacity(MB)=65536，运行时可见62420 MB。')
            boundary = ('950PR参数来自官方规格；910B4-1容量保持设备查询原单位。'
                        '各型号独立取值，型号已确认后使用对应规格。')
            files = ('../ascend-spec-public-20261121/ascend950-architecture-whitepaper.pdf#page=14; '
                     'device-query.json; runtime-properties.json')
            scope = '950PR架构白皮书表3-1；910B4-1容量查询。厂商公开信息范围。'
        elif row == 13:
            interface = '官方公开Cube各精度理论峰值，按数据格式及配置记录。'
            observed = ('950PR较高配置：MXFP4 1730 TFLOPS、INT8 865 TOPS、'
                        'FP16/BF16 432 TFLOPS、TF32 216 TFLOPS。')
            boundary = ('数值采用950PR较高配置的Cube独立表项，Cube＋Vector总峰值另列。'
                        '32bit相关表项为TF32；TF32输入有效精度与完整IEEE FP32分别理解，'
                        'FP32累加也单独区分。各精度格式逐项保留。')
            files = '../ascend-spec-public-20261121/ascend950-architecture-whitepaper.pdf#page=13'
            scope = '950PR架构白皮书表3-1，PDF第13—14页；较高配置Cube峰值。'
        elif row == 14:
            interface = '官方公开Vector FP16/BF16、FP32及INT8/16/32/64峰值；4bit条目仍缺依据。'
            observed = ('950PR较高配置：FP16/BF16 54 TFLOPS、FP32 27 TFLOPS、'
                        'INT8 54 TOPS、INT16 27 TOPS、INT32 13 TOPS、INT64 6 TOPS。')
            boundary = '本行8／16／32bit已有峰值，官方表尚未列4bit Vector峰值，故保留部分。'
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
                        'previous': old, 'interface': interface,
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
        ('比较表底稿', BASE, '保留1121版原有合并关系；按用户更新NVIDIA时序判定，并合并昇腾D4:D8与D10:D13。'),
        ('昇腾统计底稿', STATS, '合入20项昇腾指标的数值、接口与来源。'),
        ('昇腾统计调研', EVIDENCE / 'README.md', '硬件规格、采集能力与各项依据。'),
        ('设备查询', EVIDENCE / 'device-query.json', '板卡信息、HBM及匹配SDK配置。'),
        ('运行时属性', EVIDENCE / 'runtime-properties.json', '实际型号、核数、内存与L2。'),
        ('有效字段样例', EVIDENCE / 'valid-field-samples.json', 'profiling字段与实际观测。'),
        ('模拟器观察', EVIDENCE / 'simulator-observations.json', '指令热点、流水等待与空闲。'),
        ('沐曦原表依据', BASE, '1121版并行度、功能部件流水利用率：mcProfiler官方界面显示有。'),
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
    assert set(map(str, delivered.merged_cells.ranges)) == original_merges | {'D4:D8', 'D10:D13'}
    assert delivered['A19'].value == TIMING_LABEL
    assert all(delivered.cell(row, 2).value == name for row, name in METRIC_NAMES.items())
    assert [row for row, state in ascend.items() if state == PART] == [9, 14]
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
    validation = {'status': 'passed', 'output': OUTPUT.name, 'metrics': 20,
                  'vendors': 7, 'main_conclusion_cells': 140,
                  'allowed_states': [YES, NO, PART], 'ascend_evidence_rows': 20,
                  'original_merged_ranges_preserved': True,
                  'user_added_merged_ranges': ['D4:D8', 'D10:D13'],
                  'merged_range_count': len(delivered.merged_cells.ranges),
                  'visible_conclusion_cells': sum(cell.value in (YES, NO, PART)
                       for row in delivered.iter_rows(min_row=2, max_row=21, min_col=3, max_col=9) for cell in row),
                  'other_vendor_conclusions_match_1121_except_user_changes': True,
                  'five_domestic_vendor_conclusions_match_1121': True,
                  'five_domestic_vendor_metrics_checked': 100,
                  'display_slash': 'ASCII /, no adjacent spaces',
                  'user_conclusion_changes': {'C20': PART, 'C21': PART},
                  'user_metric_names': METRIC_NAMES,
                  'timing_label': TIMING_LABEL,
                  'input_files_preserved': True, 'cell_comments': 0,
                  'ascend_counts': {s: list(ascend.values()).count(s) for s in (YES, NO, PART)}}
    (REPORT / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    (REPORT / 'ascend-conclusions.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    (REPORT / 'README.md').write_text(
        '# 指标比较综合最终版\n\n'
        '2026-11-21。根目录`指标比较-综合最终版.xlsx`合并1121比较表与昇腾统计依据。\n\n'
        '主表为五层、20项指标、7家厂商，结论统一为√、×、部分；保留1121版原有41组合并单元格，'
        '按用户要求合并昇腾系统层D4:D8；硬件规格绿色同值区域为D10:D13。'
        '加深支持、不支持及部分三类单元格底色。'
        '主表采用能力与信息可获得性口径，采集窗口及实测覆盖保存在依据页。'
        '昇腾18项√、2项部分：寄存器规格、4bit向量峰值仍有缺项；'
        'L1改为“L1/SPM 容量/带宽/访问粒度”并记√，L2保留容量/带宽/cache line名称并记√。'
        '主存、矩阵峰值依据官方公开规格记√；具体型号和TF32等格式在依据页逐项记录。'
        '厂商公开资料用于判断信息可获得性，910B实测与950规格分别标明。'
        'L1缓存命中率行改为“L1/SPM 访存统计”，昇腾按L1/UB搬运、带宽及Bank冲突统计记√。'
        '流水依赖与流水空泡保留√，官方已有时序与可视化，HG-KDA进一步自动整理、量化并关联优化闭环。\n\n'
        '昇腾依据页保留20项数值、接口及统计口径；资料来源页提供原表与官方文档入口。'
        '时序层标题改为“微架构层 时序指标（模拟器/真机采样）”；NVIDIA指令热点保留√，'
        '流水线依赖与流水线空泡改为部分，资料来源页补充对应官方说明。'
        '其余厂商结论采用1121版，原表“没找到”统一为×。'
        '5家国产厂商共100项判定逐项对齐同事原表；NVIDIA仅C20、C21按用户讨论改为部分。'
        '表内统一使用半角斜杠/，斜杠前后无空格；中英文术语间距统一。'
        '两个输入文件保留，最终版无单元格批注。\n', encoding='utf-8')
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == '__main__':
    main()
