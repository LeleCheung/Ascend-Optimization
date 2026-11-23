"""Fill only the Ascend capability column, with auditable per-metric notes."""
from __future__ import annotations

from copy import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(os.environ['TEMP']) / 'ascend-stats-20261120-libs'))
from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'development/reports/ascend-statistics-20261120'
BOOK = ROOT / '统计数据.xlsx'
SCOPE = '2026-11-20；卡7 Ascend910B4-1；CANN 9.0.0；驱动25.2.0；1650MHz观察；规格与采集能力分别记录。'


def main():
    doc = json.loads((OUT / 'paired-observations.json').read_text())
    gn = doc['native-group-norm']['result']['candidate']
    gqa = doc['gqa']['candidate']
    samples = {}
    for op, candidate in [('native_group_norm', gn), ('GQA', gqa)]:
        samples[op] = {'basic': candidate['basic'], 'valid_fields': {}}
        for table, values in candidate['tables'].items():
            samples[op]['valid_fields'][table] = {k: v for k, v in values['fields'].items() if v['valid']}
    (OUT / 'valid-field-samples.json').write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding='utf-8')
    # Row, concise capability, definition/available interface, verified observation, remaining boundary, evidence.
    entries = [
        (2, '✔', 'Host启动：msprof CANN/Runtime API时间线与api_statistic；选定launch API的Duration(us)。',
         '官方msprof文档确认AscendCL、Runtime等API耗时及HostToDevice关联。',
         '本次复用文档支持结论；未新增应用级host采集。API耗时与device执行时间分别报告。', 'official/msprof-profile_data_file_references.md#api_statistic'),
        (3, '✔', '设备kernel：msOpProf OpBasicInfo.csv的Task Duration(us)。',
         f"GN当前kernel {gn['basic']['Task Duration(us)']}μs，GQA {gqa['basic']['Task Duration(us)']}μs。",
         '单kernel profiling耗时；正式完整调用性能由原KGS／GQA合同测量保存。', 'paired-observations.json'),
        (4, '✔*', 'Block Dim、Mix Block Dim与逐block/sub_block记录；系统时间线可观察任务重叠。',
         'GN配置64个program；实际设备20 Cube、40 Vector。GQA配置1024／2048。',
         '启动规模、物理核数和同时驻留／占用率分别解释；本次仅核实前两项。', 'paired-observations.json; runtime-properties.json'),
        (5, '✔', 'ArithmeticUtilization.csv：aic_cube_fops、aiv_vec_fops，工具定义为浮点运算计数。',
         'GN aiv_vec_fops 64/64有效，均值179260.5；GQA Cube 1024/1024有效，均值98304。',
         'PMU指令口径；含实际执行开销，算法FLOPs及FMA计数口径需单列。', 'valid-field-samples.json; official/msopprof-msopprof_performance_data.md'),
        (6, '✔', 'Memory.csv：read/write_main_memory_datas(KB)、GM_to_L1/UB、UB_to_GM等。',
         'GN 64/64有效：GM→UB 256 KB/program，UB→GM 128.25 KB/program。',
         '按数据通路统计；保留工具KB单位及estimate标签；逻辑搬运与HBM物理事务分别解释。', 'valid-field-samples.json; official/msopprof-msopprof_performance_data.md'),
        (7, '部分', '实测Cube/Vector各dtype cycle占比；msprof另文档化cube_utilization(%)。',
         'GN aiv_vec_ratio均值0.4083583，FP32占比0.3772072，均64/64有效。',
         '当前实测为流水占比；整卡FLOPs/s÷匹配dtype峰值尚待峰值分母及计数口径核验。', 'valid-field-samples.json; official/msprof-profile_data_file_references.md#PipeUtilization'),
        (8, '✔*', 'Memory.csv各数据通路bw_usage_rate(%)及GB/s。',
         'GN GM→UB利用率14.096889%，UB→GM 8.319647%，64/64有效。',
         '数据通路及工具分母限定；整卡HBM峰值利用率需独立峰值带宽，estimate字段保留。', 'valid-field-samples.json; official/msopprof-msopprof_performance_data.md'),
        (9, '待确认', '寄存器数量、位宽及每核总容量需按Scalar／专用寄存器区分。',
         '已查SDK platform配置、平台查询头文件及公开profiler文档，未取得完整型号匹配的寄存器规格。',
         'UB 192 KiB及L0/L1容量分别属于本地存储；PMU寄存器数值也单独解释。', 'device-query.json; runtime-properties.json; official/'),
        (10, '部分', 'Ascend L1为软件管理Buffer：SDK l1_size；msOpProf aic_l1_read/write_bw。',
         'B4-1配置L1 524288 B＝512 KiB；GQA L1读／写字段1024/1024有效。',
         '峰值带宽待确认；cache line对软件管理L1 Buffer口径不适用。UB block 32B保留为UB粒度。', 'device-query.json; valid-field-samples.json'),
        (11, '部分', 'L2容量由运行时属性与匹配SoC SDK配置查询。',
         '实际B4-1运行时L2 168MB；匹配配置176160768 B＝168 MiB。',
         '峰值带宽、cache line size待确认；通用B4.ini为96 MiB，须选择实际B4-1配置。', 'runtime-properties.json; device-query.json'),
        (12, '部分', 'npu-smi memory查询板卡HBM容量；profiler另提供实测访存带宽。',
         '板卡HBM Capacity(MB)=65536；运行时total_memory=62420MB；GN主存读均值31.080297 GB/s/program。',
         '保留容量工具原单位；物理容量与运行时可见容量分别记录。整卡理论峰值HBM带宽待确认，时钟1600MHz单列。', 'device-query.json; runtime-properties.json; valid-field-samples.json'),
        (13, '部分', '矩阵dtype能力与各dtype峰值TOPS／TFLOPS分别登记。',
         '匹配SDK配置含INT4/INT8 DtypeMKN及FP16/FP32相关mmad映射；GQA BF16目标实际通过。',
         '4/8/16/32bit全部峰值规格尚待型号匹配资料；区分INT/FP、累加dtype及稀疏性，配置声明仅作能力线索。', 'device-query.json; paired-observations.json'),
        (14, '部分', '向量算术、转换、搬运按具体指令及dtype区分。',
         'SDK vadd/vmul列FP16/FP32/INT16/INT32；8bit可见转换／搬运；本次GN执行FP32。',
         '8bit转换／搬运与同位宽原生算术分开记录；4/8/16/32bit整卡峰值表待确认。', 'device-query.json; valid-field-samples.json'),
        (15, '✔', '实际可用Cube／Vector物理核数。',
         'torch_npu属性20 Cube、40 Vector；npu-smi Aicore Count=20；匹配SDK一致。',
         '模拟器曾初始化24 AIC／48 Vector，采集配置与真机可用核数分别标识。', 'runtime-properties.json; device-query.json; ../regions-method-20261110.md'),
        (16, '✔', 'PipeUtilization.csv：Cube/Vector/MTE1/2/3/FixPipe/Scalar时间与cycle占比。',
         'GN Vector=0.4083583、MTE2=0.5621627、MTE3=0.2697368，64/64有效。',
         '各流水允许重叠；使用各自周期分母，保持逐核与全调用口径。', 'valid-field-samples.json; official/msopprof-msopprof_performance_data.md'),
        (17, '不适用*', '软件管理的L1 Buffer无通用数据cache hit/miss统计；I-cache单列。',
         '本次msOpProf支持L1带宽；另有aic/aiv_icache_miss_rate，官方定义为指令预留缓存。',
         'L1数据缓存命中率按此架构口径不适用；指令cache miss单列，保留其官方定义。', 'official/msprof-profile_data_file_references.md; valid-field-samples.json'),
        (18, '✔', 'L2Cache.csv：read/write/total_hit_rate(%)及hit/miss_allocate计数。',
         'GN读取命中率均值50.095173%、写0.763082%、合计44.585355%，64/64有效。',
         '这些为逐block算术均值示例；全调用命中率应汇总hit与miss计数再计算。', 'valid-field-samples.json; official/msopprof-msopprof_performance_data.md'),
        (19, '✔*', '模拟器instr_exe.csv、code_exe.csv及trace.json关联PC、指令、源码行。',
         '当前GN已采指令／区域热点，GQA三个核均有局部窗口报告。',
         '热点累计时长用于排序，流水可重叠；源码映射受调试信息限制。', 'simulator-observations.json; official/msopprof-msopprof_simulator_performance_data.md'),
        (20, '✔*', '模拟器逐流水事件与WAIT／FLAG／BAR用于观察依赖和同步等待。',
         'GN与GQA报告提供分流水时间轴及等待事件，已关联源码／PC。',
         '已观测同步关系与等待范围可查；完整动态依赖图、全芯片因果链仍需专项分析。', 'simulator-observations.json; official/msopprof-msopprof_simulator_performance_data.md'),
        (21, '✔*', '在共同已观测窗口内，对已报告流水执行事件取并集，补集为可观测空闲。',
         '现有分析输出active、wait_only及observable idle；可对照修改前后窗口。',
         '窗口与核边界明确；无元数据泳道保持未知，空白区原因需结合依赖及搬运进一步判断。', 'simulator-observations.json; ../regions-method-20261110.md'),
    ]
    before_path = OUT / 'statistics-before-ascend.xlsx'
    if not before_path.exists():
        shutil.copy2(BOOK, before_path)
    before = load_workbook(before_path)
    work = load_workbook(BOOK)
    sheet = work['Sheet1']
    merges = sorted(map(str, sheet.merged_cells.ranges))
    records = []
    for row, short, interface, observed, boundary, evidence in entries:
        records.append(dict(row=row, metric=sheet.cell(row, 2).value, capability=short, interface=interface,
            observed=observed, boundary=boundary, evidence=evidence, scope=SCOPE))
    for row in [2, *range(4, 19), 19]:
        group = [r for r in records if r['row'] in ({2: [2, 3], 19: [19, 20, 21]}.get(row, [row]))]
        cell = sheet.cell(row, 4)
        cell.value = next(r['capability'] for r in records if r['row'] == row)
        note = SCOPE + '\n\n' + '\n\n'.join('\n'.join(str(r[k]) for k in ['metric', 'interface', 'observed', 'boundary', 'evidence']) for r in group)
        cell.comment = Comment(note + '\n完整依据见“昇腾依据”工作表。', 'HG-KDA')
        cell.comment.width = 620
        cell.comment.height = 360
    if '昇腾依据' in work:
        del work['昇腾依据']
    proof = work.create_sheet('昇腾依据')
    headers = ['原表行', '指标', '结论', '定义／接口', '已核验事实', '适用范围／未知项', '证据文件', '设备／版本']
    proof.append(headers)
    for r in records:
        proof.append([r[k] for k in ['row', 'metric', 'capability', 'interface', 'observed', 'boundary', 'evidence', 'scope']])
    proof.freeze_panes = 'D2'
    proof.auto_filter.ref = proof.dimensions
    for col, width in zip('ABCDEFGH', [10, 32, 14, 65, 65, 70, 65, 38]):
        proof.column_dimensions[col].width = width
    for cells in proof:
        proof.row_dimensions[cells[0].row].height = 75 if cells[0].row > 1 else 28
        for cell in cells:
            cell.alignment = Alignment(vertical='top', wrap_text=True)
            if cell.row == 1:
                cell.fill = PatternFill('solid', fgColor='235A45')
                cell.font = Font(color='FFFFFF', bold=True)
    work.save(BOOK)
    after = load_workbook(BOOK)
    # This is an artifact-preservation check, not an added operator test suite.
    s0, s1 = before['Sheet1'], after['Sheet1']
    assert sorted(map(str, s0.merged_cells.ranges)) == merges == sorted(map(str, s1.merged_cells.ranges))
    changed = []
    for cells in s0:
        for cell in cells:
            other = s1[cell.coordinate]
            if cell.value != other.value:
                assert cell.column == 4 and cell.row in range(2, 22), cell.coordinate
                changed.append(cell.coordinate)
            assert copy(cell._style) == copy(other._style), cell.coordinate
    assert all(s1.cell(row, 4).value for row in [2, *range(4, 20)])
    assert after['昇腾依据'].max_row == 21
    save = lambda name, value: (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    save('matrix-evidence.json', records)
    save('workbook-validation.json', dict(status='passed', metrics=20, newly_filled=15, merged_ranges_preserved=len(merges),
        existing_cell_styles_preserved=True, other_vendor_values_preserved=True, changed_cells=changed,
        backup=str(before_path.relative_to(ROOT)), workbook_sha256=hashlib.sha256(BOOK.read_bytes()).hexdigest()))
    sources = json.loads((OUT / 'official-sources.json').read_text())
    text = '# 昇腾统计数据补齐\n\n' + SCOPE + '\n\n'
    text += '根目录`统计数据.xlsx`已补齐15项空白，并为20项指标添加逐项依据。原有合并、样式及其他厂商数值保留；原件备份为`statistics-before-ascend.xlsx`。表内简写：✔表示该项能力有文档或实际观测；部分表示仅覆盖子项；待确认表示本次资料仍缺具体规格；星号范围见批注及“昇腾依据”页。\n\n'
    text += '| 指标 | 结论 | 已核验事实 | 范围与未知项 |\n| --- | --- | --- | --- |\n'
    for r in records:
        text += f"| {r['metric']} | {r['capability']} | {r['observed']} | {r['boundary']} |\n"
    text += '\n实际型号由torch_npu与AscendCL共同查询；B4-1运行时L2为168MB，匹配SDK为168MiB。通用B4配置L2为96MiB，SDK静态memory_size为32GiB，实际板卡HBM查询为65536MB。运行时可见内存为62420MB。分别保存这些值及其来源，采用实际设备属性解释规格差异。SDK配置中的带宽rate缺少本次确认的单位，未据此推导整卡峰值。\n\n'
    text += '当前实测有效字段保存在[有效字段样例](valid-field-samples.json)，完整成对报告保存在[原始观察](paired-observations.json)。GN逐program均值仅用于展示字段有效性；加速比、全调用时间及跨核聚合沿原合同计算。当前GQA计数字段带有本次采集范围，后续性能归因继续与模拟器及实际IR交叉核对。\n\n'
    text += '官方资料固定版本：\n\n' + '\n'.join(f"- [{Path(s['local']).name}]({s['url']})，commit `{s['commit']}`。" for s in sources)
    text += '\n\n[设备查询](device-query.json) · [运行时属性](runtime-properties.json) · [模拟器观察](simulator-observations.json) · [表格保存检查](workbook-validation.json)。\n'
    (OUT / 'README.md').write_text(text, encoding='utf-8')
    print(json.dumps(dict(status='passed', filled=15, evidence_rows=20, changed=changed)))


if __name__ == '__main__':
    main()
