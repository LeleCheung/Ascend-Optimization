#!/usr/bin/env python3
"""把成对采集压缩为可注入KG的事实摘要；不将seed测量冒充新候选。"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('输出已存在，保留旧摘要')
    module_path = Path(__file__).with_name('skills')/'hgkda/scripts/compare_opprof.py'
    s = importlib.util.spec_from_file_location('paired_summarizer', module_path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    rows = []
    chosen = ['aiv_mte2_instructions', 'aiv_mte3_instructions', 'GM_to_UB_datas(KB)',
        'UB_to_GM_datas(KB)', 'aiv_vec_time(us)', 'aiv_scalar_time(us)',
        'aiv_mte2_time(us)', 'aiv_mte3_time(us)', 'aiv_scalar_wait_time(us)',
        'aiv_vec_bank_cflt_ratio', 'read_main_memory_datas(KB)']
    for directory in sorted((a.batch_dir/'preflight').glob('*/paired-v2-*')):
        if not (directory/'candidate/collection.json').exists():
            continue
        full = m.compare(directory)
        outfile = directory/'comparison.json'
        if not outfile.exists():
            outfile.write_text(json.dumps(full, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        index = full['spec']['case_id'].split('-')[-1]
        probes = {}
        for side in ['reference', 'candidate']:
            probe = directory.parent/('probe-%s-%s.json' % (index, side))
            if probe.exists():
                probes[side] = dict(path=str(probe), median_us=json.loads(probe.read_text())['median_us'])
        rows.append(dict(operator=full['spec']['operator'], source_sha256=full['spec']['source_sha256'],
            case=full['spec'], evidence=str(outfile),
            reference=full['reference']['basic'], candidate=full['candidate']['basic'],
            differences=[row for row in full['differences'] if row['metric'] in chosen
                         and not row['table'].startswith('Arithmetic')],
            derived=full['derived_differences'], walltime_probes=probes))
    if len(rows) != 6:
        raise ValueError('本批计划的两算子各3组成对证据尚未完整：'+str(len(rows)))
    result = dict(schema_version='hgkda-paired-v1', cases=rows,
        contribution='同输入成对实机计数，由确定性脚本整理，因果分析交给agent',
        guidance=['小输入先比较实际host调用路径；设备快不代表完整调用快',
            '大输入比较搬运指令/字节、分块和IR；活跃比例不自动等于瓶颈',
            '起点seed的证据用于形成实验，新候选需要同身份采集和KGS完整评测'])
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('已汇总6组成对证据：'+str(a.output))


if __name__ == '__main__':
    main()
