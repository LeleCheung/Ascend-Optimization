#!/usr/bin/env python3
"""Reconcile the original workbook with archived V2 results; no GPU execution."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
VENDORS = {'天数': 'tianshu', '摩尔': 'moer', '平头哥': 'pingtouge', '沐曦': 'muxi', '海光': 'hygon', '昇腾': 'huawei'}

def read_workbook(path):
    result = []
    with ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(x.itertext()) for x in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        relations = {x.attrib['Id']: x.attrib['Target'] for x in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        for sheet in ET.fromstring(archive.read('xl/workbook.xml')).find('m:sheets', NS):
            name = sheet.attrib['name']
            vendor = next((v for prefix, v in VENDORS.items() if name.startswith(prefix)), None)
            target = relations[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            rows = []
            for row in ET.fromstring(archive.read(target)).findall('m:sheetData/m:row', NS):
                cells = {}
                for cell in row.findall('m:c', NS):
                    col = re.match(r'[A-Z]+', cell.attrib['r']).group()
                    value = cell.find('m:v', NS)
                    value = value.text if value is not None else ''
                    if cell.attrib.get('t') == 's':
                        value = strings[int(value)] if value else ''
                    elif cell.attrib.get('t') == 'inlineStr':
                        value = ''.join(t.text or '' for t in cell.findall('.//m:t', NS))
                    cells[col] = value
                if cells.get('A') and int(row.attrib['r']) > 1:
                    rows.append({'id': f"{vendor}:{row.attrib['r']}", 'vendor': vendor, 'sheet': name, 'row': int(row.attrib['r']), 'cells': cells})
            result.append({'vendor': vendor, 'sheet': name, 'rows': rows})
    return result


def speed(value):
    match = re.fullmatch(r'\s*([0-9]+(?:\.[0-9]+)?)\s*[xX×]?\s*\*?\s*', str(value or ''))
    return float(match.group(1)) if match else None


def failure_reasons(cells):
    reasons = []
    status = ' '.join(str(cells.get(k) or '') for k in ('D', 'H'))
    if re.search(r'不通过|不过|错误|\bFAIL(?:ED)?\b', status, re.I):
        reasons.append('explicit_failure')
    if re.search(r'超时|无测试|不可运行|TIMEOUT', status, re.I):
        reasons.append('incomplete_test')
    value = speed(cells.get('E'))
    if value is not None and value < 0.8:
        reasons.append('speedup_below_0_8')
    if re.search(r'超时|TIMEOUT|benchmark.*(?:失败|错误)', str(cells.get('E') or ''), re.I):
        reasons.append('timing_unavailable')
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--attribution', type=Path, required=True, help='Previous archive identity mapping, not trusted root-cause labels')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive_rows = {r['id']: r for r in json.loads(args.attribution.read_text())['rows']}
    sheets = read_workbook(args.workbook)
    selected = []
    exclusions = []
    for sheet in sheets:
        if sheet['vendor'] == 'huawei':
            continue
        for row in sheet['rows']:
            old = archive_rows.get(row['id'])
            if old is None:
                exclusions.append({**row, 'reason': 'no_archive_mapping'})
                continue
            reasons = failure_reasons(row['cells'])
            if old.get('historical_archive_status') != '成功' or not reasons:
                exclusions.append({'id': row['id'], 'reason': 'no_prior_success' if old.get('historical_archive_status') != '成功' else 'no_reported_failure'})
                continue
            source = Path(old['candidate_path'])
            digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
            assert digest == old['candidate_sha256'], f"source changed/missing: {row['id']}"
            assert row['cells']['A'] == old['operator'], f"row identity mismatch: {row['id']}"
            selected.append({**row, 'operator': source.stem, 'archive_status': old['historical_archive_status'], 'workbook_old_status': row['cells'].get('B'), 'reported_failure_reasons': reasons, 'reported_speedup': speed(row['cells'].get('E')), 'candidate_path': str(source), 'candidate_sha256': digest, 'prior_native_result': old.get('native_result_path'), 'prior_attribution_state': old.get('attribution_state')})
    summary = {'counting_unit': 'vendor/operator workbook row with archive success; failed includes performance below 0.8', 'total': len(selected), 'unique_operator_names': len({r['operator'] for r in selected}), 'by_vendor': dict(Counter(r['vendor'] for r in selected)), 'failure_signals_overlapping': dict(Counter(reason for r in selected for reason in r['reported_failure_reasons'])), 'workbook_sha256': hashlib.sha256(args.workbook.read_bytes()).hexdigest(), 'archive_mapping_sha256': hashlib.sha256(args.attribution.read_bytes()).hexdigest(), 'sheets': [{'vendor': s['vendor'], 'data_rows': len(s['rows']), 'excluded': s['vendor'] == 'huawei'} for s in sheets]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary': summary, 'rows': selected, 'exclusions': exclusions}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
