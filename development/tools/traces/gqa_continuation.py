"""Extend a GQA best-progress curve with complete, source-bound continuation events."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .check_presentation import geometry
from .research_progress import write_report


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manual_ir_records(campaign, contract_sha256):
    """Load reviewed IR milestones without changing the source-only selection."""
    index = campaign / 'manual-ir-progress.json'
    if not index.exists():
        return []
    records = []
    for entry in read(index)['entries']:
        branch = (campaign / entry['branch']).resolve()
        assert branch.is_relative_to(campaign)
        name = entry['name']
        assert Path(name).name == name
        retention = read(branch / ('retention-' + name + '.json'))
        assert retention['retained'] and retention['post_status'] == 'complete'
        validation = branch / ('validation-' + name)
        result_path = validation / 'results.json'
        result = read(result_path)
        assert sha(result_path) == retention['formal_result_sha256']
        assert result['contract_sha256'] == contract_sha256
        assert {r['case_id'] for r in result['cases']} == {1, 2, 4, 7}
        assert all(r['status'] == 'measured' and all(
            r[side + '_correctness']['official_checker']['passed']
            for side in ['candidate', 'native']) for r in result['cases'])
        call = branch / 'agent-calls' / name
        review = read(call / 'host-review.json')
        manifest = read(validation / 'manifest.json')
        assert review['status'] == 'semantics_and_cpu_approved'
        assert review['changed_ir_sha256'] == manifest['changed_ir_sha256'] == retention['ir_sha256']
        assert result['source_sha256'] == manifest['source_sha256'] == retention['source_sha256']
        assert sha(branch / 'workspace/frozen.py') == result['source_sha256']
        inventory = read(call / 'input-workspace/ir-inventory.json')
        changed = [row for row in inventory['entries']
                   if sha(call / 'output-ir' / row['path']) == retention['ir_sha256']]
        assert len(changed) == 1
        parent_ir = sha(call / 'input-ir' / changed[0]['path'])
        assert parent_ir != retention['ir_sha256']
        binding = read(branch / ('post-' + name) / 'binary-binding-and-facts.json')
        assert binding['status'] == 'formal_profile_simulator_binary_verified'
        assert binding['source_sha256'] == result['source_sha256']
        assert binding['contract_sha256'] == contract_sha256
        assert binding['result_sha256'] == sha(result_path)
        assert binding['target_binary_sha256'] == manifest['compiled_artifact']['binary_sha256']
        target = next(r for r in result['cases'] if r['case_id'] == 4)
        record = dict(entry, source_sha256=result['source_sha256'],
            ir_sha256=retention['ir_sha256'], parent_ir_sha256=parent_ir,
            target_latency_us=target['candidate_timing']['full_call_us'],
            target_cann_us=target['native_timing']['full_call_us'],
            contract_sha256=contract_sha256, result_path=result_path,
            preedit_review=call / 'host-review.json')
        assert record['target_latency_us'] == retention['target_us']
        records.append(record)
    return records


def apply_attribution_review(events, review_path):
    review = read(review_path)
    for change in review['overrides']:
        for evidence in change['evidence']:
            path = (review_path.parent / evidence['path']).resolve()
            assert sha(path) == evidence['sha256'], 'Attribution review evidence changed'
        matches = [event for event in events if event['source_sha256'] == change['source_sha256']
                   and event.get('method')]
        assert matches, 'Reviewed source is missing from the curve'
        for event in matches:
            event['attribution_review'] = dict(
                previous_category=event['category'], previous_attribution=event.get('attribution'),
                review=str(review_path), basis=change['basis'],
                auxiliary_categories=change['auxiliary_categories'],
                retrospective_primary_review=True)
            event.update(category=change['primary_category'], method=change['method'],
                         attribution=change['attribution'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--previous-curve', type=Path, required=True)
    parser.add_argument('--methods', type=Path, required=True)
    parser.add_argument('--attribution-review', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaign, output = args.campaign.resolve(), args.output.resolve()
    reports = campaign.parent
    run = campaign / 'gqa'
    methods = read(args.methods)
    previous = deepcopy(read(args.previous_curve))
    selection_path = run / 'selection.json'
    selected = read(selection_path if selection_path.exists() else run / 'parent-selection.json')
    ir_records = manual_ir_records(campaign, selected['contract_sha256'])
    selected = min([selected, *ir_records], key=lambda r: r['target_latency_us'])
    anchor = selected['target_cann_us']
    for event in previous['events']:
        if event.get('qualified') and event.get('latency_ms'):
            event.update(score=anchor/(event['latency_ms']*1000), score_is_rebased=True)
    events = previous['events']
    known = {e['source_sha256'] for e in events}
    lineages = previous.get('source_lineage', [])
    unmeasured = []
    source_calls = {sha(path): path.parent for path in (run / 'agent-calls').glob('*/output-source.py')}
    def connect_source(digest):
        if digest in known:
            return
        assert digest in source_calls, 'Source parent missing from recorded generation history'
        call = source_calls[digest]
        parent = sha(call / 'input-source.py')
        connect_source(parent)
        known.add(digest)
        lineages.append(dict(parent=parent, child=digest, invocation=call.name,
                             formal_measurement=False))
        unmeasured.append(dict(invocation=call.name, source_sha256=digest,
            parent_source_sha256=parent, review=read(call / 'host-review.json')
                if (call / 'host-review.json').exists() else None,
            scope='Recorded source ancestor without formal performance; no plotted performance node.'))
    rows = []
    for iteration in sorted((run / 'iterations').iterdir()):
        result_path = iteration / 'parent-validation/results.json'
        if not result_path.exists():
            continue
        original_result_path = result_path
        retention = run / ('retention-' + iteration.name + '.json')
        retained_record = read(retention) if retention.exists() else {}
        if retained_record.get('measurement_result'):
            result_path = (run / retained_record['measurement_result']).resolve()
            assert result_path.is_relative_to(run)
            assert sha(result_path) == retained_record['formal_result_sha256']
            assert sha(original_result_path) == retained_record['original_formal_result_sha256']
        result = read(result_path)
        call = run / 'agent-calls' / iteration.name
        assert result['source_sha256'] == sha(call / 'output-source.py')
        parent_sha = sha(call / 'input-source.py')
        connect_source(parent_sha)
        known.add(result['source_sha256'])
        assert result['contract_sha256'] == selected['contract_sha256']
        qualified = all(r.get('status') == 'measured' and all(r.get(side+'_correctness', {}).get(
            'official_checker', {}).get('passed') for side in ['candidate','native']) for r in result['cases'])
        target = next(r for r in result['cases'] if r['case_id'] == 4)
        latency = target.get('candidate_timing', {}).get('full_call_us')
        method = methods[iteration.name]
        retained = retained_record.get('retained', False)
        review = read(call / 'host-review.json')
        assert review['status'] == 'preedit_reads_and_semantics_approved'
        event = dict(label=iteration.name + ' · ' + method['label'], case='case4',
            source_sha256=result['source_sha256'], parent_source_sha256=parent_sha,
            qualified=qualified, score=anchor/latency if qualified else None,
            presentation_eligible=retained,
            latency_ms=latency/1000 if latency else None,
            score_is_rebased=iteration.name != selected['invocation'],
            method=method['method'], category=method.get('category','sim'), location=method['location'],
            selected_layer=method['selected_layer'], attribution=method['attribution'],
            preedit_review=str((call/'host-review.json').relative_to(reports)).replace('\\','/'),
            replay=dict(result=str(result_path.relative_to(reports)).replace('\\','/')),
            formal_result_sha256=sha(result_path), actual_retention=retained)
        if result_path != original_result_path:
            event['original_measurement'] = dict(
                result=str(original_result_path.relative_to(reports)).replace('\\', '/'),
                sha256=sha(original_result_path), scope='Same-source measurement retained in details')
        events.append(event)
        lineages.append(dict(parent=parent_sha, child=result['source_sha256'], invocation=iteration.name))
        rows.append(dict(invocation=iteration.name, qualified=qualified, target_us=latency, retained=retained,
                         source_sha256=result['source_sha256'], method=method['label']))
        for ir in [r for r in ir_records if r['after_invocation'] == iteration.name]:
            assert ir['source_sha256'] in known
            events.append(dict(label=ir['invocation'] + ' · ' + ir['label'], case='case4',
                source_sha256=ir['source_sha256'], parent_source_sha256=ir['source_sha256'],
                ir_sha256=ir['ir_sha256'], parent_ir_sha256=ir['parent_ir_sha256'],
                qualified=True, score=anchor / ir['target_latency_us'],
                presentation_eligible=True, latency_ms=ir['target_latency_us'] / 1000,
                score_is_rebased=ir['invocation'] != selected['invocation'],
                method=ir['method'], category='sim', location='ir',
                selected_layer='single_operator_ir', attribution=ir['attribution'],
                preedit_review=ir['preedit_review'].relative_to(reports).as_posix(),
                replay=dict(result=ir['result_path'].relative_to(reports).as_posix()),
                formal_result_sha256=sha(ir['result_path']), actual_retention=True))
            lineages.append(dict(parent=ir['source_sha256'], child=ir['source_sha256'],
                parent_ir_sha256=ir['parent_ir_sha256'], ir_sha256=ir['ir_sha256'],
                invocation=ir['invocation'], relation='fixed_source_ir_edit'))
            rows.append(dict(invocation=ir['invocation'], qualified=True,
                target_us=ir['target_latency_us'], retained=True,
                source_sha256=ir['source_sha256'], ir_sha256=ir['ir_sha256'], method=ir['label']))
    assert all(any(row['invocation'] == ir['invocation'] for row in rows) for ir in ir_records)
    if args.attribution_review:
        apply_attribution_review(events, args.attribution_review.resolve())
    previous.update(events=events, cann_anchor_us=anchor, source_lineage=lineages,
        source_events_without_measurement=unmeasured,
        optimization_loop='Profiling／模拟器取证 → 选择修改层 → 模型修改 → 真机验证 → 重放核对',
        scope=previous['scope'] + ' 本批新增事件按真实源码父链保存；前期最佳作为独立分支共同起点，'
              '完整合格但退步的探索及同实现测量归入详情。原始成绩与原版图冻结。')
    previous['axis_max'] = max(.2, round(anchor/selected['target_latency_us']*1.2 + .025, 1))
    previous['history_links'] = [
        dict(href='../'+campaign.name+'/README.md', label='本次全部结果与机制分析'),
        dict(href='../'+campaign.name+'/gqa/selection.json' if selection_path.exists() else
                  '../'+campaign.name+'/gqa/parent-selection.json', label='当前源码保留选择'),
        dict(href='../'+args.previous_curve.parent.name+'/index.html', label='此前冻结曲线')]
    extra_links = campaign / 'presentation-history-links.json'
    if extra_links.exists():
        previous['history_links'].extend(read(extra_links))
    if ir_records:
        previous['history_links'].append(dict(href='../' + campaign.name + '/manual-ir-progress.json',
                                             label='固定源码IR进展与身份'))
    write_report(previous, output)
    rendered = read(output / 'mainline-presentation.json')
    displayed = [rendered['events'][index]
                 for index in rendered['presentation_selection']['displayed_indices']]
    assert displayed[-1]['source_sha256'] == selected['source_sha256'], 'Curve must end at the accepted best'
    assert displayed[-1].get('ir_sha256') == selected.get('ir_sha256'), 'Curve must preserve the accepted IR identity'
    assert abs(displayed[-1]['score'] - anchor / selected['target_latency_us']) < 1e-12
    assert geometry(output / 'mainline.svg') == len(rendered['presentation_selection']['displayed_indices']) - 1
    receipt = dict(status='passed', rows=rows, unified_cann_anchor_us=anchor,
        selected_indices=rendered['presentation_selection']['displayed_indices'],
        hidden_indices=rendered['presentation_selection']['detail_indices'],
        source_parents_verified=True, continuous=True, monotonic=True, remeasurements_hidden=True)
    (campaign/'trace-validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__':
    main()
