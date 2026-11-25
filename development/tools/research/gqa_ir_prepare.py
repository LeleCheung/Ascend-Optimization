"""Freeze GQA source and lowering for model-authored single-operator IR research."""
import argparse
from pathlib import Path
import shutil

from ir_campaign import read, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gqa-run', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--parent-invocation', required=True)
    p.add_argument('--target-bcmlir', type=Path, required=True)
    p.add_argument('--language-evidence', type=Path, required=True)
    p.add_argument('--mode', choices=['compile-repair', 'performance'], default='compile-repair')
    p.add_argument('--target-marker', default='tensor<256x128x2xf32>')
    p.add_argument('--host-brief', type=Path, help='Recorded source-bound analysis for another IR mechanism')
    a = p.parse_args()
    assert a.mode != 'performance' or a.host_brief, 'Performance research requires its own recorded mechanism'
    root, run = a.gqa_run.resolve(), a.run.resolve()
    parent = root / 'iterations' / a.parent_invocation
    source = parent / 'workspace/candidate.py'
    result = read(parent / 'parent-validation/results.json')
    assert result['source_sha256'] == sha(source)
    assert {row['case_id'] for row in result['cases']} == {1, 2, 4, 7}
    if a.mode == 'compile-repair':
        assert next(row for row in result['cases'] if row['case_id'] == 4)['status'] == 'failed'
    else:
        best = read(root / 'selection.json')
        assert best['source_sha256'] == sha(source) and best['invocation'] == a.parent_invocation
        assert all(row['status'] == 'measured' and all(
            row[side + '_correctness']['official_checker']['passed'] for side in ['candidate', 'native'])
            for row in result['cases'])
    raw = a.target_bcmlir.resolve()
    assert raw.is_relative_to((parent / 'parent-validation/cache').resolve())
    assert raw.suffix == '.bcmlir' and a.target_marker in raw.read_text()
    run.mkdir(parents=True, exist_ok=False)
    workspace = run / 'workspace'
    workspace.mkdir()
    shutil.copyfile(source, workspace / 'frozen.py')
    shutil.copyfile(root / 'contract.json', workspace / 'contract.json')
    shutil.copyfile(root / 'model-settings.json', run / 'model-settings.json')
    config = read(root / 'config.json')
    config.update(source_sha256=sha(source), frozen_source=str(workspace / 'frozen.py'),
                  source_frozen=True, selected_layer='single_operator_ir')
    save(run / 'config.json', config)
    relative = Path('ir/target') / raw.name
    target = workspace / relative
    target.parent.mkdir(parents=True)
    shutil.copyfile(raw, target)
    readonly = workspace / 'compiled-readonly'
    readonly.mkdir()
    for original in raw.parent.iterdir():
        if original.suffix in {'.ttir', '.ttadapter', '.bcmlir'}:
            shutil.copyfile(original, readonly / original.name)
    save(workspace / 'ir-inventory.json', dict(source_sha256=sha(source), entries=[dict(
        path=relative.as_posix(), filename=raw.name, symbol='_gqa_fwd_kernel', stage='bcmlir',
        original=str(raw), original_sha256=sha(raw))]))
    save(workspace / 'feedback.json', dict(source_sha256=sha(source), formal_result=result,
        status='target_compile_failed' if a.mode == 'compile-repair' else 'retained_four_case_best',
        original_bcmlir=str(raw),
        scope='No target latency exists for this failed source branch. Original four-case checker and timing stay fixed.'))
    best = read(root / 'selection.json')
    observation = root / 'iterations' / best['invocation']
    shutil.copyfile(observation / 'paired-parent/comparison.json', workspace / 'profiling.json')
    shutil.copyfile(observation / 'simulator-parent/summary.json', workspace / 'hotspots.json')
    summary = read(observation / 'simulator-parent/summary.json')
    save(workspace / 'timeline.json', dict(source_sha256=summary['source_sha256'],
        historical=True, reports=summary['reports'], scope='Historical retained source, separate from failed branch.'))
    shutil.copyfile(a.language_evidence, workspace / a.language_evidence.name)
    save(workspace / 'context.json', dict(source_sha256=sha(source),
        evidence_mode=a.mode, editable_ir_stage='bcmlir', target_marker=a.target_marker,
        parent_invocation=a.parent_invocation, retained_best=best,
        historical_observation_source_sha256=summary['source_sha256'],
        validation_scope='Official fixed case1/2/4/7; full-call allocation/launch timing and unchanged thresholds.',
        scope='Source branch is unretained and target compilation failed. First establish compiler repair; '
              f'compare any complete result with {best["invocation"]} separately from fixed-source IR performance gains.'))
    save(workspace / 'diagnosis.json', dict(source_sha256=sha(source),
        selected_layer='single_operator_ir', evidence=[], hypothesis='', change='', expected='', validation=''))
    save(workspace / 'codex-brief.json', dict(source_sha256=sha(source), author='Codex',
        analysis='Each source-tiled QK dot outputs 128 KiB, but 3-D join/transpose/reshape expands actual UB demand to 3146016 B.'
            'Current language.semantic.cat requires can_reorder=true and rank 1; this PV requires key-column order preservation.'
            'Investigate ordered 2-D concatenation in the same BCMLIR: directly construct final 256x256 scores, '
            'place the two 256x128 results in the first/last 128 columns, and eliminate 3-D interleaving/transposition.'
            'Preserve each FP32 value, order, both QK operations, softmax, PV, every ABI, and all attributes.'
            'Choose and edit actual IR; the host reviews equivalence/CPU compilation before full original four-case acceptance.'
            'Actual compilation determines UB/L0C usage. This branch has no target-device score; best 006 at 328.4468 us remains frozen separately.'
            'Profiling/simulation are historical 006 background. Current triggers are the full 015 compiler error and source-bound actual IR.'))
    if a.host_brief:
        brief = read(a.host_brief)
        assert brief['source_sha256'] == sha(source)
        save(workspace / 'codex-brief.json', brief)
    if a.mode == 'performance':
        context = read(workspace / 'context.json')
        context.update(historical_observation_source_sha256=None,
            observation_source_sha256=summary['source_sha256'],
            scope='Fixed-source IR research from the current fully qualified retained best. '
                  'Current profiling/simulator bind to that source; other source attempts remain background.')
        save(workspace / 'context.json', context)
        feedback = read(workspace / 'feedback.json')
        feedback['scope'] = 'Retained original IR and full-call timing are available; compare edited IR under unchanged original four cases.'
        save(workspace / 'feedback.json', feedback)
        timeline = read(workspace / 'timeline.json')
        timeline.update(historical=False, scope='Current retained source, bound per-core partial windows.')
        save(workspace / 'timeline.json', timeline)
    save(run / 'status.json', dict(status='prepared_ir_' + a.mode.replace('-', '_'), source_sha256=sha(source)))
    print(dict(status='prepared', run=str(run), source_sha256=sha(source)))


if __name__ == '__main__':
    main()
