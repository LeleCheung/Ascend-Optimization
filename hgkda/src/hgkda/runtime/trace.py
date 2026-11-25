"""Persist run events and pre-change decisions without executing candidates or inferring causal tool contributions."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_bytes())


def immutable_bytes(path, raw, allow_existing=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + uuid.uuid4().hex + '.partial')
    try:
        temporary.write_bytes(raw)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not allow_existing:
                raise
            assert path.read_bytes() == raw, 'trace content identity changed'
    finally:
        temporary.unlink(missing_ok=True)


def immutable(path, value):
    immutable_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf8'))


def reference(path, run):
    path = Path(path).resolve()
    try:
        name = path.relative_to(run.resolve()).as_posix()
    except ValueError:
        name = str(path)
    return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def source_snapshot(config):
    run = Path(config['run_directory'])
    source = run / 'workspace/candidate.py'
    if not source.exists():
        source = run / 'workspace' / config.get('active_source', 'starter.py')
    if not source.exists():
        return None
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    frozen = run / 'trace/sources' / (digest + '.py')
    immutable_bytes(frozen, raw, allow_existing=True)
    return reference(frozen, run)


def emit(config, kind, **fields):
    run = Path(config['run_directory'])
    event_id = uuid.uuid4().hex
    event = dict(schema='hgkda.trace-event/v1', id=event_id, recorded_at=now(), kind=kind,
                 operator=config.get('operator', config.get('job', {}).get('operator')),
                 round_id=config.get('trace_round_id'), parent_round_id=config.get('trace_parent_round_id'), **fields)
    # Create events independently instead of sharing an append file, preventing parallel model/host truncation.
    immutable(run / 'trace/events' / (event_id + '.json'), event)
    return event


def record_decision(config, document):
    """Bind current source and explicit evidence before editing; attribution remains an author claim."""
    run = Path(config['run_directory'])
    source = source_snapshot(config)
    assert source and document.get('source_sha256') == source['sha256'], 'decision source is not current source'
    for key in ('hypothesis', 'change', 'expected', 'validation'):
        assert isinstance(document.get(key), str) and document[key].strip(), 'missing decision field: ' + key
    evidence = []
    for entry in document.get('evidence', []):
        assert entry.get('role') in ('trigger', 'disambiguation', 'mechanism', 'background'), 'invalid evidence role'
        assert entry.get('category') in ('paired_profiling', 'simulator', 'source_ir', 'roofline', 'knowledge', 'evaluation', 'probe', 'other')
        path = Path(entry['path'])
        if not path.is_absolute():
            path = run / path
        ref = reference(path, run)
        if entry.get('sha256'):
            assert entry['sha256'] == ref['sha256'], 'evidence file changed'
        if entry.get('source_sha256'):
            assert entry['role'] == 'background' or entry['source_sha256'] == source['sha256'], 'stale evidence source'
        # Verify actual files for structured bindings instead of trusting model-supplied source identity.
        bound = None
        if path.suffix == '.json':
            payload = read(path)
            if isinstance(payload, dict):
                wrapped_result=payload.get('result')
                evaluation_feedback=(payload.get('schema_version')=='hgkda.result/v1'
                                     and isinstance(wrapped_result,dict)
                                     and 'geo_mean' in wrapped_result and 'ranked_cases' in wrapped_result)
                if evaluation_feedback:
                    assert entry['category']!='paired_profiling', (
                        'full-contract performance feedback is evaluation evidence, not paired_profiling; '
                        'use category=evaluation (or other for older schemas)')
                bound = payload.get('source_sha256')
                wrapped = payload.get('result')
                if not bound and isinstance(wrapped, dict):
                    spec = wrapped.get('spec')
                    bound = wrapped.get('source_sha256') or (spec.get('source_sha256') if isinstance(spec, dict) else None)
                if bound and entry.get('source_sha256'):
                    assert entry['source_sha256'] == bound, 'declared evidence source differs from artifact'
                if entry['role'] != 'background':
                    assert not bound or bound == source['sha256'], ('evidence artifact belongs to another source; '
                        'historical comparisons must use role=background with their actual source_sha256, '
                        'not trigger/mechanism evidence for the current candidate')
        # Context, knowledge, and candidate paths may be overwritten later; decisions must reference the bytes used then.
        raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==ref['sha256'],'evidence changed while recording'
        frozen=run/'trace/evidence'/(ref['sha256']+path.suffix)
        immutable_bytes(frozen,raw,allow_existing=True)
        evidence.append(dict(reference=reference(frozen,run), original=ref, category=entry['category'], role=entry['role'],
                             finding=entry.get('finding', ''), source_sha256=bound or entry.get('source_sha256')))
    decision_id = uuid.uuid4().hex
    saved = dict(schema='hgkda.trace-decision/v1', id=decision_id, recorded_at=now(),
                 round_id=config.get('trace_round_id'), input_source=source, evidence=evidence,
                 hypothesis=document['hypothesis'], change=document['change'], expected=document['expected'],
                 validation=document['validation'], alternatives=document.get('alternatives', []),
                 discriminating_observation=document.get('discriminating_observation', ''),
                 author=document.get('author', 'agent'), attribution_status='author_claim_not_causal_proof')
    path = run / 'trace/decisions' / (decision_id + '.json')
    immutable(path, saved)
    emit(config, 'decision_recorded_before_change', decision=reference(path, run), source=source)
    return dict(status='recorded', decision_id=decision_id, path=str(path), source_sha256=source['sha256'])


def artifacts(config, source, round_id, extra_paths=()):
    run = Path(config['run_directory'])
    paths = [run / 'workspace' / name for name in ('diagnosis.json', 'latest-preflight.json',
             'performance-feedback.json', 'context-brief.json', 'paired-context-pointer.json', 'regions-context-pointer.json')]
    if source:
        digest = source['sha256']
        paths += [run / 'attempts' / digest / 'preflight.json']
        formal = run / ('formal-' + digest[:12])
        paths += [formal / name for name in ('request.json', 'evaluation/result.json',
                                             'performance-feedback.json', 'audit-final.json')]
    if round_id:
        call = run / 'agent-calls' / round_id
        paths += [call / name for name in ('input-source.py', 'output-source.py', 'source-provenance.json',
                  'initial-context.json', 'prompt.md', 'agent-events.jsonl', 'model-usage.json', 'method-manifest.json')]
    for item in extra_paths:
        path = Path(item)
        paths += list(path.glob('*.json')) if path.is_dir() else [path]
    # workspace is mutable: snapshot bytes for each event instead of referencing a changing path.
    refs = []
    for path in dict.fromkeys(paths):
        if not path.is_file():
            continue
        if path.parent == run / 'workspace' or path.name=='source-provenance.json':
            raw = path.read_bytes()
            copy = run / 'trace/artifacts' / (hashlib.sha256(raw).hexdigest() + path.suffix)
            immutable_bytes(copy, raw, allow_existing=True)
            refs.append(dict(original=path.relative_to(run).as_posix(), **reference(copy, run)))
        elif path.name=='agent-events.jsonl':
            # The host model may still append logs when nested tools return; save prefix length and prefix SHA.
            raw=path.read_bytes()
            refs.append(dict(path=path.relative_to(run).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),
                             prefix_bytes=len(raw),scope='append-only model stream prefix'))
        else:
            refs.append(reference(path, run))
    return refs


@contextmanager
def action_trace(config, action, round_id=None, extra_paths=()):
    config = dict(config)
    if round_id is not None:
        assert Path(round_id).name == round_id and round_id not in ('.', '..'), 'invalid round id'
        config['trace_round_id'] = round_id
    run = Path(config['run_directory'])
    source = source_snapshot(config)
    request = Path(config.get('job', {}).get('request', run / 'request.json'))
    started = emit(config, 'action_started', action=action, input_source=source,
                   contract=reference(request, run) if request.is_file() else None)
    error = None
    try:
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        output = source_snapshot(config)
        refs = artifacts(config, output, config.get('trace_round_id'), extra_paths)
        decisions = []
        for path in (run / 'trace/decisions').glob('*.json'):
            item = read(path)
            if item['round_id'] == config.get('trace_round_id') and item['recorded_at'] >= started['recorded_at']:
                decisions.append(reference(path, run))
        changed = source != output
        ended = emit(config, 'action_finished', action=action, started_event=started['id'],
                     status='raised' if error else 'returned', error_type=error,
                     input_source=source, output_source=output, source_changed=changed,
                     artifacts=refs, decisions=decisions,
                     attribution_status='recorded_claim' if decisions else 'not_recorded')
        if action == 'generate' and config.get('trace_round_id'):
            immutable(run / 'trace/rounds' / config['trace_round_id'] / 'round.json', ended)


def traced_main(function, action, argv=None):
    """Thin wrapper preserving device actions, abnormal exits, and candidate bytes of existing commands."""
    argv = list(sys.argv[1:] if argv is None else argv)
    def option(flag):
        for index, token in enumerate(argv):
            if token == flag:
                return argv[index + 1] if index + 1 < len(argv) else None
            if token.startswith(flag + '='):
                return token.split('=', 1)[1]
        return None
    if not option('--config') or '--help' in argv or '-h' in argv:
        return function()
    config = read(option('--config'))
    invocation = option('--invocation')
    extra_paths = [Path(option(flag)).resolve() for flag in ('--output', '--paired-directory') if option(flag)]
    with action_trace(config, action, invocation, extra_paths):
        return function()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    events = [read(p) for p in (Path(config['run_directory']) / 'trace/events').glob('*.json')]
    events.sort(key=lambda event: (event['recorded_at'], event['id']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status='exported', events=len(events), output=str(args.output))))


if __name__ == '__main__':
    main()
