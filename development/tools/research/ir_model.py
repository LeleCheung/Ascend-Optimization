"""Record CC/DeepSeek TTIR edits while keeping operator source immutable."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(workspace):
    return {p.relative_to(workspace).as_posix(): sha(p) for p in workspace.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts}


def checked(run, invocation):
    workspace = run / 'workspace'
    call = run / 'agent-calls' / invocation
    initial = read(call / 'input-manifest.json')
    actual = files(workspace)
    editable = {row['path'] for row in read(workspace / 'ir-inventory.json')['entries']} | {'diagnosis.json'}
    assert set(actual) == set(initial), 'Workspace file set changed'
    assert all(actual[name] == digest for name, digest in initial.items() if name not in editable), 'Frozen input changed'
    assert actual['frozen.py'] == read(run / 'config.json')['source_sha256']
    return workspace, call, initial, actual, editable


def bridge(run, invocation, action):
    workspace, call, initial, actual, editable = checked(run, invocation)
    diagnosis = read(workspace / 'diagnosis.json')
    if action == 'decision':
        assert all(actual[name] == initial[name] for name in editable if name != 'diagnosis.json'), 'IR edited before decision'
        assert diagnosis['source_sha256'] == actual['frozen.py']
        wording = json.dumps(diagnosis, ensure_ascii=False)
        assert all(phrase not in wording for phrase in ['\u4e0d' + '\u662f', '\u4e0d' + '\u7b49\u4e8e', '\u4e0d\u80fd' + '\u8bc1\u660e']), \
            'Use affirmative concrete statements for evidence, scope and next actions; revise diagnosis wording'
        for name in ['hypothesis', 'change', 'expected', 'validation']:
            assert diagnosis.get(name), 'Diagnosis field required: ' + name
        assert diagnosis.get('selected_layer') == 'single_operator_ir'
        categories = set()
        evidence = []
        for item in diagnosis['evidence']:
            path = (workspace / item['path']).resolve()
            assert path.is_relative_to(workspace.resolve()) and path.is_file(), \
                'Evidence path must name a real workspace file; put sections and field names in finding'
            categories.add(item['category'])
            evidence.append(dict(item, sha256=sha(path)))
        context = read(workspace / 'context.json')
        required = ({'evaluation', 'source_ir'} if context.get('evidence_mode') == 'compile-repair'
                    else {'paired_profiling', 'simulator', 'source_ir'})
        assert required <= categories
        record = dict(timestamp=datetime.now(timezone.utc).isoformat(), diagnosis=diagnosis,
            diagnosis_sha256=sha(workspace / 'diagnosis.json'),
            evidence=evidence, input_files=initial, decision_tool_sha256=sha(Path(__file__)))
        destination = call / 'before-edit-decision.json'
        assert not destination.exists(), 'Decision already recorded for this invocation'
        save(destination, record)
        print(json.dumps(dict(status='decision_recorded', sha256=sha(destination))))
    else:
        decision = read(call / 'before-edit-decision.json')
        assert decision['input_files'] == initial
        assert diagnosis == decision['diagnosis'], 'Diagnosis changed after decision; start a new invocation from the retained input'
        if 'diagnosis_sha256' in decision:
            assert sha(workspace / 'diagnosis.json') == decision['diagnosis_sha256'], 'Registered diagnosis bytes changed'
        changed = [name for name in editable if name != 'diagnosis.json' and actual[name] != initial[name]]
        if action == 'no-change':
            assert not changed, 'No-change handoff requires the original IR bytes'
            save(call / 'no-change.json', dict(status='model_no_change', source_sha256=actual['frozen.py'],
                decision_sha256=sha(call / 'before-edit-decision.json'), output_files=actual))
            print(json.dumps(dict(status='model_no_change')))
            return
        assert changed, 'Use the no-change bridge for an explicit unchanged-IR judgment'
        save(call / 'submission.json', dict(status='awaiting_ir_review', source_sha256=actual['frozen.py'],
            changed_ir=changed, output_files=actual, decision_sha256=sha(call / 'before-edit-decision.json')))
        print(json.dumps(dict(status='awaiting_ir_review', changed_ir=changed)))


def generate(run, invocation):
    workspace = run / 'workspace'
    call = run / 'agent-calls' / invocation
    call.mkdir(parents=True, exist_ok=False)
    context = read(workspace / 'context.json')
    editable_stage = context.get('editable_ir_stage', 'ttir')
    assert editable_stage in {'ttir', 'bcmlir', 'hivm'}
    diagnosis = workspace / 'diagnosis.json'
    (call / 'input-diagnosis.json').write_bytes(diagnosis.read_bytes())
    save(diagnosis, dict(source_sha256=sha(workspace / 'frozen.py'), selected_layer='single_operator_ir',
        evidence=[], hypothesis='', alternatives=[], change='', expected='', validation='', unknown=[]))
    initial = files(workspace)
    save(call / 'input-manifest.json', initial)
    # Preserve exact evidence and host feedback bytes before the model reads them.
    shutil.copytree(workspace, call / 'input-workspace',
                    ignore=shutil.ignore_patterns('__pycache__'))
    assert files(call / 'input-workspace') == initial
    for row in read(workspace / 'ir-inventory.json')['entries']:
        original = workspace / row['path']
        destination = call / 'input-ir' / row['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original.read_bytes())
    bridge_prefix = shlex.join([sys.executable, str(Path(__file__).resolve()), '--run', str(run),
        '--invocation', invocation])
    prompt = """Research single-operator IR performance for the input identified by context.json; keep frozen.py byte-identical.
Continue from the qualified best by default. If context explicitly marks an unretained repair branch, repair that parent while comparing acceptance to the retained best.
You own actual IR changes and numerical/compiler repairs; Codex owns evidence and scheduling. Read only this workspace; do not invoke other agents.
First read context.json, profiling.json, hotspots.json, timeline.json, ir-inventory.json, and frozen.py.
Inspect actual TTIR in ir/ by kernel symbol, loops, copies, and layouts. Input files are actual compilation outputs of the current best.
If lowered-inventory.json exists, inspect actual ttadapter/bcmlir in compiled-readonly/ by symbol.
Check whether the backend already eliminated suspected redundancy. Keep these low-level artifacts frozen; edit only manifest-allowed TTIR.
Respect simulator clock qualification, observation scope, and unreported lanes. Hotspot regions and wait/idle evidence are core inputs.
Pipeline times can overlap. Hotspot rankings identify mechanisms; critical-path, device-wave, and bandwidth claims need independent evidence.
Record coverage and normalization separately before/after. Verify execution coverage before comparing dynamic counts; unreported lanes remain unknown.
Each score constrains its actual candidate and measurement scope; broader exclusions remain hypotheses.
Target improvement beyond the best; repairing regressions, changing compiler options, and source rewrites retain separate identities.
Choose a concrete costly compiler defect in current IR. State what computation/layout should be optimized and a measurable prediction.
Preserve math, special values, boundaries, aliasing, and FP requirements. No fitting test data, output caching, or external compute.
First edit only diagnosis.json: source_sha256, selected_layer=single_operator_ir, evidence, hypothesis,
alternatives, change, expected, validation, layer_reason, simulator_incremental_judgment.
layer_reason explains Triton expressibility, fixed-source experiment purpose, and IR control point; identify source alternatives when expressible.
expected lists instruction, dynamic-count, wait, and resource changes; post-change host replay checks each prediction and discrepancy.
Each evidence entry has path/category/role/finding; categories are paired_profiling/simulator/source_ir.
Cite concrete profiling, hotspot-region, and pipeline observations actually read.
Call the decision bridge below successfully before IR edits. Use only Edit on diagnosis.json and manifest-listed ir/*.ttir.
After registration keep diagnosis.json byte-identical and edit only the registered mechanism. Put new observations in the final reply.
For unchanged IR register the decision, call no-change, and end; the host continues searching.
If the mechanism changes after registration, preserve this invocation and request a new registered invocation.
Source, manifests, context, and evidence stay frozen. No Write tool or shell file writes. Choose one independently verifiable mechanism per invocation.
Edit multiple specializations if required by that mechanism, preserving function ABI. Avoid repeated complete reads of parsable original IR.
If backend elimination already occurred, explain and hand off unchanged. Otherwise apply focused edits.
After one change call submit; at awaiting_ir_review preserve files and end the invocation.
The host reviews semantics and runs full KGS/performance; documentation, preflight, or suggestions are not performance results.
No total round/token/cost/time search limit; this handoff is a review/full-evaluation boundary.
Write new diagnoses in English. Tool commands:
"""
    prompt += 'Use affirmative, concrete English for observations, scope, and next actions. Historical wording checks remain for replay compatibility.\n'
    prompt += bridge_prefix + ' decision\n' + bridge_prefix + ' submit\n' + bridge_prefix + ' no-change\n'
    if editable_stage == 'bcmlir':
        prompt = prompt.replace('actual TTIR', 'actual BCMLIR')
        prompt = prompt.replace('edit only manifest-allowed TTIR', 'edit only manifest-allowed BCMLIR')
        prompt = prompt.replace('ir/*.ttir', 'ir/*.bcmlir')
    if editable_stage == 'hivm':
        prompt = prompt.replace('actual TTIR', 'actual HIVM')
        prompt = prompt.replace('edit only manifest-allowed TTIR', 'edit only manifest-allowed HIVM')
        prompt = prompt.replace('ir/*.ttir', 'ir/*.mlir')
        prompt += ('\nEdit actual compiler-exported HIVM; original input is bound to measured binary through independent backend continuation.'
                   'Review explicit buffer addresses, aliases, cross-iteration lifetimes, event pairing, and cross-core synchronization strictly; '
                   'preserve function ABI and unrelated cores. The host reviews each change before compilation/hardware evaluation.\n')
    if context.get('evidence_mode') == 'compile-repair':
        prompt = prompt.replace('Input files are actual compilation outputs of the current best',
            'Input files are actual compiler failures from the exploratory branch; historical retained-best material is separately identified')
        prompt = prompt.replace('categories are paired_profiling/simulator/source_ir',
            'categories are evaluation/source_ir/paired_profiling/simulator')
        prompt += ('\nThis invocation repairs compilation of an explicitly unretained source branch. Read feedback.json and codex-brief.json first. '
            'Target IR actually failed compilation; profiling/hotspots/timeline are historical-version material.'
            'Pre-change evidence must include evaluation and source_ir; record historical sources accurately. '
            'Keep current source byte-identical; compilation repair, same-source IR gain, and improvement beyond retained best are separate acceptance questions.'
            'This invocation submits IR only; the host runs CPU compilation and original four-case hardware checks afterward.\n')
    if (workspace / 'feedback.json').exists():
        prompt += 'First read actual formal/supplementary results in feedback.json, preserve source/IR identities, then choose the next action.\n'
    if (workspace / 'codex-brief.json').exists():
        prompt += "Read host evidence analysis and directions in codex-brief.json, inspect actual IR, and choose implementation yourself; record Codex's analysis contribution.\n"
        requested = read(workspace / 'codex-brief.json').get('requested_mechanism')
        if requested:
            prompt += ('This invocation tests the following mechanism; you choose implementation. If evidence/semantics prevents it, '
                       'register the reason and hand off via no-change; leave other mechanisms for separate invocations:\n' + requested + '\n')
    if (workspace / 'validation-policy.json').exists():
        prompt += 'Read validation-policy.json; reuse fixed full contracts, checks, and existing diagnostics for registered risks. Use ABBA only for measurement ambiguity. Tools record SHA automatically; candidate authors focus on mechanism/implementation.\n'
    (call / 'prompt.md').write_text(prompt, encoding='utf8')
    settings = read(run / 'model-settings.json')
    environment = dict(os.environ)
    environment.update({key: str(value) for key, value in settings.get('env', {}).items()})
    environment.update(CLAUDE_AUTOCOMPACT_PCT_OVERRIDE='80',
        CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(settings.get('max_output_tokens', 16384)),
        MAX_THINKING_TOKENS=str(settings.get('thinking_tokens', 4096)))
    if 'ANTHROPIC_API_KEY' not in environment and environment.get('ANTHROPIC_AUTH_TOKEN'):
        environment['ANTHROPIC_API_KEY'] = environment['ANTHROPIC_AUTH_TOKEN']
    runtime_settings = dict(env={key: environment[key] for key in ['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE',
        'CLAUDE_CODE_MAX_OUTPUT_TOKENS', 'MAX_THINKING_TOKENS']})
    save(call / 'runtime-settings.json', runtime_settings)
    command = ['claude', '-p', '--bare', '--output-format', 'stream-json', '--verbose',
        '--tools', 'Read,Grep,Glob,Edit,Bash', '--allowedTools', 'Read', 'Grep', 'Glob', 'Edit',
        'Bash(' + bridge_prefix + ' *)', '--permission-mode', 'acceptEdits',
        '--no-session-persistence', '--settings', str(call / 'runtime-settings.json')]
    if settings.get('model'):
        command += ['--model', settings['model']]
    if settings.get('effort'):
        command += ['--effort', settings['effort']]
    save(call / 'host-invocation.json', dict(argv=command, model=settings.get('model'),
        started_at=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(Path(__file__)),
        prompt_sha256=sha(call / 'prompt.md'), source_sha256=initial['frozen.py'],
        secret_environment_recorded=False, generation_limits=dict(max_turns=None, deadline=None)))
    save(run / 'status.json', dict(status='model_generation_running', invocation=invocation,
         source_sha256=initial['frozen.py'], host_invocation=str(call / 'host-invocation.json')))
    with (call / 'agent-events.jsonl').open('xb') as out, (call / 'agent-stderr.log').open('xb') as err:
        process = subprocess.run(command, input=prompt.encode(), stdout=out, stderr=err,
            env=environment, cwd=workspace)
    result = {}
    for line in (call / 'agent-events.jsonl').read_bytes().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'result':
            result = event
    save(call / 'model-usage.json', dict(returncode=process.returncode,
        result={key: result.get(key) for key in ['subtype', 'is_error', 'num_turns', 'duration_ms',
            'usage', 'modelUsage', 'total_cost_usd', 'result']}))
    _, _, _, actual, _ = checked(run, invocation)
    for row in read(workspace / 'ir-inventory.json')['entries']:
        path = workspace / row['path']
        destination = call / 'output-ir' / row['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    (call / 'output-diagnosis.json').write_bytes(diagnosis.read_bytes())
    error = process.returncode != 0 or result.get('is_error')
    status = 'requires_attention' if error else ('awaiting_ir_review' if (call / 'submission.json').exists()
        else 'model_no_change' if (call / 'no-change.json').exists() else 'model_no_submission')
    save(run / 'status.json', dict(status=status, invocation=invocation, source_sha256=actual['frozen.py']))
    print(json.dumps(dict(status=status, invocation=invocation)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('action', choices=['generate', 'decision', 'submit', 'no-change'])
    args = parser.parse_args()
    if args.action == 'generate':
        handoff = args.run / ('generation-handoff-' + args.invocation + '.json')
        if handoff.exists():
            record = read(handoff)
            target = Path(record['run']).resolve()
            assert target.is_relative_to(args.run.resolve().parent)
            assert read(target / 'config.json')['source_sha256'] == read(args.run / 'config.json')['source_sha256']
            assert record['invocation'] == args.invocation
            print(json.dumps(dict(status='generation_handed_off', **record)))
            return
        generate(args.run, args.invocation)
    else:
        bridge(args.run, args.invocation, args.action)


if __name__ == '__main__':
    main()
