"""Let the host model load an isolated run-local skill without recapturing data or running formal evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.resources import files as resource_files
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def target_met(workspace, digest):
    """Accept stop feedback only when bound to current bytes with complete formal and supplementary results."""
    from hgkda.artifacts import unwrap
    for name in ['next-performance-feedback.json', 'performance-feedback.json']:
        path=workspace/name
        if not path.exists():continue
        feedback=unwrap(read(path))
        return (feedback.get('source_sha256')==digest
                and feedback.get('decision')=='target_met'
                and feedback.get('accepted') is True)
    return False


def generation_limits(max_turns=0, timeout_seconds=0):
    """Zero means no workflow-imposed generation budget; host limits still apply."""
    if max_turns < 0 or timeout_seconds < 0:
        raise ValueError('generation limits must be nonnegative')
    return (['--max-turns', str(max_turns)] if max_turns else [], timeout_seconds or None)


def should_finish(config, workspace, digest):
    return config.get('optimization_policy', 'performance_first') == 'target_only' and target_met(workspace, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model-settings', type=Path, required=True)
    parser.add_argument('--invocation', default='first')
    parser.add_argument('--max-turns', type=int, default=0, help='0: no workflow turn cap (default)')
    parser.add_argument('--generation-timeout', type=int, default=0, help='seconds; 0: no generation deadline (default)')
    parser.add_argument('--prompt-file', type=Path)
    args = parser.parse_args()
    turn_flags, generation_timeout = generation_limits(args.max_turns, args.generation_timeout)
    config = read(args.config)
    run = Path(config['run_directory'])
    repo = Path(config.get('repository', config['run_directory']))
    workspace = run/'workspace'
    initial_source=workspace/'candidate.py'
    initial_digest=hashlib.sha256(initial_source.read_bytes()).hexdigest() if initial_source.exists() else None
    if config.get('context_kind')=='operator_transfer':
        from hgkda.runtime.transfer_context import retire_previous
        active=workspace/config.get('active_source','starter.py')
        retire_previous(workspace,hashlib.sha256(active.read_bytes()).hexdigest())
    call = run/'agent-calls'/args.invocation
    call.mkdir(parents=True, exist_ok=False)
    active_input=workspace/config.get('active_source','starter.py')
    (call/'input-source.py').write_bytes(active_input.read_bytes())
    diagnosis=workspace/'diagnosis.json'
    if diagnosis.exists():(call/'input-diagnosis.json').write_bytes(diagnosis.read_bytes())
    # Preserve prior diagnoses; give each round a short empty template instead of an ever-growing history document.
    save(diagnosis,dict(schema_version='hgkda.diagnosis/v1',source_sha256=hashlib.sha256(active_input.read_bytes()).hexdigest(),evidence=[],hypothesis='',
                        change='',expected='',validation='',unknown=[]))
    if config.get('method_manifest'):(call/'method-manifest.json').write_bytes(Path(config['method_manifest']).read_bytes())
    call_config=dict(config)
    call_config['trace_round_id']=args.invocation
    if config.get('method_manifest'):call_config['method_manifest']=str(call/'method-manifest.json')
    save(call/'config.json',call_config)
    # Freeze method identity for this model invocation; later controller run/config switches do not change it.
    # Stable workspace paths also keep frozen-stage invocation prompts byte-identical.
    execution_config=workspace/'agent-config.json'
    save(execution_config,call_config)
    config=call_config
    operator=config.get('operator',config['job']['operator'])
    provenance_path=workspace/'starter-provenance.json'
    supplied_best=read(provenance_path).get('prior_best_source_provided',False) if provenance_path.exists() else False
    if config.get('context_kind')=='operator_transfer':
        from hgkda.runtime.transfer_context import build
        # Refresh on the host first to prevent stale context if the model omits the context command.
        brief=build(config)
        save(call/'initial-context.json',brief)
        (call/'initial-knowledge.json').write_bytes((workspace/'knowledge.json').read_bytes())
    if config.get('context_kind')!='operator_transfer' and config.get('goal_kind')!='performance_recovery_with_formal_feedback':
        assert all((Path(config['paired_directory'])/side/'collection.json').exists()
                   for side in ['candidate', 'reference'])
    tool_path=Path(config.get('agent_tool_path',Path(__file__).with_name('agent_tools.py')))
    tool_command = shlex.join([sys.executable, str(tool_path),
                              '--config', str(execution_config.resolve())])
    save(workspace/'task.json', dict(operator=operator, source=config.get('active_source','starter.py'),
         baseline_source='starter.py',
         required_formal_cases=config.get('formal_cases',36),
         required_supplementary_each=config['job']['checks']['supplementary']['previous_total'], tool_command=tool_command,
         tool_actions=dict(context=tool_command+' context',
                           preflight=tool_command+' preflight --source '+str(workspace/'candidate.py')),
         environment='Host runs only the tool bridge; device dependencies run in the container, with no Triton probing in the host virtual environment',
         initial_evidence_policy='If the original starter has a non-Triton device fallback, the model must first build a compliant candidate, then capture IR and paired profiling; missing captures do not block initial adaptation',
         feedback_file=('next-performance-feedback.json' if (workspace/'next-performance-feedback.json').exists()
                        else 'performance-feedback.json'),
         optimization_stop=('Performance first: acceptance thresholds, default rounds, time, or cost do not end search; after model return the host continues evidence collection, evaluation, and optimization'
                            if config.get('optimization_policy','performance_first')=='performance_first'
                            else 'Stop after the explicit user goal and supplementary checks are satisfied'),
         generation_limits=dict(max_turns=args.max_turns or None, timeout_seconds=generation_timeout),
         public_entrypoint=config.get('public_entrypoint', 'run; arguments follow contract.json'),
         candidate_file='candidate.py',device_probe='Tool bridge preflights candidate implementations only, not arbitrary test scripts; do not submit starter.py or probe scripts',
         current_best_source_read_allowed=supplied_best,phase=config.get('phase','development'),
         method_manifest_sha256=(hashlib.sha256(Path(config['method_manifest']).read_bytes()).hexdigest() if config.get('method_manifest') else None),
         experiment_goal=config.get('experiment_goal'),
         starter_provenance=read(workspace/'starter-provenance.json') if (workspace/'starter-provenance.json').exists() else None,
         target=config.get('generation_target','Continuously improve relative-to-CANN and per-case performance under the fixed contract; acceptance is not the default search stop condition')))
    prompt = resource_files('hgkda').joinpath('data/prompts/agent.md').read_text(encoding='utf-8')+tool_command
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding='utf8')+'\nTool bridge commands:'+tool_command
    if config.get('experiment_goal'):
        prompt=prompt.replace('If formal results bound to current source meet the task threshold and all required supplementary checks pass, preserve candidate.py bytes, record the conclusion in diagnosis.json, and stop; do not change source unconditionally for a higher score.',
            'This round is the targeted method experiment specified by experiment_goal in task.json: prior acceptance does not replace a new change and prediction check. Change only its mechanism, retain other validated paths, then hand off for read-only review and formal evaluation.')
        prompt=prompt.replace('Initially candidate.py is an exact copy of the fixed FlagGems starter; you must rewrite its entry point and implementation. Later invocations continue the generated candidate.',
            'See starter_provenance in task.json for the starting source; when a correct candidate exists, continue from its exact bytes with small changes rather than claiming fresh upstream generation.')
    if supplied_best:
        prompt=prompt.replace('do not read historical best source or other experiment directories','Read only delivery source and evidence explicitly provided in this workspace; do not read other experiment directories')
        prompt=prompt.replace('Initially candidate.py is an exact copy of the fixed FlagGems starter; you must rewrite its entry point and implementation. Later invocations continue the generated candidate.',
                              'Initial candidate.py comes from an explicitly supplied delivery version; preserve provenance and do not describe edits as generation from upstream from scratch.')
    prompt=prompt.replace('public entry point run(A),','follow contract.json for the public entry point and arguments,')
    if config.get('optimization_policy','performance_first')=='performance_first':
        prompt=prompt.replace('If formal results bound to current source meet the task threshold and all required supplementary checks pass, preserve candidate.py bytes, record the conclusion in diagnosis.json, and stop; do not change source unconditionally for a higher score.',
            'Acceptance qualifies a candidate for retention but does not end search. Continue using per-case losses, compiled output, and hardware evidence; preserve the validated best and change hypotheses, methods, or modification levels after failures or temporary stagnation.')
        prompt+='\nPerformance-first policy: do not stop for default rounds, elapsed time, cost, or prior acceptance. Each review/full-evaluation handoff is a stage boundary; the host continues from real feedback. Preserve source and evidence for every candidate and never skip validation to save calls.\n'
    prompt += """
Before each concrete change, record current source SHA, referenced evidence, hypothesis, proposed edit, expected effects, and validation in diagnosis.json. Then run tool-bridge decision --record diagnosis.json to persist the pre-change decision. Evidence items contain path, category, role, and finding. category is paired_profiling/simulator/source_ir/roofline/knowledge/evaluation/probe/other; role is trigger/disambiguation/mechanism/background. Do not claim simulator use when it was absent; reading multiple tools does not establish causality. Preserve compile failures, regressions, and unchanged-source outcomes. One mechanism can span several small edits without lengthy analysis per replacement; update observations and unresolved questions after formal results.
"""
    (workspace/'prompt.md').write_text(prompt, encoding='utf8')
    (call/'prompt.md').write_text(prompt, encoding='utf8')
    configuration = read(args.model_settings)
    output_tokens=configuration.get('max_output_tokens',16384)
    thinking_tokens=configuration.get('thinking_tokens',4096)
    environment = dict(os.environ)

    environment.update({key: str(value) for key, value in configuration.get('env', {}).items()})
    # Adjust only the inherited 1% compaction threshold for this subprocess, leaving host-global settings untouched.
    environment['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'] = '80'
    environment['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] = str(output_tokens)
    environment['MAX_THINKING_TOKENS'] = str(thinking_tokens)
    if 'ANTHROPIC_API_KEY' not in environment and environment.get('ANTHROPIC_AUTH_TOKEN'):
        environment['ANTHROPIC_API_KEY'] = environment['ANTHROPIC_AUTH_TOKEN']
    # Explicit settings override user settings.env so launch cannot restore the old 1% value.
    save(call/'runtime-settings.json', dict(env={
         'CLAUDE_AUTOCOMPACT_PCT_OVERRIDE': '80', 'CLAUDE_CODE_MAX_OUTPUT_TOKENS': str(output_tokens),
         'MAX_THINKING_TOKENS': str(thinking_tokens)}))
    command = ['claude', '-p', '--bare', '--output-format', 'stream-json', '--verbose',
               *turn_flags, '--tools', 'Read,Edit,Bash',
               '--allowedTools', 'Read', 'Edit',
               'Bash('+str(sys.executable)+' '+str(tool_path)+' *)',
               '--permission-mode', 'acceptEdits', '--no-session-persistence',
               '--settings', str(call/'runtime-settings.json')]
    if configuration.get('model'):
        command += ['--model', configuration['model']]
    if configuration.get('effort'):
        command += ['--effort',configuration['effort']]
    save(call/'host-invocation.json', dict(argv=command, model=configuration.get('model'),
         started_at=datetime.now(timezone.utc).isoformat(),
         secret_environment_recorded=False,
         generation_limits=dict(max_turns=args.max_turns or None, timeout_seconds=generation_timeout),
         runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         tool_bridge_sha256=hashlib.sha256(tool_path.read_bytes()).hexdigest(),
         config_sha256=hashlib.sha256((call/'config.json').read_bytes()).hexdigest(),
         prompt_sha256=hashlib.sha256(prompt.encode('utf8')).hexdigest(),
         context_settings=dict(autocompact_percent=80, max_output_tokens=output_tokens,
                               thinking_tokens=thinking_tokens,effort=configuration.get('effort')),
         skill_sha256=hashlib.sha256((workspace/'skills/hgkda/SKILL.md').read_bytes()).hexdigest()))
    save(call/'source-provenance.json',dict(input_source=active_input.name,
        input_sha256=hashlib.sha256(active_input.read_bytes()).hexdigest(),candidate_existed=initial_source.exists()))
    state = read(run/'controller-state.json')
    state.update(status='agent_generating_and_preflighting',
                 resumed_at=datetime.now(timezone.utc).isoformat())
    state.pop('error', None)
    state.pop('returncode', None)
    save(run/'controller-state.json', state)
    try:
        with (call/'agent-events.jsonl').open('xb') as out, (call/'agent-stderr.log').open('xb') as err:
            process = subprocess.run(command, input=prompt.encode(), stdout=out, stderr=err,
                                     env=environment, cwd=workspace, timeout=generation_timeout)
        result = {}
        for line in (call/'agent-events.jsonl').read_bytes().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('type') == 'result':
                result = record
        save(call/'model-usage.json', dict(returncode=process.returncode,
             result={key: result.get(key) for key in ['subtype', 'is_error', 'num_turns', 'duration_ms',
                     'usage', 'modelUsage', 'total_cost_usd', 'result']}))
        final_digest=hashlib.sha256(initial_source.read_bytes()).hexdigest() if initial_source.exists() else None
        model_error=result.get('is_error') or str(result.get('result','')).startswith('API Error:')
        previous=read(workspace/'latest-preflight.json') if (workspace/'latest-preflight.json').exists() else {}
        valid = (process.returncode == 0 and not model_error and final_digest!=initial_digest
                 and previous.get('source_sha256')==final_digest and previous.get('status') == 'passed')
        state.update(status='awaiting_source_review' if valid else 'requires_attention',
                     returncode=process.returncode)
        if process.returncode==0 and not model_error and final_digest==initial_digest and should_finish(config,workspace,final_digest):
            state['status']='generation_target_met'
        if not model_error and previous.get('source_sha256')==final_digest and previous.get('status')=='awaiting_source_review':
            state['status']='awaiting_preflight_source_review'
        if model_error:state['error']='model_result_error: '+str(result.get('result',''))[:500]
    except Exception as error:
        state.update(status='requires_attention', error=type(error).__name__+': '+str(error))
        raise
    finally:
        if initial_source.exists():
            (call/'output-source.py').write_bytes(initial_source.read_bytes())
            provenance=read(call/'source-provenance.json')
            provenance['output_sha256']=hashlib.sha256(initial_source.read_bytes()).hexdigest()
            save(call/'source-provenance.json',provenance)
        save(run/'controller-state.json', state)


if __name__ == '__main__':
    from hgkda.runtime.trace import traced_main
    traced_main(main, 'generate')
