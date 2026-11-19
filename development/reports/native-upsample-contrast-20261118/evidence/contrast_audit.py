"""Audit preserved decision/evidence ordering without treating claims as causal proof."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_call(run, call, simulator):
    parent=digest(call/'input-source.py');output=call/'output-source.py'
    decisions=[]
    for p in (run/'trace/decisions').glob('*.json'):
        d=json.loads(p.read_bytes())
        if d['round_id']==call.name:decisions.append(d)
    findings=[];ids={d['id'] for d in decisions}
    if not decisions:findings.append('no recorded decision yet')
    for d in decisions:
        if d['input_source']['sha256']!=parent:findings.append('decision differs from call input source')
        for e in d['evidence']:
            ref=e['reference'];p=run/ref['path']
            if not p.exists() or digest(p)!=ref['sha256']:findings.append('missing or changed frozen evidence: '+ref['path'])
        sim=[e for e in d['evidence'] if e['category']=='simulator']
        if simulator:
            if not sim:findings.append('simulator evidence not cited')
            for e in sim:
                p=run/e['reference']['path'];v=json.loads(p.read_bytes())
                if v.get('source_sha256')!=parent:findings.append('simulator bound to different source')
                if v.get('status') not in ['qualified_observation','completed']:findings.append('simulator not qualified')
                if v.get('status')=='qualified_observation' and (not v.get('reports') or not v.get('capture_scope')):
                    findings.append('partial simulator scope missing')
        elif sim:findings.append('control decision cites simulator')
    read_requests=set();read_returns=[];decision_returns=[];edits=[];unclassified=[]
    pending_edits={};replayed=(call/'input-source.py').read_bytes().decode('utf8');replay_errors=[]
    stream=call/'agent-events.jsonl'
    if stream.exists():
        for order,line in enumerate(stream.read_bytes().splitlines()):
            try:event=json.loads(line)
            except ValueError:continue
            message=event.get('message',{})
            if not isinstance(message,dict):continue
            for b in message.get('content',[]):
                if not isinstance(b,dict):continue
                if b.get('type')=='tool_use':
                    x=b.get('input',{});path=Path(x.get('file_path',''))
                    if b.get('name')=='Read' and path.name=='simulator-evidence.json':read_requests.add(b['id'])
                    if b.get('name') in ['Edit','Write'] and path.name=='candidate.py':
                        edits.append(order);pending_edits[b['id']]=(b['name'],x)
                    if b.get('name')=='Bash':
                        command=x.get('command','')
                        if 'candidate.py' in command and any(s in command for s in ['write_text','write_bytes','sed -i','> candidate','cp ']):
                            unclassified.append(dict(event_order=order,command=command))
                if b.get('type')=='tool_result' and not b.get('is_error'):
                    change=pending_edits.pop(b.get('tool_use_id'),None)
                    if change:
                        kind,x=change
                        if kind=='Write':replayed=x['content']
                        elif x.get('old_string') in replayed:
                            replayed=replayed.replace(x['old_string'],x['new_string'],-1 if x.get('replace_all') else 1)
                        else:replay_errors.append('successful Edit old_string not found during replay')
                    if b.get('tool_use_id') in read_requests:read_returns.append(order)
                    encoded=json.dumps(b,ensure_ascii=False)
                    if any(i in encoded for i in ids):decision_returns.append(order)
    first_edit=min(edits) if edits else None
    if simulator and decision_returns and not any(x<min(decision_returns) for x in read_returns):
        findings.append('simulator read before recorded decision not established')
    if first_edit is not None:
        if not any(x<first_edit for x in decision_returns):findings.append('successful recorded decision before first edit not established')
        if simulator and not any(x<first_edit for x in read_returns):findings.append('successful simulator read before first edit not established')
    if unclassified:findings.append('shell source changes need manual review')
    replay_matches=not replay_errors and replayed.encode('utf8')==output.read_bytes() if output.exists() else None
    if replay_matches is False:findings.append('model edit replay does not reproduce frozen output')
    return dict(round=call.name,input_sha256=parent,output_sha256=digest(output) if output.exists() else None,
                decisions=len(decisions),first_edit_event=first_edit,simulator_read_return_events=read_returns,
                decision_return_events=decision_returns,unclassified_shell_changes=unclassified,findings=findings,
                stream_sha256=digest(stream) if stream.exists() else None,
                edit_replay_matches_output=replay_matches,edit_replay_errors=replay_errors,
                replay_sha256=hashlib.sha256(replayed.encode('utf8')).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();r=a.study
    experiment=json.loads((r/'experiment.json').read_bytes())
    out=dict(recorded_at=datetime.now(timezone.utc).isoformat(),scope='Preserved evidence and edit ordering audit; not proof of causal contribution, statistical significance, or full goal completion.',arms={})
    for arm in ['no-simulator','with-simulator']:
        run=r/arm;rows=[audit_call(run,c,arm=='with-simulator') for c in sorted((run/'agent-calls').iterdir()) if c.is_dir()]
        ancestors={experiment['upstream_source_sha256']}
        for row in rows:
            if row['input_sha256'] not in ancestors:row['findings'].append('input source not an earlier source in this arm')
            if row['output_sha256']:ancestors.add(row['output_sha256'])
        out['arms'][arm]=dict(calls=rows,within_generation_budget=len(rows)<=experiment['budget_per_arm']['generation_calls'],
            all_eight_calls_finished=len(rows)==8 and all(x['output_sha256'] for x in rows),
            original_start=bool(rows) and rows[0]['input_sha256']==experiment['upstream_source_sha256'])
    folder=r/'audits';folder.mkdir(exist_ok=True)
    path=folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'audit':str(path),'findings':{arm:[(x['round'],x['findings']) for x in v['calls'] if x['findings']] for arm,v in out['arms'].items()}}))


if __name__=='__main__':main()
