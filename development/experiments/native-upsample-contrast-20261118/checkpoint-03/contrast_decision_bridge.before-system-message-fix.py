"""Require successful simulator evidence reads before experimental-arm decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('action',choices=['decision','preflight'])
    p.add_argument('--record',type=Path)
    p.add_argument('--source',type=Path)
    a=p.parse_args();c=json.loads(a.config.read_bytes());run=Path(c['run_directory']);workspace=run/'workspace'
    if a.action=='decision':
        path=a.record if a.record.is_absolute() else workspace/a.record
        d=json.loads(path.read_bytes())
        for field in ['selected_layer','layer_reason','simulator_incremental_judgment']:
            assert isinstance(d.get(field),str) and d[field].strip(),'missing '+field
        required=c.get('required_simulator_evidence')
        if required:
            report=workspace/required['path'];raw=report.read_bytes();info=json.loads(raw)
            assert hashlib.sha256(raw).hexdigest()==required['sha256']
            current=hashlib.sha256((workspace/'candidate.py').read_bytes()).hexdigest()
            assert info['source_sha256']==current and info['status'] in {'completed','qualified_observation'}
            if info['status']=='qualified_observation':
                assert info.get('reports') and info.get('capture_scope'), 'partial observation needs explicit scope'
            assert any(e['category']=='simulator' and Path(e['path']).name==report.name for e in d.get('evidence',[])), 'cite the current simulator evidence'
            stream=run/'agent-calls'/c['trace_round_id']/'agent-events.jsonl'
            reads=set();succeeded=set()
            for line in stream.read_bytes().splitlines():
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                for block in event.get('message',{}).get('content',[]):
                    if not isinstance(block,dict):continue
                    if block.get('type')=='tool_use' and block.get('name')=='Read':
                        file=Path(block.get('input',{}).get('file_path',''))
                        resolved=file if file.is_absolute() else workspace/file
                        if resolved.resolve()==report.resolve():reads.add(block['id'])
                    if block.get('type')=='tool_result' and not block.get('is_error',False):succeeded.add(block.get('tool_use_id'))
            assert reads&succeeded,'simulator report must be successfully read before decision'
        else:
            assert not any(e['category']=='simulator' for e in d.get('evidence',[])),'control arm has no simulator evidence'
        from hgkda.runtime.trace import record_decision, emit
        result=record_decision(c,d)
        emit(c,'research_layer_decision',selected_layer=d['selected_layer'],layer_reason=d['layer_reason'],
             simulator_incremental_judgment=d['simulator_incremental_judgment'],decision_id=result['decision_id'])
        print(json.dumps(result));return
    from hgkda.runtime.agent_tools import main as delegated
    delegated()


if __name__=='__main__':main()
