import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from development.tools.research.contrast_audit import audit_call


class EvidenceOrdering(unittest.TestCase):
    def inspect(self, late_read=False, wrong_source=False, extra_edit=False):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);call=run/'agent-calls/round-01';call.mkdir(parents=True)
            (call/'input-source.py').write_bytes(b'parent');(call/'output-source.py').write_bytes(b'child extra' if extra_edit else b'child')
            sha=hashlib.sha256(b'parent').hexdigest()
            report={'status':'qualified_observation','source_sha256':'wrong' if wrong_source else sha,
                    'reports':[{'core':'core0'}],'capture_scope':'partial window'}
            evidence=run/'trace/evidence/sim.json';evidence.parent.mkdir(parents=True)
            evidence.write_text(json.dumps(report))
            decision={'id':'decision-123','round_id':'round-01','input_source':{'sha256':sha},
                'evidence':[{'category':'simulator','reference':{'path':'trace/evidence/sim.json',
                             'sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}}]}
            folder=run/'trace/decisions';folder.mkdir();(folder/'d.json').write_text(json.dumps(decision))
            read={'type':'tool_use','name':'Read','id':'read-1','input':{'file_path':'simulator-evidence.json'}}
            returned={'type':'tool_result','tool_use_id':'read-1','content':'report'}
            recorded={'type':'tool_result','tool_use_id':'decision-tool','content':'decision-123'}
            edit={'type':'tool_use','name':'Edit','id':'edit-1','input':{'file_path':'candidate.py','old_string':'parent','new_string':'child'}}
            edit_return={'type':'tool_result','tool_use_id':'edit-1','content':'applied'}
            events=[read,recorded,edit,edit_return,returned] if late_read else [read,returned,recorded,edit,edit_return]
            (call/'agent-events.jsonl').write_text('\n'.join(json.dumps({'message':{'content':[e]}}) for e in events))
            return audit_call(run,call,True)['findings']

    def test_read_then_decision_then_edit(self):
        self.assertEqual(self.inspect(),[])

    def test_late_successful_read_is_not_pre_edit_evidence(self):
        self.assertTrue(any('before first edit' in s for s in self.inspect(late_read=True)))

    def test_valid_report_from_other_source_is_rejected(self):
        self.assertIn('simulator bound to different source',self.inspect(wrong_source=True))

    def test_unrecorded_source_edit_is_detected(self):
        self.assertIn('model edit replay does not reproduce frozen output',self.inspect(extra_edit=True))


if __name__=='__main__':unittest.main()
