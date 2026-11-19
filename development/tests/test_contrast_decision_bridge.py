"""Simulator-arm decisions require both source binding and a successful tool read."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from development.tools.research.contrast_decision_bridge import main


class DecisionAdmission(unittest.TestCase):
    def run_case(self, read_ok, stale=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);w=root/'workspace';w.mkdir()
            source=b'def run(x): return x\n';(w/'candidate.py').write_bytes(source)
            digest=hashlib.sha256(source).hexdigest()
            report={'source_sha256':'wrong' if stale else digest,'status':'completed'}
            raw=json.dumps(report).encode();(w/'simulator-evidence.json').write_bytes(raw)
            call=root/'agent-calls/round-01';call.mkdir(parents=True)
            events=[{'type':'system','message':'runtime notification'},
                    {'message':{'content':[{'type':'tool_use','name':'Read','id':'read1','input':{'file_path':str(w/'simulator-evidence.json')}}]}},
                    {'message':{'content':[{'type':'tool_result','tool_use_id':'read1','is_error':not read_ok}]}}]
            (call/'agent-events.jsonl').write_text('\n'.join(map(json.dumps,events)))
            c={'run_directory':str(root),'trace_round_id':'round-01','required_simulator_evidence':{'path':'simulator-evidence.json','sha256':hashlib.sha256(raw).hexdigest()}}
            (root/'config.json').write_text(json.dumps(c))
            d={'source_sha256':digest,'hypothesis':'test','change':'test','expected':'test','validation':'test','selected_layer':'Triton','layer_reason':'test','simulator_incremental_judgment':'test',
               'evidence':[{'path':str(w/'simulator-evidence.json'),'category':'simulator','role':'mechanism'}]}
            (w/'diagnosis.json').write_text(json.dumps(d))
            with patch.object(sys,'argv',['bridge','--config',str(root/'config.json'),'decision','--record','diagnosis.json']):main()
            return len(list((root/'trace/decisions').glob('*.json')))

    def test_successful_read_admitted(self):self.assertEqual(self.run_case(True),1)
    def test_failed_read_rejected(self):
        with self.assertRaisesRegex(AssertionError,'successfully read'):self.run_case(False)
    def test_stale_report_rejected(self):
        with self.assertRaises(AssertionError):self.run_case(True,True)


if __name__=='__main__':unittest.main()
