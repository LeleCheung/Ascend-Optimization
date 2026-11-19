import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hgkda.timeline import relative_timeline


class RelativeTimelineTests(unittest.TestCase):
    def test_cli_dispatch_preserves_relative_scope(self):
        from hgkda.cli import parser, dispatch
        listing,raw,binding=self.capture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'listing.json').write_text(json.dumps(listing))
            (root/'trace.json').write_bytes(raw)
            (root/'binding.json').write_text(json.dumps(binding))
            args=parser().parse_args(['timeline','--listing',str(root/'listing.json'),
                '--trace',str(root/'trace.json'),'--binding',str(root/'binding.json')])
            result,inputs=dispatch(args)
            self.assertEqual(result['timeline']['time_unit'],'fraction')
            self.assertFalse(result['timeline']['absolute_time_qualified'])
            self.assertEqual(len(inputs),3)

    def capture(self,scale=1,shift=0):
        events=[dict(ph='M',name='process_name',pid=i,args=dict(name=name))
                for i,name in enumerate(['MTE2','SCALAR','VECTOR'])]
        events += [dict(ph='X',pid=pid,name=name,ts=shift+start*scale,dur=duration*scale,args=dict(pc_addr='0x10'))
                   for pid,name,start,duration in [(0,'LOAD',0,2),(0,'WAIT_FLAG',1,3),(1,'ADD',0,6)]]
        raw=json.dumps(dict(traceEvents=events)).encode()
        listing=dict(source_sha256='source',case={'shape':[8]},kernels=[dict(name='k',cores=[dict(name='c',instructions=[])])])
        binding=dict(source_sha256='source',case={'shape':[8]},kernel='k',core='c',raw_trace_sha256=hashlib.sha256(raw).hexdigest())
        return listing,raw,binding

    def test_partition_and_explicit_empty_lane(self):
        listing,raw,binding=self.capture()
        t=relative_timeline(listing,raw,binding=binding)['timeline']
        rows={x['pipeline']:x for x in t['pipelines']}
        self.assertAlmostEqual(rows['MTE2']['productive_ratio'],1/3)
        self.assertAlmostEqual(rows['MTE2']['wait_only_ratio'],1/3)
        self.assertAlmostEqual(rows['MTE2']['idle_ratio'],1/3)
        self.assertEqual(rows['VECTOR']['idle_ratio'],1)
        self.assertFalse(t['absolute_time_qualified'])

    def test_clock_scale_and_offset_cancel(self):
        outputs=[]
        for scale,shift in [(1,0),(1000,123456),(0.001,1.5)]:
            listing,raw,binding=self.capture(scale,shift)
            outputs.append(relative_timeline(listing,raw,binding=binding)['timeline']['pipelines'])
        for rows in outputs[1:]:
            for old,new in zip(outputs[0],rows):
                for key in ['productive_ratio','wait_only_ratio','idle_ratio']:
                    self.assertAlmostEqual(old[key],new[key])

    def test_rejects_changed_evidence_identity(self):
        listing,raw,binding=self.capture();binding['source_sha256']='other'
        with self.assertRaises(ValueError):relative_timeline(listing,raw,binding=binding)
        listing,raw,binding=self.capture()
        with self.assertRaises(ValueError):relative_timeline(listing,raw+b' ',binding=binding)
