import unittest
import tempfile
import json
import hashlib
from pathlib import Path
from hgkda.runtime.collect_simulator import select_kernel, freeze_autotune_replay


class SimulatorKernelSelectionTests(unittest.TestCase):
    def test_single_observed_symbol(self):
        self.assertEqual(select_kernel(['one', 'one']), 'one')

    def test_multikernel_requires_explicit_choice(self):
        with self.assertRaises(ValueError):
            select_kernel(['sort', 'combine'])
        self.assertEqual(select_kernel(['sort', 'combine'], 'combine'), 'combine')

    def test_unknown_or_empty_rejected(self):
        for names, choice in [([], None), (['sort'], 'other')]:
            with self.assertRaises(ValueError):
                select_kernel(names, choice)

    def test_replay_preserves_bytes_and_rejects_wrong_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pair=root/'pair';out=root/'out'
            (pair/'actual-ir').mkdir(parents=True);out.mkdir()
            source=pair/'actual-ir/autotune-cache.json'
            payload=dict(source_sha256='a'*64,case={'shape':[8,512]},entries=[dict(cache=[{'key':[8],'config':{'kwargs':{'BLOCK':64}}}])])
            raw=json.dumps(payload).encode();source.write_bytes(raw)
            values,digest=freeze_autotune_replay(pair,out,'a'*64,payload['case'])
            self.assertEqual(Path(values['autotune_replay']).read_bytes(),raw)
            self.assertEqual(digest,hashlib.sha256(raw).hexdigest())
            self.assertEqual(values['warmup_before_capture'],0)
            for sha,case in [('b'*64,payload['case']),('a'*64,{'shape':[4,512]})]:
                with self.assertRaises(ValueError):freeze_autotune_replay(pair,out,sha,case)

    def test_empty_replay_does_not_override_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'actual-ir').mkdir()
            (root/'actual-ir/autotune-cache.json').write_text(json.dumps(dict(source_sha256='a',case={},entries=[])))
            self.assertEqual(freeze_autotune_replay(root,root,'a',{}),({},None))
