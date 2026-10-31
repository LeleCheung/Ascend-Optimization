"""证据核验回归：真实CSV重算，以及拒绝错误哈希、时长和RTL运行声明。"""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_artifacts import audit
from evidence_layers import validate


class EvidenceChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.batch = Path(self.temp.name) / 'batch'; self.batch.mkdir()
        self.directory = self.batch / 'results/probe/benchmark-0'; self.directory.mkdir(parents=True)
        path = self.directory / 'task_time.csv'
        with path.open('w', newline='', encoding='utf8') as stream:
            writer = csv.writer(stream); writer.writerow(['kernel_name','task_time(us)'])
            writer.writerow(['outer',1])
            for symbol, value in [('parent_kernel',20),('candidate_kernel',10)]:
                for _ in range(8): writer.writerow([symbol,value])
            writer.writerow(['outer',1])
        self.record = dict(csv_path='results/probe/benchmark-0/task_time.csv',
            csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=[
                dict(key='parent_ms',symbol='parent_kernel',active_device_us=[20]*6,device_ms=.02),
                dict(key='candidate_ms',symbol='candidate_kernel',active_device_us=[10]*6,device_ms=.01)])
        self.save()

    def tearDown(self): self.temp.cleanup()
    def save(self): (self.directory/'measurement.json').write_text(json.dumps(self.record),encoding='utf8')

    def test_valid_csv_recomputed(self):
        self.assertEqual(audit(self.batch)['device_measurements'],1)

    def test_wrong_hash_rejected(self):
        self.record['csv_sha256']='0'*64; self.save()
        with self.assertRaisesRegex(ValueError,'原件身份'): audit(self.batch)

    def test_wrong_duration_rejected(self):
        self.record['records'][1]['device_ms']=.009; self.save()
        with self.assertRaisesRegex(ValueError,'无法重算'): audit(self.batch)

    def test_wrong_kernel_rejected(self):
        self.record['records'][1]['symbol']='unrelated'; self.save()
        with self.assertRaisesRegex(ValueError,'符号'): audit(self.batch)

    def test_rtl_claim_rejected(self):
        with self.assertRaisesRegex(ValueError,'RTL执行器'):
            validate(dict(layer='rtl',status='available',source_sha256='0'*64,artifacts=['fake'],measurements=[]))


if __name__=='__main__': unittest.main()
