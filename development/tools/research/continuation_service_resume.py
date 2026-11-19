"""Resume a recorded project-owned evaluation service without altering experiments."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
from urllib.parse import urlparse
from datetime import datetime, timezone

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
a = p.parse_args()
r = a.run.resolve()
service = json.loads((r / 'service.json').read_bytes())
assert service['scope'] == 'project-owned'
command = service['command']
assert any(str(r) in part for part in command)
address = urlparse(service['server'])
assert address.hostname == '127.0.0.1'
with socket.socket() as probe:
    assert probe.connect_ex((address.hostname, address.port)) != 0, 'Service already listening'
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
with (r / ('kgs-resume-' + stamp + '.log')).open('x') as log:
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
record = dict(service, pid=proc.pid, resumed_at=stamp, previous_pid=service['pid'])
(r / ('service-before-' + stamp + '.json')).write_bytes((r / 'service.json').read_bytes())
(r / 'service.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(pid=proc.pid, server=service['server'])))
