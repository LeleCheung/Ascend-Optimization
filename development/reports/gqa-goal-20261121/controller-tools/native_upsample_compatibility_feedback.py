"""Record read-only compatibility review and return native metadata to the model."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

r=Path('/data/hanle/ascend-optimization/runs/native-upsample-contrast-20261118');run=r/'compatibility';w=run/'workspace'
sha=hashlib.sha256((w/'candidate.py').read_bytes()).hexdigest()
review={'source_sha256':sha,'status':'changes_required','author':'Codex read-only review',
        'issues':['Tuple and Optional remain in annotations but typing imports are absent; import will fail.',
                  'The invented 32x32 single autotune configuration is not equivalent to the original search space. Actual native metadata is now provided.',
                  'Original tle.program_id returns int64; preserve that semantic or explicitly document a required compatibility difference.'],
        'scope':'Compatibility review only; no performance strategy or candidate code patch supplied.'}
(w/'compatibility-review.json').write_text(json.dumps(review,indent=2))
shutil.copyfile(r/'native-runtime-facts.json',w/'native-runtime-facts.json')
prompt=(run/'prompt.md').read_text()+'\nThis is compatibility follow-up 02. Read compatibility-review.json and native-runtime-facts.json. Correct the review issues using the observed native configuration space rather than inventing a new tiling strategy. The facts are bound to the original source and must be cited as background when recording a decision on the current candidate. Preserve the underlying algorithm and loops. The explicit output cast is a hypothesis, not yet a verified fix. Register a new decision before edits. Do not repeatedly resubmit preflight after awaiting_source_review.\n'
(run/'prompt-02.md').write_text(prompt)
cmd=[sys.executable,'-m','hgkda.runtime.agent_runner','--config',str(run/'config.json'),'--model-settings',str(run/'model-settings.json'),'--invocation','compatibility-02','--max-turns','48','--prompt-file',str(run/'prompt-02.md')]
with (run/'generation-02.log').open('x') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(r/'method/hgkda/src')),start_new_session=True)
print(child.pid)
