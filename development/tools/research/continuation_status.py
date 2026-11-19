"""Print compact generation progress without credentials or full tool payloads."""
import json
from pathlib import Path
import sys

r=Path(sys.argv[1]);name=sys.argv[2]
stream=r/'agent-calls'/name/'agent-events.jsonl'
items=[]
for line in stream.read_bytes().splitlines():
    try:e=json.loads(line)
    except json.JSONDecodeError:continue
    m=e.get('message',{})
    if not isinstance(m,dict):continue
    for b in m.get('content',[]):
        if not isinstance(b,dict):continue
        if b.get('type')=='tool_use':
            args=b.get('input',{})
            items.append(dict(tool=b.get('name'),file=args.get('file_path'),command=args.get('command')))
        elif b.get('type')=='text':items.append(dict(text=b.get('text','')[:1200]))
print(json.dumps(items[-10:],ensure_ascii=False,indent=2))
