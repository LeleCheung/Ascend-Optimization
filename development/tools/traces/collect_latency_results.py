"""Export formal per-case timings by recorded SHA for fixed-reference recalculation without experiments."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.inventory.read_bytes());a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for r in d['records']:
        record=r['result'];raw=Path(record['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==record['sha256'],record['path']
        f=a.output/(record['sha256']+'.json')
        if not f.exists():f.write_bytes(raw)
        rows.append(dict(path=record['path'],sha256=record['sha256'],export=f.name))
    (a.output/'manifest.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n',encoding='utf8')
    print(json.dumps({'records':len(rows),'unique_results':len(set(x['sha256'] for x in rows))}))


if __name__=='__main__':main()
