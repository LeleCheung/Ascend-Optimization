"""Export reproducible study evidence without credentials, compiler caches or huge raw dumps."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();r=a.study
    skip={'cache','server-cache','checker-cache','triton-cache','profile-triton-cache','compiler-cache','raw','server-profiles','server-debug','server-bundles','method','__pycache__','method-package'}
    files=[]
    for x in r.rglob('*'):
        if not x.is_file() or any(part in skip for part in x.relative_to(r).parts):continue
        if x.name in {'model-settings.json','runtime-settings.json','environment.json'} or x.suffix in {'.gz','.tar','.pyc','.npubin','.mlirbc'}:continue
        if x.name.startswith('evidence-export-'):continue
        if x.stat().st_size>40_000_000:continue
        files.append(x)
    manifest={'files':[{'path':str(x.relative_to(r)),'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in files],
              'omitted':'Credentials/model-settings, compiler caches and large raw simulator dumps remain on server; original source and decision references retain identities.'}
    target=r/('evidence-export-'+a.name+'.tar.gz')
    with tarfile.open(target,'w:gz') as tar:
        for x in files:tar.add(x,arcname=str(x.relative_to(r)))
    (r/('evidence-export-'+a.name+'.json')).write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'archive':str(target),'files':len(files),'bytes':target.stat().st_size}))


if __name__=='__main__':main()
