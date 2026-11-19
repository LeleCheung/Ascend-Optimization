"""Import an exported evidence archive, verifying paths and every file digest."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_bytes())
    expected={row['path']:row for row in manifest['files']}
    assert len(expected)==len(manifest['files']),'duplicate manifest path'
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    seen=set()
    with tarfile.open(args.archive,'r:gz') as archive:
        for member in archive:
            path=PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert ':' not in member.name and '\\' not in member.name
            assert member.name in expected and member.name not in seen
            target=root.joinpath(*path.parts)
            assert target.resolve().is_relative_to(root),'path escapes evidence directory'
            raw=archive.extractfile(member).read()
            row=expected[member.name]
            assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
            # Extended-length paths avoid Windows MAX_PATH without changing names.
            disk=Path('\\\\?\\'+str(target)) if os.name=='nt' else target
            disk.parent.mkdir(parents=True,exist_ok=True)
            if disk.exists():
                assert disk.read_bytes()==raw,'refusing to overwrite different frozen evidence'
            else:disk.write_bytes(raw)
            assert hashlib.sha256(disk.read_bytes()).hexdigest()==row['sha256']
            seen.add(member.name)
    assert seen==set(expected),'archive omits manifest entries'
    result=dict(status='verified',files=len(seen),archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    (root/'import-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
