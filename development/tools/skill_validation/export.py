"""Export reproducible originals after completion, checking for credentials and excluding caches."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


EXCLUDE = {'__pycache__', 'triton-cache', 'profiling-triton-cache', 'offset-triton-cache',
           'service-triton-cache', 'supplement-triton-cache', '.claude'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credential-settings', type=Path, required=True,
                        help='Compare credentials in memory only; do not output or archive settings files')
    args = parser.parse_args()
    run = args.run_directory.resolve()
    assert not args.output.exists()
    for path in run.glob('formal-*/state.json'):
        assert json.loads(path.read_bytes())['status'] == 'completed'
    settings = json.loads(args.credential_settings.read_bytes())
    secrets = [str(value).encode() for key, value in settings.get('env', {}).items()
               if any(word in key.upper() for word in ['TOKEN', 'API_KEY', 'PASSWORD']) and len(str(value)) > 8]
    files = [p for p in run.rglob('*') if p.is_file() and not p.is_symlink()
             and p.resolve()!=args.credential_settings.resolve()
             and not (set(p.relative_to(run).parts) & EXCLUDE)
             and not any(part.endswith('-cache') for part in p.relative_to(run).parts)]
    assert files
    records = []
    for path in sorted(files):
        raw = path.read_bytes()
        assert not any(secret in raw for secret in secrets), 'Artifact contains credentials; export stopped without disclosing their contents'
        records.append(dict(path=path.relative_to(run).as_posix(), bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest()))
    # Export preserves reference-profile links as relative paths within the extracted evidence.
    manifest = dict(files=records, file_count=len(records), total_bytes=sum(r['bytes'] for r in records),
                    credential_scan='passed_exact_values_not_recorded',
                    source_run=str(run), excluded=sorted(EXCLUDE),
                    credential_settings_excluded=True)
    manifest_path = run/'export-manifest.json'
    assert not manifest_path.exists()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'w:gz') as archive:
        for path in sorted(files)+[manifest_path]:
            archive.add(path, arcname=path.relative_to(run).as_posix(), recursive=False)
        profile = run/'baseline-paired/reference/profile'
        if profile.is_symlink():
            target = profile.resolve()
            assert target.is_relative_to(run)
            info = tarfile.TarInfo(profile.relative_to(run).as_posix())
            info.type = tarfile.SYMTYPE
            info.linkname = target.name
            archive.addfile(info)
    print(json.dumps(dict(files=len(files), bytes=args.output.stat().st_size,
                          sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
