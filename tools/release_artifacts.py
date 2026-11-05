"""Audit tracked release files and build a deterministic source archive."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(rb'sk-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    records, blobs = {}, {}
    for name in filter(None, names):
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Invalid release file: ' + name)
        if any(part in {'.git', '.venv', 'logs', 'tasks', 'uploads', '__pycache__'} for part in path.relative_to(ROOT).parts):
            raise ValueError('Runtime directory entered tracked release: ' + name)
        if path.suffix in {'.env', '.token', '.pem', '.key', '.sqlite', '.db', '.mrc', '.mrcs', '.pth', '.h5', '.pt'}:
            raise ValueError('Sensitive/runtime file entered tracked release: ' + name)
        data = path.read_bytes()
        if path.suffix != '.png' and TOKEN.search(data):
            raise ValueError('Potential credential; inspect privately: ' + name)
        if path.suffix == '.py':
            ast.parse(data, filename=name)
        records[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        blobs[name] = data
    report = dict(version='2026.11.5', files=records, credentials_found=False,
                  weights_included=False, runtime_data_included=False)
    if args.archive:
        target = args.archive.expanduser().resolve()
        if target.exists():
            raise ValueError('Refusing to overwrite an existing archive')
        manifest = json.dumps(report, indent=2, sort_keys=True).encode() + b'\n'
        blobs['RELEASE_MANIFEST.json'] = manifest
        with tarfile.open(target, 'w:gz') as archive:
            for name, data in sorted(blobs.items()):
                item = tarfile.TarInfo('cryosage/' + name)
                item.size, item.mtime = len(data), 0
                item.mode = 0o755 if name.endswith(('.sh', '.sbatch')) else 0o644
                archive.addfile(item, io.BytesIO(data))
        target.chmod(0o644)
        checksum = target.with_name(target.name + '.sha256')
        checksum.write_text(hashlib.sha256(target.read_bytes()).hexdigest() + '  ' + target.name + '\n')
        checksum.chmod(0o644)
    print(json.dumps(dict(files=len(records), bytes=sum(r['bytes'] for r in records.values()),
        credential_scan='passed', python_syntax='passed', archive=str(args.archive) if args.archive else None)))


if __name__ == '__main__':
    main()
