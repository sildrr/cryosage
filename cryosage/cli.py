"""Credential-file bootstrap and read-only deployment diagnostics."""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path.home() / '.config/cryosage/deployment.env'


def load_config(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError('Config not found; run cryosage-web init-config first')
    if path.stat().st_mode & 0o077:
        raise ValueError('Config must be private: chmod 600 ' + str(path))
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:]
        key, separator, value = line.partition('=')
        if not separator or not re.fullmatch(r'[A-Z_][A-Z0-9_]*', key):
            raise ValueError(f'Invalid config syntax on line {number}')
        tokens = shlex.split(value, comments=True)
        if len(tokens) > 1:
            raise ValueError(f'Quote values containing spaces on line {number}')
        value = os.path.expandvars(tokens[0] if tokens else '')
        if '$(' in value or '`' in value or '\x00' in value:
            raise ValueError(f'Shell expressions are not allowed on line {number}')
        os.environ.setdefault(key, value)
    sys.path[:0] = [str(ROOT), str(ROOT / 'cryosparc_agent')]
    os.environ.setdefault('CRYOAGENT_REPO_ROOT', str(ROOT))
    os.environ.setdefault('CRYOAGENT_WEB_PYTHON', sys.executable)
    os.environ.setdefault('PYTHONPATH', str(ROOT) + ':' + str(ROOT / 'cryosparc_agent'))
    os.environ.setdefault('CRYOAGENT_MCP_SERVER_PYTHON', sys.executable)
    os.environ.setdefault('CRYOAGENT_TIFF_PROBE_PYTHON', sys.executable)
    os.environ.setdefault('CRYOAGENT_KNOWN_WORKFLOW_DIRS', '')
    os.environ.setdefault('CRYOAGENT_WEB_GPU_ROUTING', '1')
    os.environ.setdefault('CRYOAGENT_WEB_DYNAMIC_GPUS', '1')
    os.environ.setdefault('CRYOAGENT_WEB_MAX_GPUS', '8')
    os.environ.setdefault('CRYOAGENT_CRYOSPARC_HOST', 'localhost')
    for name in ('CRYOAGENT_WEB_DATA_ROOT', 'CRYOAGENT_WEB_LOG_ROOT'):
        value = os.getenv(name)
        if value:
            if not Path(value).is_absolute():
                raise ValueError(name + ' must be absolute')
            Path(value).mkdir(parents=True, exist_ok=True)


def doctor(*, workflow=False, online=False):
    checks = []
    def add(name, ok, detail, required=True):
        checks.append(dict(name=name, ok=bool(ok), required=required, detail=detail))
    for module in ('mcp', 'cryosparc.tools', 'pydantic', 'numpy', 'PIL', 'scipy', 'matplotlib', 'mrcfile', 'tifffile', 'requests'):
        try:
            importlib.import_module(module)
            add('import:' + module, True, 'available')
        except Exception as exc:
            add('import:' + module, False, type(exc).__name__)
    for binary in ('sbatch', 'squeue', 'sinfo', 'scontrol', 'scancel'):
        add(binary, shutil.which(binary), 'on PATH' if shutil.which(binary) else 'not found', workflow)
    for name in ('OPENAI_API_KEY', 'CRYOSPARC_EMAIL', 'CRYOSPARC_PASSWORD'):
        value = os.getenv(name, '')
        valid = bool(value and value != 'CHANGE_ME')
        if name == 'OPENAI_API_KEY' and os.getenv('CRYOAGENT_API_ENV_FILE'):
            valid = Path(os.environ['CRYOAGENT_API_ENV_FILE']).is_file()
        add(name, valid, 'configured' if valid else 'not configured', workflow)
    for name in ('CRYOAGENT_WEB_DATA_ROOT', 'CRYOAGENT_WEB_LOG_ROOT'):
        path = Path(os.environ[name])
        add(name, path.is_dir() and os.access(path, os.W_OK), str(path))
    add('input roots', all(Path(p).is_dir() for p in os.getenv('CRYOAGENT_WEB_INPUT_ROOTS', '').split(':') if p),
        'Check read permissions from controller and native compute nodes', workflow)
    tools = ('CRYOAGENT_CRYOSIFT_PYTHON', 'CRYOAGENT_CRYOSIFT_RUNNER', 'CRYOAGENT_CRYOSIFT_WEIGHTS')
    for name in tools:
        value = os.getenv(name)
        ok = bool(value and Path(value).is_file() and os.access(value, os.R_OK))
        add(name, ok, 'readable' if ok else 'missing external CryoSift asset', workflow)
    for name in ('CRYOAGENT_CRYOLO_WEIGHTS', 'CRYOAGENT_PROSHADE_BINARY'):
        value = os.getenv(name)
        add(name, bool(value and Path(value).is_file()), 'optional external asset', False)
    if online:
        try:
            from cryosparc_client import cryosparc_client
            client = cryosparc_client()
            project = client.find_project(os.getenv('CRYOAGENT_WEB_PROJECT', 'P1'))
            lanes = [t.lane for t in client.get_targets()]
            required = [x.strip() for x in os.getenv('CRYOAGENT_WEB_GPU_LANES', 'gpu_slurm').split(',') if x.strip()]
            add('CryoSPARC project', True, project.uid)
            add('registered GPU lanes', bool(required) and set(required).issubset(lanes), ','.join(required))
        except Exception as exc:
            add('CryoSPARC read-only connection', False, type(exc).__name__)
    passed = all(row['ok'] for row in checks if row['required'])
    print(json.dumps(dict(passed=passed, checks=checks, real_jobs_submitted=0,
        paid_model_requests=0), indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(os.getenv('CRYOSAGE_CONFIG', str(DEFAULT_CONFIG))))
    parser.add_argument('action', choices=['init-config', 'doctor', 'start', 'stop', 'status', 'serve', 'resume', 'restart-proxy'])
    parser.add_argument('task_id', nargs='?')
    parser.add_argument('--workflow', action='store_true', help='Require external tools and scheduler setup')
    parser.add_argument('--online', action='store_true', help='Read-only CryoSPARC project/lane verification')
    args = parser.parse_args()
    if args.action == 'init-config':
        args.config.parent.mkdir(parents=True, exist_ok=True)
        content = (ROOT / 'deploy/deployment.env.example').read_text()
        descriptor = os.open(args.config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(content)
        print('Created private config: ' + str(args.config))
        return 0
    load_config(args.config)
    if args.action == 'doctor':
        return doctor(workflow=args.workflow, online=args.online)
    if args.action == 'resume':
        if not args.task_id:
            parser.error('resume requires a task ID')
        from webapp.resume import main as resume
        resume(args.task_id)
        return 0
    if args.action == 'serve':
        from webapp.server import main as serve
        sys.argv = ['webapp.server', '--port', os.getenv('CRYOAGENT_WEB_PORT', '8777')]
        serve()
    else:
        from webapp.manage import main as manage
        sys.argv = ['webapp.manage', args.action]
        manage()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
