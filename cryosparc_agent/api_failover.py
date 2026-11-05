"""Opt-in, per-user transport failover without changing scientific decisions."""
import json
import os
from pathlib import Path
from urllib import error
from urllib.parse import urlsplit
from datetime import datetime, timezone


def route_for(api_base, model):
    path = Path(os.getenv('CRYOAGENT_API_FALLBACK_CONFIG', '~/.config/cryoagent/api-fallback.json')).expanduser()
    if not path.is_file():
        return None
    config = json.loads(path.read_text())
    if not config.get('enabled') or api_base.rstrip('/') != config['primary_base'].rstrip('/'):
        return None
    active = Path(os.getenv('CRYOAGENT_API_ENV_FILE', '~/.config/cryoagent/api.env')).expanduser().resolve()
    if active != Path(config['primary_env_file']).expanduser().resolve():
        return None
    target_model = config.get('models', {}).get(model)
    if not target_model:
        return None
    base = config['fallback_base'].rstrip('/')
    parsed = urlsplit(base)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Invalid configured fallback API base')
    return {**config, 'fallback_base':base, 'fallback_model':target_model}


def recoverable(exc):
    """Only transport/capacity/model-route failures, not invalid scientific JSON."""
    cause = exc
    while cause is not None:
        if isinstance(cause, error.HTTPError):
            return cause.code in {408, 429, 500, 502, 503, 504} or (
                cause.code == 404 and 'model_not_found' in str(exc).lower())
        cause = cause.__cause__
    text = str(exc).lower()
    return 'openai-compatible request failed to reach ' in text and not any(
        word in text for word in ('certificate', 'hostname mismatch', 'self-signed', '403', '401'))


def audit(route, record):
    path = route.get('audit_file')
    if not path:
        return
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {'at':datetime.now(timezone.utc).isoformat(), **record}
    # Never write credentials, request bodies or provider error bodies.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(value)+'\n').encode())
    finally:
        os.close(fd)
