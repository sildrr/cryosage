import os
from pathlib import Path
import subprocess
import sys
import zipfile
import json
import socket
import time
import urllib.error
import urllib.request

import pytest

from cryosage import cli


def test_config_private_and_no_shell_execution(tmp_path, monkeypatch):
    config = tmp_path / 'deployment.env'
    config.write_text('CRYOAGENT_WEB_DATA_ROOT=' + str(tmp_path / 'data') + '\n'
                      'CRYOAGENT_WEB_LOG_ROOT=' + str(tmp_path / 'logs') + '\n'
                      'CRYOAGENT_WEB_PORT=18999\n')
    config.chmod(0o644)
    with pytest.raises(ValueError, match='private'):
        cli.load_config(config)
    config.chmod(0o600)
    monkeypatch.setenv('CRYOAGENT_WEB_PORT', '19000')
    monkeypatch.delenv('CRYOAGENT_WEB_DATA_ROOT', raising=False)
    monkeypatch.delenv('CRYOAGENT_WEB_LOG_ROOT', raising=False)
    cli.load_config(config)
    assert os.environ['CRYOAGENT_WEB_PORT'] == '19000'
    assert (tmp_path / 'data').is_dir()
    config.write_text('EVIL=$(touch /tmp/not-allowed)\n')
    with pytest.raises(ValueError):
        cli.load_config(config)


def test_templates_and_assets_are_installed():
    root = cli.ROOT
    for name in ('webapp/controller.sbatch', 'deploy/deployment.env.example',
                 'webapp/static/index.html', 'webapp/static/app.js',
                 'cryosparc_agent/autonomous_mcp_closed_loop.py',
                 'cryosparc_agent/cryosparc_mcp_server.py',
                 'cryosparc_agent/guidance/reviewed_stage_guidance.json',
                 'scripts/cryolo_size_probe.py',
                 'benchmarks/agent_vs_codex_20261017/kb_server.py',
                 'cryosparc_result_agent/scripts/submit_cryonet_mode22.py'):
        assert (root / name).is_file(), name


def test_generic_controller_contains_no_developer_paths():
    script = (cli.ROOT / 'webapp/controller.sbatch').read_text()
    assert '/home/lisongyang' not in script
    assert '/ssd1/' not in script
    assert 'CRYOAGENT_REPO_ROOT' in script
    assert 'webapp.supervisor' in script
    assert '--gres=' not in script


def test_help_no_credentials_or_services_required():
    result = subprocess.run([sys.executable, '-m', 'cryosage.cli', '--help'],
                            capture_output=True, text=True, check=True)
    assert 'doctor' in result.stdout
    assert 'resume' in result.stdout


def test_init_config_never_overwrites(tmp_path):
    path = tmp_path / 'config.env'
    command = [sys.executable, '-m', 'cryosage.cli', '--config', str(path), 'init-config']
    subprocess.run(command, capture_output=True, text=True, check=True)
    assert path.stat().st_mode & 0o077 == 0
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_custom_gpu_lanes_loaded_before_backend_import(tmp_path):
    result = subprocess.run([sys.executable, '-c',
        'from webapp.gpu_routing import LANES; print(LANES)'],
        env=dict(os.environ, CRYOAGENT_WEB_GPU_LANES='my4090,myh20'),
        capture_output=True, text=True, check=True)
    assert "('my4090', 'myh20')" in result.stdout


def test_real_loopback_ui_uses_isolated_token_and_no_scheduler(tmp_path):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    config = tmp_path / 'ui.env'
    config.write_text('CRYOAGENT_WEB_DATA_ROOT=' + str(tmp_path / 'data') + '\n'
                      'CRYOAGENT_WEB_LOG_ROOT=' + str(tmp_path / 'logs') + '\n'
                      'CRYOAGENT_WEB_TOKEN_FILE=' + str(tmp_path / 'web.token') + '\n'
                      'CRYOAGENT_WEB_PORT=' + str(port) + '\n')
    config.chmod(0o600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRYOAGENT_WEB_')}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (tmp_path / 'server.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'cryosage.cli', '--config', str(config), 'serve'],
                                   env=env, stdout=log, stderr=log)
        try:
            url = f'http://127.0.0.1:{port}'
            for _ in range(100):
                try:
                    with opener.open(url + '/api/health', timeout=1) as response:
                        assert response.status == 200
                    break
                except (OSError, urllib.error.URLError):
                    if process.poll() is not None:
                        pytest.fail((tmp_path / 'server.log').read_text())
                    time.sleep(.05)
            else:
                pytest.fail('UI did not start')
            with opener.open(url) as response:
                assert b'CryoSage' in response.read()
            with pytest.raises(urllib.error.HTTPError) as error:
                opener.open(url + '/api/tasks')
            assert error.value.code == 401
            token = (tmp_path / 'web.token').read_text().strip()
            request = urllib.request.Request(url + '/api/tasks', headers={'Authorization': 'Bearer ' + token})
            with opener.open(request) as response:
                assert json.load(response) == []
            with opener.open(url + '/api/showcase') as response:
                assert len(json.load(response)) == 4
        finally:
            process.terminate()
            process.wait(timeout=10)
