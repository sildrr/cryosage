import json
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from api_failover import route_for, recoverable
from model_direct_runner import run_openai_compatible_model, read_api_key_file


@pytest.fixture
def configured(tmp_path, monkeypatch):
    config = dict(enabled=True,primary_base='https://primary.test/v1',
        primary_env_file=str(tmp_path/'primary.env'),fallback_base='http://backup.test/v1',
        fallback_env_file=str(tmp_path/'backup.env'),direct=True,
        models={'gpt-6.1-sol':'gpt-6.1-sol'},audit_file=str(tmp_path/'audit.jsonl'))
    (tmp_path/'backup.env').write_text('sk-test-only')
    path=tmp_path/'fallback.json'
    path.write_text(json.dumps(config))
    monkeypatch.setenv('CRYOAGENT_API_FALLBACK_CONFIG',str(path))
    monkeypatch.setenv('CRYOAGENT_API_ENV_FILE',config['primary_env_file'])
    return config


def http_failure(code):
    original=HTTPError('https://primary.test',code,'error',{},None)
    result=RuntimeError(f'OpenAI-compatible request failed with HTTP {code}: model_not_found')
    result.__cause__=original
    return result


def test_same_model_scoped_to_primary_credentials(configured,monkeypatch):
    assert route_for(configured['primary_base'],'gpt-6.1-sol')
    assert route_for('https://other.test/v1','gpt-6.1-sol') is None
    assert route_for(configured['primary_base'],'gpt-6-luna') is None
    monkeypatch.setenv('CRYOAGENT_API_ENV_FILE','/different/account.env')
    assert route_for(configured['primary_base'],'gpt-6.1-sol') is None


def test_primary_success_never_contacts_backup(configured):
    with patch('model_direct_runner._run_openai_compatible_model_once',return_value={'raw_text':'ok'}) as call:
        assert run_openai_compatible_model([],configured['primary_base'],'primary-key','gpt-6.1-sol')['raw_text']=='ok'
        assert call.call_count==1


def test_failover_preserves_messages_tools_and_limits(configured):
    messages=[{'role':'user','content':'synthetic'}]
    tools=[{'type':'function','function':{'name':'test'}}]
    with patch('model_direct_runner._run_openai_compatible_model_once',side_effect=[http_failure(503),{'raw_text':'ok'}]) as call:
        result=run_openai_compatible_model(messages,configured['primary_base'],'primary-key',
            'gpt-6.1-sol',max_new_tokens=8192,tools=tools,max_retries=0)
        args,kwargs=call.call_args
        assert args==(messages,configured['fallback_base'],'sk-test-only','gpt-6.1-sol')
        assert kwargs['tools'] is tools and kwargs['max_new_tokens']==8192 and kwargs['direct'] is True
        assert result['provider_failover']['requested_model']=='gpt-6.1-sol'
    audit=Path(configured['audit_file']).read_text()
    assert 'fallback_succeeded' in audit and 'primary-key' not in audit and 'sk-test-only' not in audit


@pytest.mark.parametrize('code',[400,401,403,413])
def test_invalid_requests_do_not_failover(configured,code):
    with patch('model_direct_runner._run_openai_compatible_model_once',side_effect=http_failure(code)) as call:
        with pytest.raises(RuntimeError):
            run_openai_compatible_model([],configured['primary_base'],'key','gpt-6.1-sol')
        assert call.call_count==1


def test_backup_failure_does_not_loop(configured):
    with patch('model_direct_runner._run_openai_compatible_model_once',side_effect=[http_failure(503),http_failure(503)]) as call:
        with pytest.raises(RuntimeError):
            run_openai_compatible_model([],configured['primary_base'],'key','gpt-6.1-sol')
        assert call.call_count==2


def test_certificate_failure_and_json_errors_do_not_failover():
    assert not recoverable(RuntimeError('OpenAI-compatible request failed to reach host: certificate verify failed'))
    assert not recoverable(ValueError('Invalid JSON'))
    assert recoverable(RuntimeError('OpenAI-compatible request failed to reach host: connection reset'))


def test_credential_formats(tmp_path):
    path=tmp_path/'key.env'
    path.write_text('sk-test-only\n')
    assert read_api_key_file(path)=='sk-test-only'
    path.write_text('export OPENAI_API_KEY="sk-test-assignment"\n')
    assert read_api_key_file(path)=='sk-test-assignment'
