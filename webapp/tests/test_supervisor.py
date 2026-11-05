import json
from pathlib import Path

import pytest

from webapp import supervisor, worker


@pytest.mark.parametrize('error,kind,status', [
    ('Could not connect to CryoSPARC: connection refused', None, 'waiting_for_cryosparc'),
    ('OpenAI-compatible request failed to reach: Remote end closed connection', None, 'waiting_for_model_api_network'),
    ('HTTP Error 503', 'api', 'waiting_for_model_api_network'),
    ('HTTP 504', 'cryosparc', 'waiting_for_cryosparc'),
    ('Tunnel connection failed: 502 Bad Gateway', 'api', 'waiting_for_model_api_network'),
    ('HTTP 429', 'api', 'waiting_for_model_api_network'),
    ('[SSL] record layer failure (_ssl.c:2590)', 'api', 'waiting_for_model_api_network'),
    ('EOF occurred in violation of protocol', 'api', 'waiting_for_model_api_network'),
    ('certificate verify failed', 'api', 'blocked_credentials_or_configuration'),
    ('certificate verify failed: record layer failure', 'api', 'blocked_credentials_or_configuration'),
    ('HTTP 503 model_not_found', 'api', 'blocked_credentials_or_configuration'),
    ('HTTP 400 provider_type_unavailable', 'api', 'blocked_credentials_or_configuration'),
    ('HTTP 403 Tunnel connection failed: 403 Forbidden', 'api', 'blocked_credentials_or_configuration'),
    ('HTTP 401', 'api', 'blocked_credentials_or_configuration'),
    ('insufficient_quota', 'api', 'blocked_credentials_or_configuration'),
    ('HTTP 429 exceeded quota', 'api', 'blocked_credentials_or_configuration'),
    ('Job is unresponsive - no heartbeat received', None, 'blocked_non_transport_error'),
    ('Invalid bool parameter', None, 'blocked_non_transport_error'),
])
def test_error_classification(error, kind, status):
    assert supervisor.classify_error(error, kind) == status


def test_direct_tls_exception_in_model_summary_waits_for_api_network(tmp_path):
    folder = tmp_path / 'agent/run_01'
    folder.mkdir(parents=True)
    (folder / 'summary.json').write_text(json.dumps({
        'stop_reason':'model_call_error',
        'error':{'error_type':'SSLError', 'error':'[SSL] record layer failure (_ssl.c:2590)'},
    }))
    assert supervisor.failure_state(tmp_path, 0) == 'waiting_for_model_api_network'


def setup_checkpoint(root, **changes):
    data = dict(project_uid='P6', workspace_uid='W90', current_node='J1',
                pending_jobs=[{'job_uid': 'J2'}], next_round=2)
    data.update(changes)
    path = root / 'agent/run_01/checkpoint.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    return path


TASK = dict(project='P6', workspace='W90', import_job='J1', config={'max_jobs': 80, 'target': 3.5})
JOBS = [{'uid': 'J1', 'status': 'completed'}, {'uid': 'J2', 'status': 'running'}]


def test_pending_gpu_job_is_reused(tmp_path):
    path = setup_checkpoint(tmp_path)
    assert supervisor.recovery_checkpoint(tmp_path, TASK, ['J1'], JOBS) == path


def test_first_attempt_records_import_and_prior_history(tmp_path):
    path = setup_checkpoint(tmp_path, current_node='J2', pending_jobs=[{'job_uid': 'J3'}],
                            experiment_history=[{'created_job_uid': 'J2'}])
    jobs = JOBS + [{'uid': 'J3', 'status': 'running'}]
    assert supervisor.recovery_checkpoint(tmp_path, TASK, [], jobs) == path


def test_unrecorded_created_job_blocks(tmp_path):
    setup_checkpoint(tmp_path)
    with pytest.raises(ValueError, match='Uncheckpointed'):
        supervisor.recovery_checkpoint(tmp_path, TASK, ['J1'], JOBS + [{'uid': 'J3', 'status': 'building'}])


def test_ambiguous_budget_intent_blocks(tmp_path):
    (tmp_path / 'benchmark_budget.intent.json').write_text('{}')
    with pytest.raises(ValueError, match='Ambiguous'):
        supervisor.recovery_checkpoint(tmp_path, TASK, ['J1'], JOBS)


@pytest.mark.parametrize('changes,message', [
    ({'workspace_uid': 'W91'}, 'identity'),
    ({'pending_jobs': [{'job_uid': 'J3'}]}, 'Uncheckpointed'),
])
def test_invalid_checkpoint(tmp_path, changes, message):
    setup_checkpoint(tmp_path, **changes)
    with pytest.raises(ValueError, match=message):
        supervisor.recovery_checkpoint(tmp_path, TASK, ['J1'], JOBS)


def test_building_pending_job_requires_manual_reconciliation(tmp_path):
    setup_checkpoint(tmp_path)
    with pytest.raises(ValueError, match='unqueued'):
        supervisor.recovery_checkpoint(tmp_path, TASK, ['J1'], [JOBS[0], {'uid': 'J2', 'status': 'building'}])


def test_import_only_checkpoint_fallback(tmp_path):
    assert supervisor.recovery_checkpoint(tmp_path, TASK, ['J1'], [JOBS[0]]) is None
    with pytest.raises(ValueError, match='No durable'):
        supervisor.recovery_checkpoint(tmp_path, TASK, ['J1'], JOBS)


def test_latest_durable_checkpoint_not_latest_failed_run(tmp_path):
    path = setup_checkpoint(tmp_path)
    failed = tmp_path / 'agent/run_02/summary.json'
    failed.parent.mkdir()
    failed.write_text('{}')
    assert supervisor.recovery_checkpoint(tmp_path, TASK, ['J1'], JOBS) == path


def test_scientific_stop_never_retried_even_if_prompt_mentions_network(tmp_path):
    path = tmp_path / 'agent/run_01/summary.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'stop_reason': 'request_input', 'error': 'connection refused'}))
    assert supervisor.failure_state(tmp_path, 0) is None


def test_api_failure_retry_and_stale_failure_ignored(tmp_path):
    path = tmp_path / 'agent/run_01/summary.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'stop_reason': 'model_call_error', 'error': {'error': 'OpenAI-compatible timeout'}}))
    assert supervisor.failure_state(tmp_path, 0) == 'waiting_for_model_api_network'
    assert supervisor.failure_state(tmp_path, path.stat().st_mtime + 1) is None


def test_resume_uses_checkpoint_and_preserves_round_budget(tmp_path, monkeypatch):
    path = setup_checkpoint(tmp_path)
    (path.parent / 'round_01').mkdir()
    monkeypatch.setenv('CRYOAGENT_WEB_SUPERVISED', '1')
    args, remaining = worker.supervised_resume({**TASK, 'service_resume_checkpoint': str(path)}, tmp_path)
    assert args == ['--resume-checkpoint', str(path)]
    assert remaining == 79


def test_external_checkpoint_rejected(tmp_path):
    with pytest.raises(ValueError, match='belong'):
        worker.supervised_resume({**TASK, 'service_resume_checkpoint': '/tmp/not-this-task.json'}, tmp_path)


def test_ordinary_worker_arguments_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv('CRYOAGENT_WEB_SUPERVISED', raising=False)
    assert worker.supervised_resume(TASK, tmp_path) == ([], 80)


def test_full_loop_waits_for_stability_and_resumes_pending_job(tmp_path, monkeypatch):
    task = {**TASK, 'id': 'test', 'status': 'running', 'slurm_job': '123'}
    (tmp_path / 'task.json').write_text(json.dumps(task))
    monkeypatch.setattr(supervisor, 'job_path', lambda _: tmp_path)
    monkeypatch.setattr(supervisor, 'read_task', lambda _: json.loads((tmp_path / 'task.json').read_text()))
    monkeypatch.setenv('SLURM_JOB_ID', '124')
    monkeypatch.setenv('CRYOAGENT_WEB_SUPERVISED', '1')
    jobs = [JOBS[0]]
    sleeps, launches = [], []
    monkeypatch.setattr(supervisor.time, 'sleep', sleeps.append)
    monkeypatch.setattr(supervisor, 'probe', lambda _, kind: {'healthy': True, 'jobs': list(jobs)})

    def run(command, **kwargs):
        launches.append(command)
        if len(launches) == 1:
            setup_checkpoint(tmp_path)
            jobs.append(JOBS[1])
            (tmp_path / 'agent/run_01/summary.json').write_text(json.dumps({
                'stop_reason': 'model_call_error', 'error': 'OpenAI-compatible connection reset'}))
        else:
            updated = json.loads((tmp_path / 'task.json').read_text())
            assert updated['service_resume_checkpoint'].endswith('run_01/checkpoint.json')
            assert updated['service_recoveries'][-1]['checkpoint']
            path = tmp_path / 'agent/run_02/summary.json'
            path.parent.mkdir()
            path.write_text(json.dumps({'stop_reason': 'target_met'}))
        return type('Result', (), {'returncode': 0})()

    monkeypatch.setattr(supervisor.subprocess, 'run', run)
    supervisor.supervise('test')
    assert len(launches) == 2
    assert sleeps == [15, 120, 15]
    assert json.loads((tmp_path / 'service_supervision/status.json').read_text())['status'] == 'stopped'


def test_adoption_does_not_restart_normally_stopped_worker(tmp_path, monkeypatch):
    task = {**TASK, 'status': 'needs_attention', 'stop_reason': 'request_input'}
    path = tmp_path / 'agent/run_01/summary.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'stop_reason': 'request_input'}))
    monkeypatch.setattr(supervisor, 'job_path', lambda _: tmp_path)
    monkeypatch.setattr(supervisor, 'read_task', lambda _: task)
    monkeypatch.setattr(supervisor, 'active_controller', lambda _: False)
    monkeypatch.setattr(supervisor, 'probe', lambda *_: pytest.fail('No probe or relaunch after scientific stop'))
    supervisor.supervise('test', adopt='123')
    assert json.loads((tmp_path / 'service_supervision/status.json').read_text())['launches'] == 0


def test_health_failure_waits_without_model_calls_or_worker_submission(tmp_path, monkeypatch):
    task = {**TASK, 'status': 'running'}
    monkeypatch.setattr(supervisor, 'job_path', lambda _: tmp_path)
    monkeypatch.setattr(supervisor, 'read_task', lambda _: task)
    probes = []
    monkeypatch.setattr(supervisor, 'probe', lambda _, kind: probes.append(kind) or
                        {'healthy': False, 'state': 'waiting_for_cryosparc'})

    def stop_after_sleep(seconds):
        assert seconds == 120
        raise InterruptedError('end test')

    monkeypatch.setattr(supervisor.time, 'sleep', stop_after_sleep)
    with pytest.raises(InterruptedError):
        supervisor.supervise('test')
    assert probes == ['cryosparc']
    assert json.loads((tmp_path / 'service_supervision/status.json').read_text())['launches'] == 0


def test_current_failure_not_masked_by_old_api_summary(tmp_path):
    path = tmp_path / 'agent/run_01/summary.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'stop_reason': 'model_call_error', 'error': 'HTTP 400'}))
    (tmp_path / 'controller_failure.json').write_text(json.dumps({'error': 'Could not connect to CryoSPARC'}))
    assert supervisor.failure_state(tmp_path, 0) == 'waiting_for_cryosparc'


def test_legacy_missing_summary_uses_exception_not_old_result(tmp_path, monkeypatch):
    old = tmp_path / 'agent/run_01/summary.json'
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({'stop_reason': 'model_call_error', 'error': 'HTTP 400'}))
    (tmp_path / 'agent/run_02').mkdir()
    monkeypatch.setattr(supervisor, 'LOGS', tmp_path)
    (tmp_path / 'test-123.log').write_text('Traceback (most recent call last):\nRuntimeError: Could not connect to CryoSPARC: connection refused\n')
    assert supervisor.adoption_failure(tmp_path, 'test', '123') == 'waiting_for_cryosparc'


def test_finalize_does_not_launch_model_or_create_jobs(tmp_path, monkeypatch):
    summary = tmp_path / 'agent/run_01/summary.json'
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({'stop_reason': 'target_met', 'success': True}))
    task = {**TASK, 'status': 'running', 'service_finalize_checkpoint': True}
    monkeypatch.setattr(worker, 'snapshot', lambda *_: {'best_refine': {'job_uid': 'J2'}})
    exports = []
    monkeypatch.setattr(worker, 'export_result', lambda *_: exports.append(True))
    monkeypatch.setattr(worker.subprocess, 'Popen', lambda *_: pytest.fail('No inference on export recovery'))
    monkeypatch.setenv('CRYOAGENT_WEB_WORKER_PHASE', 'executing')
    worker.finalize(task, tmp_path, {'summary_file': str(summary), 'runner_exit_code': 0})
    assert exports == [True]
    assert json.loads((tmp_path / 'task.json').read_text())['status'] == 'finished'
    assert 'service_finalize_checkpoint' not in task


def test_legacy_request_input_recovers_only_when_authorized(tmp_path, monkeypatch):
    path = tmp_path / 'agent/run_01/summary.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'stop_reason': 'model_request_input'}))
    monkeypatch.delenv('CRYOAGENT_UNATTENDED', raising=False)
    assert supervisor.failure_state(tmp_path, 0) is None
    monkeypatch.setenv('CRYOAGENT_UNATTENDED', '1')
    assert supervisor.failure_state(tmp_path, 0) == 'automatic_input_recovery'


def test_unrecoverable_unattended_run_is_terminal_not_manual_wait(tmp_path, monkeypatch):
    path = tmp_path / 'agent/run_01/summary.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'stop_reason': 'unattended_recovery_exhausted', 'success': False,
                               'error': {'error': 'No safe legal action remains'}}))
    monkeypatch.setattr(worker, 'snapshot', lambda *_: {'best_refine': None})
    monkeypatch.setattr(worker, 'export_result', lambda *_: None)
    monkeypatch.setenv('CRYOAGENT_WEB_WORKER_PHASE', 'executing')
    worker.finalize(dict(TASK), tmp_path, {'summary_file': str(path), 'runner_exit_code': 0})
    assert json.loads((tmp_path / 'task.json').read_text())['status'] == 'failed'
