"""Low-frequency transport supervision, never a scientific retry policy."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from webapp.core import LOGS, PYTHON, REPO, job_path, now, read_task, write
from cryosparc_agent.transport_errors import certificate_failure, transient_tls_failure

WAITING = {'waiting_for_cryosparc', 'waiting_for_model_api_network'}
RECOVERABLE = WAITING | {'automatic_input_recovery'}


def classify_error(error, kind=None):
    text = str(error).lower()
    if certificate_failure(error) or any(word in text for word in (
        'model_not_found', 'provider_type_unavailable', 'invalid_api_key',
        'quota', 'insufficient balance', 'credit exhausted', '余额不足', '额度不足', 'permission denied',
        'invalid credentials', 'unauthorized', 'forbidden',
    )) or re.search(r'(?:http(?: error)?|code)[ :]*(?:400|401|402|403|404|413)\b', text):
        return 'blocked_credentials_or_configuration'
    transient = transient_tls_failure(error) or any(word in text for word in (
        'connection refused', 'connection reset', 'connection aborted', 'timed out',
        'timeout', 'remote end closed', 'server disconnected', 'network is unreachable',
        'name or service not known', 'temporary failure in name resolution',
        'could not connect to cryosparc', 'service unavailable', 'bad gateway',
        'connection closed', 'broken pipe',
    )) or re.search(r'(?:http(?: error)?|code|failed with http)[ :]*(?:408|429|500|502|503|504)\b', text)
    if not transient:
        return 'blocked_non_transport_error'
    api = kind == 'api' or kind is None and any(word in text for word in (
        'openai-compatible', 'packyapi', 'api.ofox', 'tunnel connection', 'proxy',
    ))
    return 'waiting_for_model_api_network' if api else 'waiting_for_cryosparc'


def checkpoint_ids(value):
    ids = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'job_uid', 'current_node', 'cryosparc_job_uid', 'created_job_uid'} and isinstance(item, str):
                ids.add(item)
            ids.update(checkpoint_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.update(checkpoint_ids(item))
    return ids


def recovery_checkpoint(root, task, baseline, jobs):
    root = Path(root)
    if (root / 'benchmark_budget.intent.json').exists():
        raise ValueError('Ambiguous benchmark submission requires manual reconciliation')
    paths = sorted((root / 'agent').glob('run_*/checkpoint.json'), reverse=True)
    live = {j['uid'] for j in jobs}
    if not set(baseline).issubset(live):
        raise ValueError('Previously recorded workspace Jobs disappeared')
    from webapp.pilot_pipeline import VERSION
    pipeline_ids = set()
    if task.get('pipeline_version') == VERSION:
        pipeline_ids = {task.get('import_job'), task.get('particle_class2d_job')} - {None}
        for name in ('preprocessing.json', 'state.json'):
            record_path = root / name
            if record_path.exists():
                pipeline_ids.update(checkpoint_ids(json.loads(record_path.read_text())))
        manifest = root/'manifest.json'
        if manifest.exists():
            identity = json.loads(manifest.read_text())
            if (identity.get('task_id'), identity.get('project'), identity.get('workspace')) != (task['id'], task['project'], task['workspace']):
                raise ValueError('Pilot checkpoint identity mismatch')
        if not pipeline_ids.issubset(live):
            raise ValueError('Pilot checkpoint references missing workspace Jobs')
    if not paths:
        if task.get('pipeline_version') == VERSION:
            # Before the Planner, the serial pipeline owns its durable records.
            if not live.issubset(pipeline_ids):
                raise ValueError('Uncheckpointed production creation; refusing duplicate execution')
            return None
        if live != {task.get('import_job')}:
            raise ValueError('No durable checkpoint for workspace Jobs')
        return None
    path = paths[0]
    data = json.loads(path.read_text())
    if (data.get('project_uid'), data.get('workspace_uid')) != (task['project'], task['workspace']):
        raise ValueError('Checkpoint workspace identity mismatch')
    recorded = checkpoint_ids(data) | {task.get('import_job')} | pipeline_ids
    if not live.difference(baseline).issubset(recorded):
        raise ValueError('Uncheckpointed workspace creation; refusing duplicate execution')
    required = checkpoint_ids(data)
    if not required.issubset(live):
        raise ValueError('Checkpoint references missing workspace Jobs')
    pending = {j['job_uid'] for j in data.get('pending_jobs', [])}
    if any(j['uid'] in pending and j['status'] == 'building' for j in jobs):
        raise ValueError('Created but unqueued Job requires manual reconciliation')
    return path


def probe_child(identifier, kind):
    from webapp.proxy import configure_compute_proxy
    configure_compute_proxy()
    if kind == 'api':
        sys.path.insert(0, str(REPO / 'cryosparc_agent'))
        from model_direct_runner import resolve_api_key
        base = os.getenv('OPENAI_BASE_URL', 'https://api.ofox.ai/v1')
        req = Request(base.rstrip('/') + '/models', headers={'Authorization': 'Bearer ' + resolve_api_key()})
        with urlopen(req, timeout=30) as response:
            models = {item['id'] for item in json.load(response).get('data', [])}
        configured = set(read_task(identifier)['model_config'].values())
        if any(m not in models and m.removeprefix('openai/') not in models and
               'openai/' + m not in models for m in configured):
            raise ValueError('Configured model absent from API model catalog')
        return {'healthy': True, 'model_calls': 0}
    sys.path.insert(0, str(REPO / 'cryosparc_agent'))
    from cryosparc_client import cryosparc_client
    cs = cryosparc_client()
    task = read_task(identifier)
    if not task.get('workspace'):
        cs.find_project(os.getenv('CRYOAGENT_WEB_PROJECT', 'P6'))
        return {'healthy': True, 'jobs': []}
    jobs = list(cs.find_project(task['project']).find_workspace(task['workspace']).find_jobs())
    return {'healthy': True, 'jobs': [{'uid': j.uid, 'status': j.status} for j in jobs]}


def probe(identifier, kind):
    try:
        result = subprocess.run([PYTHON, '-m', 'webapp.supervisor', identifier, '--probe', kind],
                                cwd=REPO, capture_output=True, text=True, timeout=45)
        if result.returncode:
            return {'healthy': False, 'state': classify_error(result.stderr, kind),
                    'error_type': 'health_probe_failed'}
        return json.loads(result.stdout.splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {'healthy': False, 'state': classify_error('timeout', kind), 'error_type': 'probe_timeout'}


def failure_versions(root):
    paths = list((Path(root) / 'agent').glob('run_*/summary.json'))
    failure = Path(root) / 'controller_failure.json'
    if failure.exists():
        paths.append(failure)
    return {str(p): p.stat().st_mtime_ns for p in paths}


def failure_state(root, since):
    """Inspect structured failures only, never prompts or scientific observations."""
    def changed(path):
        return (since.get(str(path)) != path.stat().st_mtime_ns if isinstance(since, dict)
                else path.stat().st_mtime >= since)
    paths = [p for p in (Path(root) / 'agent').glob('run_*/summary.json') if changed(p)]
    failure = Path(root) / 'controller_failure.json'
    if failure.exists() and changed(failure) and (not paths or failure.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in paths)):
        state = classify_error(json.loads(failure.read_text()).get('error'))
        return state if state in WAITING else None
    if paths:
        data = json.loads(max(paths).read_text())
        reason = data.get('stop_reason')
        if reason == 'model_request_input' and os.getenv('CRYOAGENT_UNATTENDED') == '1':
            return 'automatic_input_recovery'
        if reason not in {'model_call_error', 'job_wait_timeout_or_not_terminal', 'resume_pending_job_not_terminal'}:
            return None
        errors = [data.get('error'), data.get('resumed_pending_jobs')]
        for record in data.get('rounds', []):
            for feedback in record.get('job_results', []):
                errors.append(feedback.get('result', {}).get('wait_attempts'))
        state = classify_error(json.dumps(errors), 'api' if reason == 'model_call_error' else None)
        return state if state in WAITING else None
    return None


def adoption_failure(root, identifier, controller):
    """Legacy workers may die before a summary or structured failure is written."""
    runs = sorted((Path(root) / 'agent').glob('run_*'))
    if runs and (runs[-1] / 'summary.json').exists():
        return failure_state(root, 0)
    for path in (LOGS / f'{identifier}-{controller}.log', Path(root) / 'agent.log'):
        if not path.exists():
            continue
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 16000))
            tail = stream.read().decode('utf-8', errors='replace')
        if 'Traceback (most recent call last)' not in tail:
            continue
        tail = tail.rsplit('Traceback (most recent call last)', 1)[-1]
        # Classify exception lines, never archived model prompts or reasoning.
        errors = [line.strip(' |+') for line in tail.splitlines()
                  if re.match(r'^[ |+]*(?:[\w.]*Error|ExceptionGroup|Exception):', line)]
        state = classify_error('\n'.join(errors))
        if state in WAITING:
            return state
    return None


def active_controller(controller):
    result = subprocess.run(['squeue', '-h', '-o', '%i'], capture_output=True, text=True,
                            check=True, timeout=30)
    return str(controller) in result.stdout.split()


def supervise(identifier, adopt=None, poll=120, stable=15):
    root = job_path(identifier)
    directory = root / 'service_supervision'
    directory.mkdir(exist_ok=True)
    with (directory / 'controller.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not adopt:
                    raise
                status_path = directory / 'status.json'
                owner = json.loads(status_path.read_text()).get('controller') if status_path.exists() else None
                if owner != str(adopt):
                    raise RuntimeError('Another supervisor owns this workflow; refusing duplicate recovery')
                write(directory / ('adopter-' + str(os.getenv('SLURM_JOB_ID', 'local')) + '.json'),
                      dict(status='waiting_for_existing_supervisor', previous_controller=adopt, at=now()))
                time.sleep(poll)
        baseline = None
        healthy = 0
        recovering = False
        launches = 0

        def record(status, **extra):
            value = dict(status=status, updated_at=now(), controller=os.getenv('SLURM_JOB_ID'),
                         adopted_controller=adopt, launches=launches, poll_seconds=poll, **extra)
            write(directory / 'status.json', value)
            with (directory / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(value) + '\n')
            print(json.dumps(value), flush=True)

        def stop_requested():
            control = root / 'control'
            state = control / 'state.json'
            stopped = state.exists() and json.loads(state.read_text()).get('stop')
            # Leave commands in the inbox; the execution layer owns receipts.
            stopped = stopped or any(json.loads(p.read_text()).get('action') == 'stop'
                                     for p in (control / 'inbox').glob('*.json'))
            if stopped:
                task = read_task(identifier)
                task.update(status='user_stopped', stop_reason='user_stop', finished_at=now())
                write(root / 'task.json', task)
                record('stopped', reason='user_stop')
            return stopped

        since = time.time()
        if adopt:
            # Observe the existing controller; do not interrupt any process or GPU Job.
            record('monitoring_existing_controller')
            while active_controller(adopt):
                time.sleep(poll)
            task = read_task(identifier)
            # Existing runs began before this supervisor was attached.
            state = adoption_failure(root, identifier, adopt)
            if state not in RECOVERABLE:
                record('stopped', reason=task.get('stop_reason', task.get('status')))
                return
            recovering = True
            record(state)
        while True:
            if stop_requested():
                return
            cs_health = probe(identifier, 'cryosparc')
            api_health = probe(identifier, 'api') if cs_health.get('healthy') else None
            failed = next((p for p in (cs_health, api_health) if p and not p.get('healthy')), None)
            if failed:
                healthy = 0
                record(failed['state'])
                if failed['state'] not in WAITING:
                    task = read_task(identifier)
                    task.update(status='failed', error='Service health check blocked; inspect supervisor status')
                    write(root / 'task.json', task)
                    return
                time.sleep(poll)
                continue
            healthy += 1
            if healthy < 2:
                record('service_stabilizing', consecutive_healthy=healthy)
                time.sleep(stable)
                continue
            task = read_task(identifier)
            jobs = cs_health['jobs']
            if baseline is None:
                saved = directory / 'adoption_baseline.json'
                baseline = json.loads(saved.read_text())['job_uids'] if saved.exists() else [j['uid'] for j in jobs]
            os.environ['CRYOAGENT_WEB_SUPERVISED'] = '1'
            try:
                if recovering or task.get('service_resume_checkpoint'):
                    if not task.get('workspace'):
                        raise ValueError('Initialization may have created an unrecorded workspace')
                    failure = root / 'controller_failure.json'
                    summary_times = [p.stat().st_mtime_ns for p in (root / 'agent').glob('run_*/summary.json')]
                    finalizing = (task.get('service_finalize_checkpoint') or
                                  failure.exists() and
                                  failure.stat().st_mtime_ns >= max(summary_times, default=0) and
                                  json.loads(failure.read_text()).get('phase') == 'finalizing')
                    checkpoint = recovery_checkpoint(root, task, baseline, jobs)
                    if finalizing and (root / 'finalization_checkpoint.json').exists():
                        task['service_finalize_checkpoint'] = True
                    elif checkpoint:
                        task['service_resume_checkpoint'] = str(checkpoint)
                    task['slurm_job'] = os.getenv('SLURM_JOB_ID', task.get('slurm_job'))
                    task.setdefault('service_recoveries', []).append(dict(at=now(), checkpoint=str(checkpoint) if checkpoint else None))
                    write(root / 'task.json', task)
            except ValueError as exc:
                record('blocked_ambiguous_submission', reason=str(exc))
                task.update(status='failed', error=str(exc))
                write(root / 'task.json', task)
                return
            baseline = [j['uid'] for j in jobs]
            write(directory / 'attempt_baseline.json', {'job_uids': baseline, 'at': now()})
            since = failure_versions(root)
            launches += 1
            record('running', attempt=launches)
            with (directory / f'worker-{time.time_ns()}.log').open('a') as log:
                result = subprocess.run([PYTHON, '-u', '-m', 'webapp.worker', identifier],
                                        cwd=REPO, stdout=log, stderr=log)
            state = failure_state(root, since)
            if state not in RECOVERABLE:
                record('stopped', worker_exit_code=result.returncode, reason=read_task(identifier).get('stop_reason'))
                return
            record(state, worker_exit_code=result.returncode)
            recovering = True
            healthy = 0
            time.sleep(poll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('identifier')
    parser.add_argument('--probe', choices=('cryosparc', 'api'))
    parser.add_argument('--adopt-controller')
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe_child(args.identifier, args.probe)))
    else:
        supervise(args.identifier, args.adopt_controller)


if __name__ == '__main__':
    main()
