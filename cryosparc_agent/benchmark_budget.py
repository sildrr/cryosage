"""Opt-in actual-Job cap shared by the web worker and its MCP subprocesses."""
import fcntl
import inspect
import json
import os
from pathlib import Path
import uuid


class BenchmarkBudgetReached(RuntimeError):
    pass


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def budget_status(client=None):
    path = os.getenv('CRYOAGENT_BENCHMARK_BUDGET_FILE')
    if not path:
        return None
    state = json.loads(Path(path).read_text())
    if client is None:
        from cryosparc_client import cryosparc_client
        client = cryosparc_client()
    jobs = list(client.find_project(state['project']).find_workspace(state['workspace']).find_jobs())
    ids = sorted(j.uid for j in jobs)
    return dict(project=state['project'], workspace=state['workspace'],
                used=len(ids), maximum=state['maximum'], job_uids=ids,
                remaining=max(0, state['maximum'] - len(ids)))


def protect_client(client):
    path = os.getenv('CRYOAGENT_BENCHMARK_BUDGET_FILE')
    if not path or getattr(client, '_benchmark_protected', False):
        return client
    path = Path(path)

    def guarded(original, requested_type=None):
        signature = inspect.signature(original)

        def create_job(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            with path.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = json.loads(path.read_text())
                if (bound.arguments['project_uid'], bound.arguments['workspace_uid']) != (
                        state['project'], state['workspace']):
                    raise ValueError('Benchmark creation outside assigned workspace is forbidden')
                marker = path.with_suffix('.intent.json')
                if marker.exists():
                    raise RuntimeError('Ambiguous prior benchmark submission; reconcile before retrying')
                before = budget_status(client)
                if before['remaining'] == 0:
                    raise BenchmarkBudgetReached(f"Actual new-Job budget exhausted ({state['maximum']} including import/pilots/failures)")
                receipt = dict(before=before, requested_type=bound.arguments.get('type') or requested_type)
                destination = path.parent / 'budget_receipts' / (uuid.uuid4().hex + '.json')
                write(marker, receipt)
                try:
                    result = original(*args, **kwargs)
                except Exception as exc:
                    receipt['error'] = str(exc)
                    # Creation can already have produced an ID. Keep the
                    # intent on any exception rather than blindly resubmitting.
                    write(destination, receipt)
                    raise
                after = budget_status(client)
                created = set(after['job_uids']) - set(before['job_uids'])
                receipt.update(after=after, created_job_uids=sorted(created))
                write(destination, receipt)
                if len(created) != 1 or after['used'] > state['maximum']:
                    raise RuntimeError('Benchmark creation differs from admitted one-Job budget')
                marker.unlink()
                return result

        return create_job

    client.create_job = guarded(client.create_job)
    # External jobs call api.jobs.create directly, not client.create_job.
    # Count them under the same admission lock and durable creation intent.
    if hasattr(client, 'create_external_job'):
        client.create_external_job = guarded(client.create_external_job, 'snowflake')
    client._benchmark_protected = True
    return client
