"""GPU routing for production web controllers, not frozen experiments."""
import os
import json

LANES = tuple(x.strip() for x in os.getenv(
    'CRYOAGENT_WEB_GPU_LANES', 'g8m192_4090_slurm,h20_slurm').split(',') if x.strip())


def configure():
    if os.getenv('CRYOAGENT_WEB_GPU_ROUTING') != '1':
        return
    os.environ.pop('CRYOAGENT_GPU_LANE', None)
    os.environ['CRYOAGENT_GPU_LANES'] = ','.join(LANES)
    os.environ['CRYOAGENT_AUTO_SELECT_LANE'] = '1'
    os.environ.setdefault('CRYOAGENT_WEB_DYNAMIC_GPUS', '1')
    if os.environ['CRYOAGENT_WEB_DYNAMIC_GPUS'] == '1':
        os.environ['CRYOAGENT_GPU_COUNT_CAP'] = str(maximum_gpus())


def maximum_gpus():
    try:
        return max(1, min(8, int(os.getenv('CRYOAGENT_WEB_MAX_GPUS', '8'))))
    except ValueError:
        return 8


def allocation(capacity, *, supports_multi):
    """Use only measured same-node free GPUs, never the partition total."""
    lane = capacity.get('lane') if capacity.get('success') else LANES[0]
    row = next((r for r in capacity.get('observations', []) if r.get('lane') == lane), {})
    idle = row.get('gpu_idle_max_node', row.get('gpu_idle', 0))
    count = min(maximum_gpus(), max(1, idle if isinstance(idle, int) else 1)) if supports_multi else 1
    return dict(lane=lane, gpu_count=count, supports_multi=supports_multi,
                observed_idle_gpus=row.get('gpu_idle'), observed_same_node_idle=idle,
                maximum_gpus=maximum_gpus())


def protect_client_enqueue(client):
    """Cover every production queue path, including executor and pilot recovery.

    The hook belongs to this client, not the SDK class. It changes resource
    parameters only on unqueued Jobs with native multi-GPU support.
    """
    if getattr(client, '_web_gpu_enqueue_protected', False):
        return client
    if os.getenv('CRYOAGENT_WEB_GPU_ROUTING') != '1':
        return client
    original = client.api.jobs.enqueue

    def enqueue(project_uid, job_uid, **kwargs):
        if (os.getenv('CRYOAGENT_WEB_DYNAMIC_GPUS') != '1'
                or kwargs.get('lane') not in LANES
                or kwargs.get('hostname') or kwargs.get('gpus')):
            return original(project_uid, job_uid, **kwargs)
        job = client.find_job(project_uid, job_uid)
        spec = job.full_spec
        if job.status != 'building' or spec.interactive or spec.lightweight:
            return original(project_uid, job_uid, **kwargs)
        from job_specs import get_job_spec
        local = get_job_spec(job.type)
        if not local.get('requires_gpu'):
            return original(project_uid, job_uid, **kwargs)
        from resource_strategy import select_available_lane
        capacity = select_available_lane()
        supported = 'compute_num_gpus' in spec.params
        plan = allocation(capacity, supports_multi=supported)
        if supported:
            job.set_param('compute_num_gpus', plan['gpu_count'])
        kwargs['lane'] = plan['lane']
        print('WEB_GPU_ALLOCATION ' + json.dumps(dict(project=project_uid, job=job_uid,
            job_type=job.type, **plan)), flush=True)
        return original(project_uid, job_uid, **kwargs)

    client.api.jobs.enqueue = enqueue
    client._web_gpu_enqueue_protected = True
    return client


def choose_lane():
    """Resolve capacity immediately before queueing, not at workflow startup."""
    configure()
    fallback = os.getenv('CRYOAGENT_GPU_LANE') or LANES[0]
    if os.getenv('CRYOAGENT_AUTO_SELECT_LANE', '1') != '1':
        return fallback
    from resource_strategy import select_available_lane
    result = select_available_lane()
    # Queuing remains possible when both lanes are busy or inventory is unavailable.
    return result['lane'] if result.get('success') else fallback


def queue_and_wait(job, lane, timeout_s, poll_s):
    from prepare_visual_rl_batch import queue_and_wait as wait
    if str(job.status).lower() == 'building' and os.getenv('CRYOAGENT_WEB_GPU_ROUTING') == '1':
        lane = choose_lane()
        print(f'WEB_GPU_ROUTING job={job.uid} lane={lane}', flush=True)
    return wait(job, lane, timeout_s, poll_s)
