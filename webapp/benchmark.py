"""Explicit benchmark profile; ordinary web submissions are unchanged."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

PROFILE = 'codex_20261017'
TARGETS = {'11280': 3.5, '10228': 3.2}
BENCHMARK = Path('/hdd1/linweifan/cryosparc/benchmarks/agent_vs_codex_20261017')
SUPPORT = Path(__file__).resolve().parents[1] / 'benchmarks/agent_vs_codex_20261017'


def validate_profile(payload, config):
    profile = payload.get('benchmark_profile')
    if not profile:
        return config
    if profile != PROFILE or config['empiar_id'] not in ('11280', '10228'):
        raise ValueError('Unsupported benchmark profile/dataset')
    repeat = payload.get('benchmark_repeat')
    if isinstance(repeat, bool) or repeat not in (1, 2, 3):
        raise ValueError('Benchmark repeat must be 1, 2 or 3')
    original = BENCHMARK / 'repeats/repeat03/codex' / ('empiar' + config['empiar_id']) / 'workspace/config.json'
    reference = json.loads(original.read_text())
    fields = {'pixel_size_A': 'pixel_size_A', 'voltage_kV': 'accelerating_voltage_kV',
              'cs_mm': 'spherical_aberration_mm', 'dose': 'total_dose_e_per_A2'}
    if any(not math.isclose(config[k], reference[v], rel_tol=0, abs_tol=1e-9) for k, v in fields.items()):
        raise ValueError('Acquisition must match the Codex reference configuration')
    if not math.isclose(config['target'], TARGETS[config['empiar_id']], rel_tol=0, abs_tol=1e-9):
        raise ValueError('Target must match the prespecified stricter frontend benchmark target')
    if config['input_path'] != reference['input_glob'] or config['input_type'] != reference['input_type']:
        raise ValueError('Benchmark raw input must match the Codex reference')
    if not config['kb'] or config['symmetry'] is not None:
        raise ValueError('Benchmark requires held-out KB enabled and initially unknown symmetry')
    config.update(benchmark_profile=PROFILE, benchmark_repeat=repeat, max_created_jobs=40,
                  max_jobs=80, benchmark_reference=str(original),
                  benchmark_reference_target=reference['target_resolution_A'])
    return config


def configure(task, root, client):
    config = task['config']
    if config.get('benchmark_profile') != PROFILE:
        return
    from webapp.core import write
    from benchmark_budget import protect_client
    root = Path(root)
    snapshot = BENCHMARK / 'shared/kb.sqlite'
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    budget = root / 'benchmark_budget.json'
    identity = dict(project=task['project'], workspace=task['workspace'], maximum=40)
    if budget.exists():
        if json.loads(budget.read_text()) != identity:
            raise ValueError('Benchmark budget identity changed')
    else:
        if list(client.find_project(task['project']).find_workspace(task['workspace']).find_jobs()):
            raise ValueError('Benchmark must start in a fresh empty workspace')
        write(budget, identity)
    os.environ.update(CRYOAGENT_BENCHMARK_BUDGET_FILE=str(budget),
        CRYOAGENT_BENCHMARK_KB_SNAPSHOT=str(snapshot),
        CRYOAGENT_BENCHMARK_KB_AUDIT=str(root / 'kb_queries.jsonl'),
        CRYOAGENT_KNOWN_WORKFLOW_DIRS='', CRYOAGENT_GPU_COUNT_CAP='1',
        CRYOAGENT_AUTO_SELECT_LANE='0', CRYOAGENT_GPU_LANE='g8m192_4090_slurm')
    os.environ['CRYOAGENT_WEB_DYNAMIC_GPUS'] = '0'
    record = root / 'benchmark_protocol.json'
    protocol = dict(profile=PROFILE, reference=config['benchmark_reference'], kb_snapshot=str(snapshot),
        kb_sha256=digest, max_created_jobs=40, counting='All actual Job IDs, including import, pilots, failed jobs and validation',
        max_decision_rounds=80, gpu_lane='g8m192_4090_slurm', max_gpus_per_job=1,
        target_resolution_A=config['target'], reference_target_resolution_A=config['benchmark_reference_target'],
        known_workflow_retrieval=False, reference_maps=False, symmetry='unknown',
        models=task.get('model_config'),
        deliberate_difference='Stricter frontend stopping targets; current web Agent planner/executor models versus Codex Sol medium; tool interfaces differ')
    if record.exists() and json.loads(record.read_text()) != protocol:
        raise ValueError('Benchmark protocol changed on recovery')
    write(record, protocol)
    protect_client(client)


def call_snapshot(tool_name, arguments):
    """Never fall through to the unfiltered original KB or its image files."""
    snapshot = Path(os.environ['CRYOAGENT_BENCHMARK_KB_SNAPSHOT'])
    spec = importlib.util.spec_from_file_location('heldout_web_kb', SUPPORT / 'kb_server.py')
    module = importlib.util.module_from_spec(spec)
    os.environ['BENCHMARK_KB_SNAPSHOT'] = str(snapshot)
    os.environ['BENCHMARK_KB_AUDIT'] = os.environ['CRYOAGENT_BENCHMARK_KB_AUDIT']
    spec.loader.exec_module(module)
    arguments = arguments or {}
    if tool_name == 'search_cryoem_kb':
        return module.query_docs(arguments.get('query', ''), arguments.get('top_k', 5), arguments.get('kb_types'))
    if tool_name == 'get_job_doc':
        return module.get_job_doc(arguments.get('job_type', ''), arguments.get('query', ''), arguments.get('top_k', 5))
    if tool_name == 'get_images':
        return dict(source='frozen_heldout_kb', results=[], workflow_case=None,
                    notice='Original KB image/raw-file access is disabled; use frozen document retrieval only.')
    if tool_name in ('get_dataset_summary', 'get_workflow', 'get_maps', 'get_manual_annotations'):
        result = module.query_docs(arguments.get('dataset_id', ''), arguments.get('top_k', 5))
        result['workflow_case'] = None
        return result
    if tool_name in ('find_similar_cases', 'get_next_steps', 'get_failures'):
        return module.query_docs(arguments.get('job_type', '') or arguments.get('molecule_type', ''),
                                 arguments.get('top_k', 5))
    raise ValueError('Unsupported held-out KB tool: ' + tool_name)
