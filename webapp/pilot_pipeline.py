"""Production adapter for the validated serial pilot-to-full strategy.

No benchmark dataset IDs, historical particles, maps or acquisition defaults are
used here. A version is pinned when a web task is submitted, not on recovery.
"""
import asyncio
import copy
import json
import os
from pathlib import Path

from webapp.core import now, write

VERSION = 'web-pilot-to-full-v1'


def enabled(task):
    return task.get('pipeline_version') == VERSION and task['config']['input_type'] != 'particles'


def pin(task, payload):
    if task['config'].get('benchmark_profile'):
        return
    task['pipeline_version'] = VERSION
    maximum = int(payload.get('max_created_jobs', 100))
    if isinstance(payload.get('max_created_jobs'), bool) or not 10 <= maximum <= 200:
        raise ValueError('Actual Job budget must be between 10 and 200')
    task['config'].update(max_created_jobs=maximum, decision_round_limit=300)


def configure(task, root, client):
    if task.get('pipeline_version') != VERSION:
        return
    from benchmark_budget import protect_client
    identity = dict(project=task['project'], workspace=task['workspace'],
                    maximum=task['config']['max_created_jobs'])
    path = root / 'benchmark_budget.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Pinned production Job budget changed')
    if not path.exists():
        if list(client.find_project(task['project']).find_workspace(task['workspace']).find_jobs()):
            raise ValueError('New strategy requires an empty fresh workspace')
        write(path, identity)
    os.environ.update(CRYOAGENT_BENCHMARK_BUDGET_FILE=str(path),
                      CRYOAGENT_BENCHMARK_KB_AUDIT=str(root / 'kb_queries.jsonl'),
                      CRYOAGENT_VISION_CACHE_DIR=str(root / 'vision_inputs'))
    from webapp.gpu_routing import configure as configure_gpu_routing
    configure_gpu_routing()
    os.environ.setdefault('CRYOAGENT_API_MAX_REQUEST_BYTES', str(8 * 1024**2))
    protect_client(client)


def create_preprocessing(task, root, workspace, key, kind, params, connections=None):
    """Persist creation intent before mutation; an uncertain creation is not retried."""
    path = root / 'preprocessing.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    record = state.setdefault(key, {})
    contract = dict(kind=kind, params=params, connections=connections or {})
    # JSON restores tuples as lists, so compare canonical serialized values.
    if record.get('contract') and json.dumps(record['contract'], sort_keys=True) != json.dumps(contract, sort_keys=True):
        raise ValueError('Preprocessing contract changed: ' + key)
    if not record.get('job_uid'):
        if record.get('creation_intent'):
            raise RuntimeError('Ambiguous preprocessing creation; reconcile before retry: ' + key)
        record.update(contract=contract, creation_intent=True)
        write(path, state)
        job = workspace.create_job(kind, params=params, connections=connections or {},
                                   title='Web ' + task['id'] + ' ' + key)
        record['job_uid'] = job.uid
        write(path, state)
    return workspace.find_job(record['job_uid'])


def preprocessing(task, root, workspace, imported):
    from webapp.gpu_routing import queue_and_wait
    lane = os.getenv('CRYOAGENT_GPU_LANE', 'g8m192_4090_slurm')
    source, output = imported, 'imported_micrographs'
    if task['config']['input_type'] == 'movies':
        source = create_preprocessing(task, root, workspace, 'motion', 'patch_motion_correction_multi',
                                      {'compute_num_gpus': 1}, {'movies': (imported.uid, 'imported_movies')})
        if queue_and_wait(source, lane, 86400, 30) != 'completed':
            raise RuntimeError('Motion correction failed: ' + source.uid)
        output = 'micrographs'
    ctf = create_preprocessing(task, root, workspace, 'ctf', 'patch_ctf_estimation_multi',
                               {'compute_num_gpus': 1}, {'exposures': (source.uid, output)})
    if queue_and_wait(ctf, lane, 86400, 30) != 'completed':
        raise RuntimeError('Patch CTF failed: ' + ctf.uid)
    return ctf


def make_manifest(task, root, ctf, count, scale=None):
    from two_tier_picker import PROTOCOL
    c = task['config']
    scales = sorted({min(n, count) for n in (50, 100, 200)})
    if count < 1:
        raise ValueError('Patch CTF has no usable micrographs')
    result = dict(production_pipeline=VERSION, protocol='sequential-picker-v1', protocol_revision=4,
        experimental_gate=PROTOCOL, dataset=task['id'], empiar_id=c['empiar_id'], task_id=task['id'],
        project=task['project'], workspace=task['workspace'], shared_jobs=[task['import_job'], ctf.uid],
        full_micrographs=count, pilot_scales=scales, target_resolution_A=c['target'],
        acquisition_facts=dict(psize_A=c['pixel_size_A'], accel_kv=c['voltage_kV'], cs_mm=c['cs_mm'],
                               total_dose_e_per_A2=c['dose']),
        symmetry=c.get('symmetry'), kb_enabled=c['kb'], models=task['model_config'],
        api_override=dict(base_url=os.getenv('OPENAI_BASE_URL', 'https://api.ofox.ai/v1')),
        max_new_jobs=c['max_created_jobs'], max_seed_jobs=max(1, c['max_created_jobs'] - 20),
        max_stage_reviews=3, max_adjustment_rounds=3, max_confirmation_adjustments=1,
        pilot_core_refinement='homo_only', early_symmetry_assessment=True,
        extract_scale_memory=True, occasional_neighbor_picks_acceptable=True,
        signal_aware_expansion=True, cross_stage_scale_consistency=True,
        route_comparison='same_qualified_pilot_once', diameter_recovery_enabled=True, diameter_recovery_rounds=3,
        gate=dict(duplicate_ncc=.90), created_at=now())
    if scale:
        result['cryolo_initial_scale'] = scale
    return result


def picker_class():
    # Lazy imports keep the HTTP process independent of scientific dependencies.
    from two_tier_picker import TwoTierPicker, gate_policy
    from pilot_reclassification_experiment import PilotExperiment
    from sequential_picker_rollout import SequentialPickerRollout

    class ProductionPicker(TwoTierPicker):
        def run(self):
            if (self.root/'pilot_diversity_approval.json').exists() and not self.state.get('pilot_accepted'):
                if self.state.get('stop_reason'):
                    self.state['pre_approval_stop_reason'] = self.state.pop('stop_reason')
                self.state['status'] = 'running'
                self.persist()
            return super().run()

        def guard_budget(self, slug):
            if (self.root/'pilot_diversity_approval.json').exists():
                # Reallocate unused Planner allowance, without raising the total Job cap.
                # Keep the in-memory limit through nested submit/External checks;
                # restoring it here made their second guard see the old 80-Job cap.
                self.manifest['max_seed_jobs'] = min(self.manifest['max_new_jobs'],
                    json.loads((self.root/'benchmark_budget.json').read_text())['maximum'])
            return super().guard_budget(slug)

        def pilot(self):
            from webapp.pilot_approval import apply_approval
            accepted = apply_approval(self)
            return accepted if accepted else super().pilot()

        def confirm(self, recipe):
            from webapp.pilot_approval import approved_diversity
            if approved_diversity(recipe['promotion_gate']):
                # Continue the approved route, not another optional picker comparison.
                self.state['promoted_symmetry'] = self.state.get('symmetry_by_refine', {}).get(recipe['best_refine'], 'C1')
                self.persist()
                return dict(recipe, kind='blob', gate=recipe['promotion_gate'])
            return super().confirm(recipe)

        def __init__(self, root):
            PilotExperiment.__init__(self, root)
            if self.manifest.get('production_pipeline') != VERSION:
                raise ValueError('Explicit production pipeline manifest required')
            self.scales = tuple(self.manifest['pilot_scales'])
            self.allowed_samples = (*self.scales, self.manifest['full_micrographs'])
            self.policy = gate_policy(self.manifest)
            self.cryosift_threshold = 3.
            self.api_base = self.manifest['api_override']['base_url']
            self.acquisition_facts = dict(self.manifest['acquisition_facts'])
            self.native_pixel_size_A = self.acquisition_facts['psize_A']
            self.initial_scale()

        def boundary(self):
            from operator_control import decision_boundary
            async def reject_jump(target):
                raise ValueError('During the structured pilot, use textual guidance; node jumps are available in the later Planner phase')
            result = asyncio.run(decision_boundary(str(self.root/'control'), None, reject_jump))
            if result['stop']:
                raise PipelineStopped('User stopped structured pilot')
            self.guidance = result.get('guidance')

        def service(self, name, function):
            def checked():
                self.boundary()
                return function()
            return super().service(name, checked)

        def submit(self, slug, kind, sources, params, wait=True):
            self.boundary()
            if self.manifest.get('symmetry'):
                self.state['promoted_symmetry'] = self.manifest['symmetry']
                if kind in {'homo_refine_new', 'nonuniform_refine_new'}:
                    params = dict(params, refine_symmetry=self.manifest['symmetry'])
            return super().submit(slug, kind, sources, params, wait)

        def model(self, slug, messages, model, validate):
            self.boundary()
            if self.guidance:
                messages = copy.deepcopy(messages)
                messages.insert(0, dict(role='system', content='Current user guidance: ' + self.guidance['text']))
            return super().model(slug, messages, model, validate)

        def core_refinements(self, tag, core, initial):
            if self.manifest.get('symmetry'):
                return SequentialPickerRollout.core_refinements(self, tag, core, initial)
            return super().core_refinements(tag, core, initial)

        def review(self, slug, context, role, excluded=(), reference=None):
            ids = super().review(slug, context, role, excluded, reference)
            if role == 'core' and slug == 'full_core_review':
                path = self.root/'evidence/full_core_context.json'
                self.write(path, context)
                self.state['current_core'] = dict(source=context['source'], class_ids=ids, context_file=str(path))
                self.persist()
            return ids

        def persist(self):
            super().persist()
            write(self.root/'pilot_progress.json', dict(version=VERSION, updated_at=now(),
                status=self.state.get('status'), full_data_promoted=bool(self.state.get('full')),
                pilot_accepted=bool(self.state.get('pilot_accepted')),
                created_jobs=len({v.get('job_uid') for v in self.state.get('steps', {}).values() if v.get('job_uid')}),
                stop_reason=self.state.get('stop_reason'), service_wait=self.state.get('service_wait')))
    return ProductionPicker


class PipelineStopped(RuntimeError):
    pass


def run(task, root, project, workspace, imported):
    from webapp.cryolo_probe import estimate
    from two_tier_picker import POLICY
    os.environ['CRYOAGENT_WEB_WORKER_PHASE'] = 'pilot'
    path = root/'manifest.json'
    if not path.exists():
        ctf = preprocessing(task, root, workspace, imported)
        import numpy as np
        exposures = ctf.load_output('exposures')
        if not np.allclose(exposures['micrograph_blob/psize_A'], task['config']['pixel_size_A'], rtol=1e-5):
            raise ValueError('CTF micrograph pixel size differs from supplied acquisition; do not apply miscalibrated sizing')
        count = len(exposures)
        scale = estimate(task, root, project, ctf)
        write(path, make_manifest(task, root, ctf, count, scale))
    manifest = json.loads(path.read_text())
    if (manifest['task_id'], manifest['project'], manifest['workspace']) != (task['id'], project.uid, workspace.uid):
        raise ValueError('Pilot manifest identity mismatch')
    experiment = picker_class()(root)
    best = experiment.run()
    if best:
        context = root/'coverage_planner_context.json'
        write(context, dict(project_uid=project.uid, workspace_uid=workspace.uid, source=VERSION,
            guidance=POLICY, best_refine=best, chosen_route=experiment.state['chosen_route'],
            note='Qualified pilot and full-data reconstruction complete. Preserve the best checkpoint.'))
        os.environ['CRYOAGENT_COVERAGE_CONTEXT'] = str(context)
        # Baseline is written once, before the first downstream Planner launch.
        baseline = root/'service_supervision/adoption_baseline.json'
        if not baseline.exists():
            write(baseline, dict(at=now(), job_uids=[j.uid for j in workspace.find_jobs()]))
    return best
