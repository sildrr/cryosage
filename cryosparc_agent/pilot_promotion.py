"""Audited, conditional launch of two full runs from an objectively passing pilot."""
import base64
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid


def promotion_trial(state):
    if state.get('status') != 'pilot_usable_for_promotion':
        return None
    trials = [t for t in state.get('trials', []) if t.get('outcome') == 'pilot_usable_for_promotion'
              and (t.get('objective_gate') or {}).get('passed') is True
              and (t.get('objective_gate') or {}).get('available') is True]
    if len(trials) != 1:
        raise ValueError('Expected exactly one passing objective trial')
    return trials[0]


def _input(job, name):
    c = job.model.model_dump()['spec']['inputs'][name].get('connections', [])
    if len(c) != 1:
        raise ValueError('Promotion requires a single source per input')
    return c[0]['job_uid'], c[0]['output']


def build_recipe(root, project):
    from collections import Counter
    from pilot_objective_gate import GatePolicy, VERSION, evaluate_gate, load_class_images
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    state = json.loads((root/'state.json').read_text())
    if manifest.get('protocol_version') != VERSION:
        raise ValueError('Only the V2 objective protocol can trigger full runs')
    trial = promotion_trial(state)
    if trial is None:
        return None
    policy = GatePolicy(**manifest['gate_policy'])
    if state.get('gate_policy') != asdict(policy) or trial['objective_gate'].get('policy') != asdict(policy):
        raise ValueError('Objective policy changed after the trial')
    context = json.loads((root/'evidence'/f"n{trial['micrographs']}_after"/'context.json').read_text())
    after = project.find_job(trial['after_job'])
    if context.get('source') != dict(project_uid=project.uid, job_uid=after.uid):
        raise ValueError('Wrong source class evidence')
    jobs = {name: project.find_job(trial[name]) for name in ('picker','inspect','extract','before_job','select','after_job')}
    for job in jobs.values():
        if job.status != 'completed' or manifest['workspace'] not in job.model.model_dump().get('workspace_uids', []):
            raise ValueError('Promotion source is incomplete or outside pilot workspace')
    checks = [(jobs['inspect'],'particles',(jobs['picker'].uid,'particles')),
              (jobs['extract'],'particles',(jobs['inspect'].uid,'particles')),
              (jobs['before_job'],'particles',(jobs['extract'].uid,'particles')),
              (jobs['select'],'particles',(jobs['before_job'].uid,'particles')),
              (jobs['after_job'],'particles',(jobs['select'].uid,'particles_selected')),
              (jobs['picker'],'micrographs',(manifest['shared_jobs'][-1],'exposures'))]
    if any(_input(job,name) != expected for job,name,expected in checks):
        raise ValueError('Validated pilot lineage changed')
    counts = Counter(map(int, after.load_output('particles')['alignments2D/class']))
    if any(counts.get(r['class_id'],0) != r['particle_count'] for r in context['class_statistics']):
        raise ValueError('Native source populations differ from scored evidence')
    gate = evaluate_gate(context, load_class_images(project,after.uid), policy)
    if not gate.get('passed') or not gate.get('available'):
        raise ValueError('Objective gate did not pass fresh read-only verification')
    if gate['promotion_class_ids'] != trial['objective_gate']['promotion_class_ids']:
        raise ValueError('Qualified class IDs changed after validation')
    names = dict(picker=('diameter','diameter_max','use_circle','use_ellipse','use_ring','min_distance',
                         'use_ctf','lowpass_res','lowpass_res_template','max_num_hits'),
                 inspect=('ncc_score_thresh','lpower_thresh_min','lpower_thresh_max','calibrate_ncc','calibrate_pow','do_auto_cluster'),
                 extract=('box_size_pix','bin_size_pix','output_f16'))
    params = {}
    for stage, keys in names.items():
        native = jobs[stage].model.model_dump()['spec']['params']
        params[stage] = {k:native[k] for k in keys if native.get(k) is not None}
    if any(params['inspect'].get(k) is not False for k in ('calibrate_ncc','calibrate_pow','do_auto_cluster')):
        raise ValueError('Promotion requires explicit raw-score Inspect settings')
    if params['extract'].get('bin_size_pix', params['extract']['box_size_pix']) != params['extract']['box_size_pix']:
        raise ValueError('Promotion must preserve native extraction sampling')
    params['extract'].pop('bin_size_pix',None)
    return dict(protocol_version=VERSION, project=manifest['project'], pilot_workspace=manifest['workspace'],
        source_class_job=after.uid, selected_class_ids=gate['promotion_class_ids'], objective_gate=gate,
        parameters=params, classification_hints=trial['parameters'].get('classify',{}),
        reclassification_hints=trial['parameters'].get('reclassify',{}),
        contact_sheet=context['contact_sheet'], shared_jobs=manifest['shared_jobs'],
        kb_snapshot=manifest['kb_snapshot'], kb_sha256=manifest['kb_sha256'])


def append_reference(messages, candidates, path, project_uid, workspace_uid):
    """Opt-in visual reference; never changes allowed actions or reuses pilot IDs."""
    if not path or not any(c.get('job_type') in {'select_2D','class_2D_new'} for c in candidates):
        return messages
    value = json.loads(Path(path).read_text())
    if (value.get('project_uid'),value.get('workspace_uid')) != (project_uid,workspace_uid):
        raise ValueError('Pilot reference belongs to another full run')
    recipe = value['recipe']
    if hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest() != value.get('recipe_sha256'):
        raise ValueError('Pilot reference recipe hash mismatch')
    if not recipe['objective_gate'].get('passed') or not recipe['objective_gate'].get('available'):
        raise ValueError('Unvalidated pilot reference is forbidden')
    url = recipe['contact_sheet'].get('data_url')
    if not isinstance(url,str) or not url.startswith('data:image/png;base64,') or len(url)>32*1024**2:
        raise ValueError('Invalid pilot reference image')
    import io
    from PIL import Image
    with Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1],validate=True))) as image:
        image.verify()
    text = ('Validated small-pilot morphology reference ONLY, not a current full-data class sheet or ground truth. '
        'Only the listed passing pilot IDs are recommended references; other tiles are not validated. '
        'IDs belong to the pilot source job and MUST NOT be copied as full-data selected_templates. '
        'Choose full-data class IDs only from the current source images and legal candidates. '
        'No pilot particles may be used for full-data reconstruction. Class-count/window hints may '
        'be adjusted for the much larger full-data population.\n' + json.dumps(dict(
            source_class_job=recipe['source_class_job'], passing_pilot_ids=recipe['selected_class_ids'],
            classification_hints=recipe['classification_hints'], reclassification_hints=recipe['reclassification_hints'],
            diversity_kind=recipe['objective_gate']['diversity_kind'])))
    return [*messages, dict(role='user',content=[dict(type='text',text=text),
        dict(type='image_url',image_url=dict(url=url))])]


def launch_full_repeats(root):
    from webapp.core import write, now, job_path, REPO
    from cryosparc_client import cryosparc_client
    from scripts.launch_optimized_repeats import DATASETS, MODELS
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    config = manifest.get('conditional_full_runs') or {}
    if config.get('repeats',0) == 0:
        return
    if config['repeats'] != 2:
        raise ValueError('This promotion protocol requires exactly two full repeats')
    directory = root/'promotion'
    directory.mkdir(exist_ok=True)
    with (directory/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = directory/'full_runs.json'
        state = json.loads((root/'state.json').read_text())
        if promotion_trial(state) is None:
            if not ledger_path.exists():
                write(ledger_path,dict(status='not_triggered',reason='pilot_did_not_pass_objective_gate',runs={}))
            return
        project = cryosparc_client().find_project(manifest['project'])
        recipe = build_recipe(root,project)
        if recipe is None:
            raise ValueError('Pilot state changed during promotion verification')
        if hashlib.sha256(Path(recipe['kb_snapshot']).read_bytes()).hexdigest() != recipe['kb_sha256']:
            raise ValueError('Frozen KB snapshot changed')
        encoded = json.dumps(recipe,sort_keys=True).encode()
        fingerprint = hashlib.sha256(encoded).hexdigest()
        value = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(
            status='preparing',recipe_sha256=fingerprint,runs={},created_at=now())
        if value.get('recipe_sha256') != fingerprint:
            raise ValueError('Promotion recipe differs from existing launch ledger')
        write(ledger_path,value)
        write(directory/'recipe.json',recipe)
        # Pilot budget is not applicable to new independent full workspaces.
        old_budget = os.environ.pop('CRYOAGENT_BENCHMARK_BUDGET_FILE',None)
        try:
            for repeat in (1,2):
                key = str(repeat)
                entry = value['runs'].setdefault(key,dict(task_id=uuid.uuid4().hex))
                write(ledger_path,value)
                identifier = entry['task_id']
                task_root = job_path(identifier)
                if not entry.get('workspace'):
                    if entry.get('creation_intent'):
                        raise RuntimeError('Ambiguous workspace creation; reconcile before retrying: '+identifier)
                    entry['creation_intent'] = now()
                    write(ledger_path,value)
                    ws = project.create_workspace(title=f'11280 pilot-v2 validated full repeat {repeat} {identifier[:8]}',
                        desc=f'Conditional pilot {manifest["workspace"]}, task {identifier}. 40 new Jobs; shared validated recipe, independent full-data picking/extraction; no pilot particles.')
                    entry['workspace'] = ws.uid
                    write(ledger_path,value)
                ws = project.find_workspace(entry['workspace'])
                if not entry.get('prepared'):
                    linked = {j.uid for j in ws.find_jobs()}
                    for uid in recipe['shared_jobs']:
                        if uid not in linked:
                            project.cs.api.jobs.link_to_workspace(project.uid,uid,ws.uid)
                    spec = DATASETS['11280']
                    task = dict(id=identifier,created_at=now(),status='prepared',project=project.uid,workspace=ws.uid,
                        config=dict(name=f'11280 validated pilot-v2 full repeat {repeat}',empiar_id='11280',input_type='movies',
                            input_path=spec['pattern'],input_file_count=spec['count'],pixel_size_A=spec['pixel'],voltage_kV=300.,
                            cs_mm=2.7,dose=spec['dose'],target=config['target_resolution_A'],symmetry=None,kb=True,
                            max_jobs=160,external_consent=True),model_config=MODELS,import_job=recipe['shared_jobs'][0],
                        shared_upstream_jobs=recipe['shared_jobs'],controls_supported=True,experiment=str(root))
                    write(task_root/'task.json',task)
                    write(task_root/'benchmark_budget.json',dict(project=project.uid,workspace=ws.uid,
                        maximum=config['max_new_jobs']+len(recipe['shared_jobs'])))
                    write(task_root/'validated_pilot_reference.json',dict(project_uid=project.uid,workspace_uid=ws.uid,
                        recipe_sha256=fingerprint,recipe=recipe))
                    write(task_root/'protocol.json',dict(parent_pilot=str(root),recipe_sha256=fingerprint,
                        target_resolution_A=config['target_resolution_A'],max_new_jobs=config['max_new_jobs'],
                        shared_pilot_recipe=True,independent_full_picking=True,model_config=MODELS,
                        kb_snapshot=recipe['kb_snapshot'],kb_sha256=recipe['kb_sha256'],no_pilot_particles_for_reconstruction=True))
                    write(task_root/'source_hashes_at_promotion.json', {
                        str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for folder in ('cryosparc_agent','webapp','scripts') for p in (REPO/folder).glob('*.py')})
                    entry.update(prepared=True,output=str(task_root))
                    write(ledger_path,value)
                if entry.get('controller'):
                    continue
                if entry.get('submission_intent'):
                    raise RuntimeError('Ambiguous controller submission; reconcile before retrying: '+identifier)
                entry['submission_intent'] = now()
                write(ledger_path,value)
                task = json.loads((task_root/'task.json').read_text())
                task['status'] = 'queued'
                write(task_root/'task.json',task)
                log = str(REPO/'logs'/f'pilot-full-11280-r{repeat}-%j.log')
                receipt = subprocess.check_output(['sbatch','--parsable',f'--job-name=pilot-full-11280-r{repeat}',
                    '--comment=cryoagent:'+identifier,'--output='+log,'--error='+log,
                    str(REPO/'scripts/pilot_full_repeat_controller.sbatch'),str(root),str(repeat)],text=True).strip().split(';')[0]
                if not receipt.isdigit():
                    raise RuntimeError('Ambiguous Slurm receipt')
                entry['controller'] = receipt
                write(ledger_path,value)
                task['slurm_job'] = receipt
                write(task_root/'task.json',task)
                print(json.dumps(dict(full_repeat=repeat,workspace=ws.uid,controller=receipt)),flush=True)
            value['status'] = 'submitted'
            write(ledger_path,value)
        finally:
            if old_budget is not None:
                os.environ['CRYOAGENT_BENCHMARK_BUDGET_FILE'] = old_budget


def run_full_repeat(root,repeat):
    from scripts.launch_optimized_repeats import runtime
    runtime('11280',inherit_server=False)
    from webapp.proxy import configure_compute_proxy
    configure_compute_proxy()
    from webapp.worker import configure_execution_policy
    configure_execution_policy()
    from webapp.core import write, now
    from webapp.usage import initialize_usage
    from cryosparc_client import cryosparc_client
    from pilot_reclassification_experiment import PilotExperiment
    root = Path(root)
    ledger = json.loads((root/'promotion/full_runs.json').read_text())
    entry = ledger['runs'][str(repeat)]
    target = Path(entry['output'])
    recipe = json.loads((root/'promotion/recipe.json').read_text())
    if hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest() != ledger['recipe_sha256']:
        raise ValueError('Promotion recipe hash mismatch')
    if not recipe['objective_gate'].get('passed') or not recipe['objective_gate'].get('available'):
        raise ValueError('Full-run recipe has not passed the objective gate')
    if hashlib.sha256(Path(recipe['kb_snapshot']).read_bytes()).hexdigest() != recipe['kb_sha256']:
        raise ValueError('Frozen KB snapshot changed')
    os.environ.update(CRYOAGENT_BENCHMARK_BUDGET_FILE=str(target/'benchmark_budget.json'),
        CRYOAGENT_USAGE_LEDGER=str(target/'api_usage.jsonl'),CRYOAGENT_BENCHMARK_KB_AUDIT=str(target/'kb_queries.jsonl'),
        CRYOAGENT_VISION_CACHE_DIR=str(target/'vision_inputs'),CRYOAGENT_GPU_LANE='g8m192_4090_slurm',
        CRYOAGENT_GPU_COUNT_CAP='1',CRYOAGENT_AUTO_SELECT_LANE='0',
        CRYOAGENT_VALIDATED_PILOT_REFERENCE=str(target/'validated_pilot_reference.json'))
    initialize_usage(target)
    with (target/'full_controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        cs = cryosparc_client()
        task = json.loads((target/'task.json').read_text())
        if task.get('status') == 'finished':
            return
        project = cs.find_project(task['project'])
        seed = PilotExperiment.__new__(PilotExperiment)
        seed.root,seed.write,seed.project,seed.workspace = target,write,project,project.find_workspace(task['workspace'])
        seed.path = target/'seed_state.json'
        seed.state = json.loads(seed.path.read_text()) if seed.path.exists() else dict(status='seeding',steps={})
        count = project.find_job(recipe['shared_jobs'][-1]).outputs['exposures'].num_items
        seed.allowed_samples = (count,)
        if not seed.state.get('finished'):
            picker = seed.submit('full_blob','blob_picker_gpu',{'micrographs':(recipe['shared_jobs'][-1],'exposures')},
                dict(recipe['parameters']['picker'],num_process=count))
            inspect = seed.submit('full_inspect','inspect_picks_v2',dict(micrographs=(picker.uid,'micrographs'),particles=(picker.uid,'particles')),
                recipe['parameters']['inspect'])
            extracted = seed.submit('full_extract','extract_micrographs_multi',dict(micrographs=(inspect.uid,'micrographs'),particles=(inspect.uid,'particles')),
                dict(recipe['parameters']['extract'],num_extract=count,compute_num_gpus=1))
            seed.state.update(finished=True,status='seeded',resume_node=extracted.uid)
            seed.persist()
        task = json.loads((target/'task.json').read_text())
        task.update(resume_node=seed.state['resume_node'],status='queued')
        if seed.state.get('main_started'):
            from webapp.supervisor import recovery_checkpoint
            baseline_path = target/'service_supervision/adoption_baseline.json'
            baseline = json.loads(baseline_path.read_text())['job_uids']
            jobs = [dict(uid=j.uid,status=j.status) for j in seed.workspace.find_jobs()]
            checkpoint = recovery_checkpoint(target,task,baseline,jobs)
            if not checkpoint:
                raise RuntimeError('Full Agent already started but has no recoverable checkpoint; do not replay seed')
            task['service_resume_checkpoint'] = str(checkpoint)
        write(target/'task.json',task)
        if not (target/'service_supervision/adoption_baseline.json').exists():
            write(target/'service_supervision/adoption_baseline.json',dict(at=now(),job_uids=[j.uid for j in seed.workspace.find_jobs()]))
        seed.state['main_started'] = True
        seed.persist()
        from webapp.supervisor import supervise
        supervise(entry['task_id'])
