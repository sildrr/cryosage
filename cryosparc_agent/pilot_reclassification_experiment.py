"""Isolated small-pilot experiment; does not modify the production candidate policy."""
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

SCALES = (20, 50, 100, 200)
SHARED = ('J1215', 'J1250', 'J1442')
MIN_SELECTED_FOR_SUCCESS = 2000


def native_thresholds(parameters):
    from inspect_score_contract import raw_parameters
    aliases = dict(ncc_score_min='ncc_score_thresh', local_power_min='lpower_thresh_min',
                   local_power_max='lpower_thresh_max')
    values = {}
    for key, value in parameters.items():
        target = aliases.get(key, key)
        if target in values and values[target] != value:
            raise ValueError('Conflicting Inspect aliases: ' + target)
        values[target] = value
    return raw_parameters(values)


def class_count(particles):
    """Avoid splitting a sparse pilot into the fixed production 50 classes."""
    if particles < 100:
        raise ValueError('Too few particles for this classification experiment')
    return max(5, min(50, particles // 500))


def selection(response, allowed):
    quality = response.get('quality')
    ids = response.get('selected_class_ids')
    if quality not in {'usable', 'uncertain', 'poor'} or not isinstance(response.get('reason'), str):
        raise ValueError('Expected quality usable|uncertain|poor and a reason')
    if not isinstance(ids, list) or any(type(i) is not int or i not in allowed for i in ids):
        raise ValueError('Select only integer IDs present in this source class job')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate class IDs')
    if quality == 'usable' and not ids:
        raise ValueError('A usable route needs at least one selected class')
    return response


def next_step(quality, selected_particles, scale):
    if quality == 'usable' and selected_particles >= MIN_SELECTED_FOR_SUCCESS:
        return 'pilot_usable_for_promotion'
    return 'expand' if scale < SCALES[-1] else 'pilot_exhausted_no_validated_route'


def stats(scores, counts):
    if not scores.get('success'):
        return dict(available=False, error=scores.get('error', scores.get('message')))
    values = {int(r['class_id']): float(r['score']) for r in scores.get('class_scores', [])
              if isinstance(r.get('score'), (float, int)) and math.isfinite(r['score'])}
    total = sum(counts.get(i, 0) for i in values)
    threshold = scores.get('threshold', 3.0)
    return dict(available=bool(values), class_count=len(values), scored_particles=total,
        minimum=min(values.values()) if values else None,
        particle_weighted_mean=sum(v * counts.get(i, 0) for i, v in values.items()) / total if total else None,
        recommended_particle_fraction=sum(counts.get(i, 0) for i, v in values.items() if v < threshold) / total if total else None,
        lower_is_better=True, threshold=threshold)


class PilotExperiment:
    def __init__(self, root):
        from webapp.core import write
        from cryosparc_client import cryosparc_client
        self.root, self.write = Path(root), write
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.cs = cryosparc_client()
        self.project = self.cs.find_project(self.manifest['project'])
        self.workspace = self.project.find_workspace(self.manifest['workspace'])
        self.path = self.root / 'state.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(status='running', steps={}, trials=[])

    def persist(self):
        from webapp.core import now
        self.state['updated_at'] = now()
        self.write(self.path, self.state)

    def report(self):
        import csv
        fields = ['micrographs', 'extracted_particles', 'selected_particles', 'before_job', 'after_job',
                  'before_weighted_score', 'after_weighted_score', 'before_recommended_fraction',
                  'after_recommended_fraction', 'credible_particles', 'quality', 'outcome']
        rows = []
        for t in self.state['trials']:
            row = {k: t.get(k) for k in fields}
            for prefix in ('before', 'after'):
                score = t.get(prefix+'_scores') or {}
                row[prefix+'_weighted_score'] = score.get('particle_weighted_mean')
                row[prefix+'_recommended_fraction'] = score.get('recommended_particle_fraction')
            row['quality'] = (t.get('assessment') or {}).get('quality')
            rows.append(row)
        with (self.root/'metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        lines = ['# Adaptive Pilot and Selected-Class Reclassification', '',
            f"Workspace: {self.project.uid}/{self.workspace.uid}. Status: {self.state['status']}.", '',
            'Only sample size and focused reclassification are explored. Picker diameter, raw Inspect '
            'thresholds and native extraction box stay fixed across sample sizes. Class count is '
            'mechanically sized to particle count (approximately 500 particles/class, 5..50).', '',
            '| Micrographs | Extracted | Selected for reclassification | Before Class 2D | After Class 2D | Outcome |',
            '| ---: | ---: | ---: | --- | --- | --- |']
        for t in self.state['trials']:
            lines.append('| ' + ' | '.join(str(t.get(k, 'pending')) for k in (
                'micrographs', 'extracted_particles', 'selected_particles', 'before_job', 'after_job', 'outcome')) + ' |')
        lines += ['', '## Interpretation', '',
            '- Full-data picking and 3D reconstruction are NOT submitted by this experiment.',
            '- CryoSift lower scores are better. An empty recommendation is distinct from scoring failure.',
            '- Before/after populations differ by selection. Better class scores alone do not prove better particles or preserved angular coverage.',
            '- usable + 2000 selected particles is a pilot stopping heuristic, not a gold-standard structural validation.',
            '- All raw API responses, images, class populations, selections and costs remain under this directory.']
        (self.root/'REPORT.md').write_text('\n'.join(lines)+'\n')

    def service(self, name, function):
        from webapp.supervisor import classify_error, WAITING
        while True:
            try:
                result = function()
                self.state.pop('service_wait', None)
                return result
            except Exception as exc:
                if classify_error(str(exc)) not in WAITING:
                    raise
                self.state['service_wait'] = dict(operation=name, error=str(exc)[:1500])
                self.persist()
                print(json.dumps(self.state['service_wait']), flush=True)
                time.sleep(120)

    def candidate(self, slug, kind, sources, params):
        from refinement_optimization import candidate
        return candidate('pilot_exp_' + slug, kind, sources, params,
                         'Controlled small-pilot reclassification experiment; exact recorded inputs/parameters.')

    def submit(self, slug, kind, sources, params, wait=True):
        from v2_decision_adapter import adapt_v2_decision_to_internal
        from action_registry import execute_model_decision_payload
        from job_executor import build_connections
        allowed_types = getattr(self, 'allowed_job_types',
            {'blob_picker_gpu', 'inspect_picks_v2', 'extract_micrographs_multi', 'class_2D_new', 'select_2D'})
        if kind not in allowed_types:
            raise ValueError('This experiment cannot submit full reconstruction or other job types')
        allowed_samples = getattr(self, 'allowed_samples', SCALES)
        if kind == 'blob_picker_gpu' and params.get('num_process') not in allowed_samples:
            raise ValueError('Full-data picking is forbidden in this pilot experiment')
        if kind == 'extract_micrographs_multi' and params.get('num_extract') not in allowed_samples:
            raise ValueError('Unbounded extraction is forbidden in this pilot experiment')
        c = self.candidate(slug, kind, sources, params)
        record = self.state['steps'].setdefault(slug, {})
        if not record.get('job_uid'):
            if record.get('submission_intent'):
                # A request might have created a Job before its response was lost.
                matches = self.service('reconcile ' + slug, lambda: [j for j in self.workspace.find_jobs()
                    if j.title == 'Agent ' + c['action_id']])
                if len(matches) != 1:
                    raise RuntimeError('Ambiguous submission; no blind resubmission: ' + slug)
                record['job_uid'] = matches[0].uid
                self.persist()
            else:
                decision = dict(schema_version='2.0', decision_type='forward', action=kind,
                    job_type=kind, action_id=c['action_id'], parameters=params,
                    reason='Execute the recorded model decision or fixed experimental classification step.')
                adapted = adapt_v2_decision_to_internal(decision, [c])['internal_decision']
                dry = execute_model_decision_payload(adapted, [c], dry_run=True)
                if not dry['success']:
                    raise ValueError('Candidate validation failed: ' + json.dumps(dry.get('issues')))
                record.update(submission_intent=True, candidate=c, decision=decision)
                self.persist()
                # Never retry creation on an uncertain transport outcome.
                result = execute_model_decision_payload(adapted, [c], dry_run=False,
                    project_uid=self.project.uid, workspace_uid=self.workspace.uid)
                record['execution'] = result
                rows = result.get('execution_results') or []
                if len(rows) == 1 and rows[0].get('job_uid'):
                    record['job_uid'] = rows[0]['job_uid']
                self.persist()
                if not result.get('success'):
                    raise RuntimeError('Native submission needs reconciliation: ' + slug)
        uid = record['job_uid']
        while True:
            job = self.service('poll ' + uid, lambda: self.project.find_job(uid))
            record['status'] = job.status
            self.persist()
            if job.status == 'completed':
                spec = job.model.model_dump()['spec']
                record['counts'] = {k: v.get('num_items') for k, v in spec.get('outputs', {}).items()}
                self.persist()
                return job
            if job.status in {'failed', 'killed'}:
                raise RuntimeError(f'Native Job {uid} {job.status}; trajectory retained')
            if job.status == 'building':
                actual = job.model.model_dump()['spec']['inputs']
                expected = build_connections(c['required_inputs'])
                for name, connection in expected.items():
                    matches = [(v['job_uid'], v['output']) for v in actual[name].get('connections', [])]
                    if matches != [connection]:
                        raise RuntimeError('Unqueued Job has incomplete inputs; do not duplicate: ' + uid)
                if kind in {'inspect_picks_v2', 'select_2D'}:
                    self.service('queue ' + uid, lambda: job.queue())
                else:
                    lane = os.getenv('CRYOAGENT_GPU_LANE', 'g8m192_4090_slurm') if getattr(self, 'manifest', {}).get('production_pipeline') else 'g8m192_4090_slurm'
                    if getattr(self, 'manifest', {}).get('production_pipeline') and os.getenv('CRYOAGENT_WEB_GPU_ROUTING') == '1':
                        from webapp.gpu_routing import choose_lane
                        lane = choose_lane()
                    self.service('queue ' + uid, lambda: job.queue(lane=lane))
            if not wait:
                return job
            time.sleep(30)

    def model(self, slug, messages, model, validate):
        from model_direct_runner import run_openai_compatible_model, resolve_api_key, parse_model_decision_text
        if getattr(self, 'manifest', {}).get('production_pipeline'):
            # Pilot calls interpret scientific/visual evidence; keep the selected
            # provider's exact model ID instead of experimental hardcoded names.
            model = self.manifest['models']['planner']
        directory = self.root / 'decisions' / slug
        if (directory / 'decision.json').exists():
            return validate(json.loads((directory / 'decision.json').read_text()))
        self.write(directory / 'messages.json', messages)
        working = list(messages)
        for attempt in range(3):
            response = self.service('model ' + slug, lambda: run_openai_compatible_model(
                working, getattr(self, 'api_base', 'https://api.ofox.ai/v1'), resolve_api_key(), model,
                max_new_tokens=4096, temperature=0, timeout_seconds=240, max_retries=2))
            self.write(directory / f'response_{attempt}.json', response)
            try:
                result = validate(parse_model_decision_text(response['raw_text']))
            except (ValueError, TypeError, KeyError) as exc:
                self.write(directory / f'invalid_{attempt}.json', dict(error=str(exc)))
                working += [dict(role='assistant', content=response['raw_text']),
                            dict(role='user', content='Correct this contract violation: ' + str(exc))]
                continue
            self.write(directory / 'decision.json', result)
            return result
        raise RuntimeError('Model contract failed three times: ' + slug)

    def stage(self, slug, stage, visual):
        from visual_decision_contract import build_stage_decision_messages
        from upstream_evidence import metadata
        from kb_bridge import call_kb_tool
        docs = self.service('generic KB docs', lambda: call_kb_tool('get_job_doc', dict(
            job_type={'blob_picker_diameter': 'blob_picker_gpu', 'inspect_picks_threshold': 'inspect_picks_v2',
                      'extract_box_size': 'extract_micrographs_multi'}[stage], top_k=2))) if getattr(self, 'manifest', {}).get('kb_enabled', True) else {'enabled': False}
        self.write(self.root / 'evidence' / (slug + '.json'), visual)
        context = dict(acquisition_facts=getattr(self, 'acquisition_facts',
            dict(psize_A=.856, accel_kv=300, cs_mm=2.7, total_dose_e_per_A2=68.4)),
            visual_context=metadata(visual), official_documentation=docs,
            experiment='Small pilot only; no known workflow, target identity, historical picks or reference map.')
        images = [i['data_url'] for i in [visual.get('contact_sheet') or {}, visual.get('score_scatter') or {},
                                         *visual.get('evidence_images', [])] if i.get('data_url')]
        messages = build_stage_decision_messages(stage, context, image_data_urls=list(dict.fromkeys(images)), include_reason=True)
        def validate(d):
            p = d['parameters']
            if stage == 'blob_picker_diameter':
                for k in ('diameter', 'diameter_max'):
                    if type(p.get(k)) not in (float, int) or not math.isfinite(p[k]) or p[k] <= 0:
                        raise ValueError('Provide positive finite diameter and diameter_max')
                if p['diameter_max'] < p['diameter']:
                    raise ValueError('Maximum diameter must not be below minimum')
            elif stage == 'inspect_picks_threshold':
                p = native_thresholds(p)
                d = {**d, 'parameters': p}
                for k in ('ncc_score_thresh', 'lpower_thresh_min', 'lpower_thresh_max'):
                    if type(p.get(k)) not in (float, int) or not math.isfinite(p[k]):
                        raise ValueError('Provide explicit finite native NCC/Power thresholds')
                if p['lpower_thresh_max'] <= p['lpower_thresh_min']:
                    raise ValueError('Power maximum must exceed minimum')
            else:
                b = p.get('box_size_pix')
                if type(b) is not int or not 64 <= b <= 720 or b % 8:
                    raise ValueError('Box must be a multiple of 8 in 64..720')
            return d
        model = 'openai/gpt-6.1-sol' if stage == 'inspect_picks_threshold' else 'openai/gpt-6-luna'
        return self.model(slug, messages, model, validate)['parameters']

    def classes(self, slug, job):
        from vision_inputs import build_class_average_visual_context
        from cryosift_adapter import evaluate_2d_classes_with_cryosift
        import numpy as np
        directory = self.root / 'evidence' / slug
        if (directory / 'context.json').exists():
            return json.loads((directory / 'context.json').read_text())
        context = self.service('class images ' + job.uid,
            lambda: build_class_average_visual_context(self.project.uid, job.uid, max_classes=0))
        particles = self.service('class populations ' + job.uid, lambda: job.load_output('particles'))
        counts = Counter(map(int, np.asarray(particles['alignments2D/class'])))
        for row in context['class_statistics']:
            row['particle_count'] = counts.get(row['class_id'], 0)
            if isinstance(row.get('resolution_A'), (float, int)) and not math.isfinite(row['resolution_A']):
                row['resolution_A'] = None
        context['cryosift'] = self.service('CryoSift ' + job.uid,
            lambda: evaluate_2d_classes_with_cryosift(self.project.uid, job.uid,
                threshold=getattr(self, 'cryosift_threshold', 3.0), timeout_seconds=600))
        context['score_summary'] = stats(context['cryosift'], counts)
        self.write(directory / 'context.json', context)
        return context

    def choose(self, slug, context, previous=None):
        from upstream_evidence import metadata
        from cryosift_adapter import CRYOSIFT_SCORE_GUIDANCE
        content = [dict(type='text', text=(
            'Assess pilot Class 2D images and select a relatively better subset for ONE focused '
            'reclassification, or assess the reclassified subset for possible later promotion. '
            'Return JSON: {"quality":"usable|uncertain|poor", "selected_class_ids":[integer IDs], '
            '"reason":"auditable image and score evidence"}. Usable means multiple credible, coherent, '
            'complete and centered molecular views, not merely compact spots, radial texture, or '
            'an attractive reported 2D resolution. Do not assume circular/compact averages are better '
            'than coherent elongated or multilobed views. Preserve credible view diversity; no fixed '
            'retention quota. For uncertain/poor initial classes, selecting a defensible relative '
            'subset is allowed as a TEST, not validation. Return [] if no credible subset exists. '
            'In the initial stage, prefer a bounded diagnostic reclassification of a relatively '
            'better nonempty subset when any repeatable signal is visible, even if CryoSift '
            'recommends no class. Definitive particle identity is not required for this diagnostic test. '
            'There is no need to stop or request human input; the controller can expand the sample. '
            'Compare old/new images when both supplied; IDs belong only to their source Job. '
            'Select IDs ONLY from the CURRENT context. Success of this pilot is not a guarantee '
            'of high-resolution reconstruction. ' + CRYOSIFT_SCORE_GUIDANCE))]
        for label, c in [('PREVIOUS', previous), ('CURRENT', context)]:
            if c:
                content += [dict(type='text', text=label + ' ' + json.dumps(metadata(c))),
                            dict(type='image_url', image_url=dict(url=c['contact_sheet']['data_url']))]
        return self.model(slug, [dict(role='system', content='You evaluate cryo-EM pilot evidence; return the specified JSON only.'),
            dict(role='user', content=content)], 'openai/gpt-6.1-sol',
            lambda d: selection(d, set(context['class_ids'])))

    def run(self):
        from vision_inputs import build_micrograph_preview_visual_context, build_pick_inspection_visual_context, build_extract_box_visual_context
        from webapp.usage import task_usage
        if self.state.get('status') in {'pilot_usable_for_promotion', 'pilot_exhausted_no_validated_route'}:
            return
        for scale in SCALES:
            trial = next((r for r in self.state['trials'] if r['micrographs'] == scale), None)
            if trial is None:
                trial = dict(micrographs=scale)
                self.state['trials'].append(trial)
                self.persist()
            if trial.get('finished'):
                continue
            prefix = f'n{scale}'
            if 'picker_parameters' not in self.state:
                preview = self.service('initial micrographs', lambda: build_micrograph_preview_visual_context(
                    self.project.uid, SHARED[-1], max_micrographs=3))
                p = self.stage('initial_picker', 'blob_picker_diameter', preview)
                self.state['picker_parameters'] = {k: p[k] for k in ('diameter', 'diameter_max')}
                self.persist()
            picker = self.submit(prefix+'_blob', 'blob_picker_gpu', {'micrographs': (SHARED[-1], 'exposures')},
                dict(self.state['picker_parameters'], num_process=scale))
            if 'inspect_parameters' not in self.state:
                visual = self.service('picker overlays', lambda: build_pick_inspection_visual_context(
                    self.project.uid, picker.uid, max_micrographs=3))
                p = self.stage('initial_inspect', 'inspect_picks_threshold', visual)
                self.state['inspect_parameters'] = {k: p[k] for k in ('ncc_score_thresh', 'lpower_thresh_min', 'lpower_thresh_max')}
                self.persist()
            inspect = self.submit(prefix+'_inspect', 'inspect_picks_v2',
                dict(micrographs=(picker.uid, 'micrographs'), particles=(picker.uid, 'particles')),
                dict(self.state['inspect_parameters'], calibrate_ncc=False, calibrate_pow=False, do_auto_cluster=False))
            if inspect.outputs['particles'].num_items == 0:
                trial.update(finished=True, outcome='empty_inspect')
                self.persist()
                continue
            if 'extract_parameters' not in self.state:
                visual = self.service('extraction evidence', lambda: build_extract_box_visual_context(
                    self.project.uid, picker.uid, particle_job_uid=inspect.uid, max_examples=3))
                p = self.stage('initial_extract', 'extract_box_size', visual)
                self.state['extract_parameters'] = dict(box_size_pix=p['box_size_pix'])
                self.persist()
            extracted = self.submit(prefix+'_extract', 'extract_micrographs_multi',
                dict(micrographs=(inspect.uid, 'micrographs'), particles=(inspect.uid, 'particles')),
                dict(self.state['extract_parameters'], num_extract=scale, compute_num_gpus=1))
            number = extracted.outputs['particles'].num_items
            trial.update(extracted_particles=number, picker=picker.uid, inspect=inspect.uid, extract=extracted.uid)
            if number < 100:
                trial.update(finished=True, outcome='too_few_extracted_particles')
                self.persist()
                continue
            baseline = self.submit(prefix+'_class', 'class_2D_new', {'particles': (extracted.uid, 'particles')},
                dict(class2D_K=class_count(number), random_seed=26026, compute_num_gpus=1))
            before = self.classes(prefix+'_before', baseline)
            first = self.choose(prefix+'_choose', before)
            trial.update(before_job=baseline.uid, before_scores=before['score_summary'], first_selection=first)
            if not first['selected_class_ids']:
                trial.update(finished=True, outcome='no_credible_subset_expand')
                self.persist()
                continue
            selected = self.submit(prefix+'_select', 'select_2D',
                dict(particles=(baseline.uid, 'particles'), templates=(baseline.uid, 'class_averages')),
                dict(selected_templates=','.join(map(str, first['selected_class_ids']))))
            chosen = selected.outputs['particles_selected'].num_items
            trial.update(selected_particles=chosen, select=selected.uid)
            if chosen < 100:
                trial.update(finished=True, outcome='too_few_selected_particles_expand')
                self.persist()
                continue
            reclass = self.submit(prefix+'_reclass', 'class_2D_new', {'particles': (selected.uid, 'particles_selected')},
                dict(class2D_K=class_count(chosen), random_seed=26027, compute_num_gpus=1))
            after = self.classes(prefix+'_after', reclass)
            last = self.choose(prefix+'_assess', after, previous=before)
            counts = {r['class_id']: r['particle_count'] for r in after['class_statistics']}
            credible = sum(counts[i] for i in last['selected_class_ids'])
            trial.update(after_job=reclass.uid, after_scores=after['score_summary'], assessment=last,
                credible_particles=credible, outcome=next_step(last['quality'], credible, scale), finished=True)
            self.state['usage'] = task_usage(self.root)
            self.persist()
            self.report()
            if trial['outcome'] == 'pilot_usable_for_promotion':
                self.state['status'] = trial['outcome']
                break
        else:
            self.state['status'] = 'pilot_exhausted_no_validated_route'
        self.state['usage'] = task_usage(self.root)
        self.state['note'] = 'Small-pilot experiment only. No full-data picking or 3D reconstruction submitted.'
        self.persist()
        self.report()
        self.write(self.root / 'results.json', self.state)
        print(json.dumps(dict(status=self.state['status'], trials=len(self.state['trials']), usage=self.state['usage'])), flush=True)
