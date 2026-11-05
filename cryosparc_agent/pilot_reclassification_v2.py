"""Parameter-adaptive pilot experiment with a deterministic promotion policy."""
import json
import math
from dataclasses import asdict
from pathlib import Path

from pilot_reclassification_experiment import PilotExperiment, SCALES, SHARED, class_count
from pilot_objective_gate import GatePolicy, VERSION, evaluate_gate, load_class_images


def next_step(gate, scale):
    if gate.get('passed') is True and gate.get('available') is True:
        return 'pilot_usable_for_promotion'
    return 'expand' if scale < SCALES[-1] else 'pilot_exhausted_no_validated_route'


def selected_ids(response, allowed):
    ids = response.get('selected_class_ids')
    if not isinstance(ids, list) or any(type(i) is not int or i not in allowed for i in ids):
        raise ValueError('Select only integer class IDs from the current source job')
    if len(ids) != len(set(ids)) or not isinstance(response.get('reason'), str) or not response['reason'].strip():
        raise ValueError('Unique IDs and a nonempty rationale required')
    # Ignore any unsolicited subjective quality vote; it has no gate authority.
    return dict(selected_class_ids=ids, reason=response['reason'])


def diagnostic_fallback(context):
    """A bounded relative-quality diagnostic, never permission to promote."""
    populations = {r['class_id']: r.get('particle_count', 0) for r in context['class_statistics']}
    sift = context.get('cryosift') or {}
    if (not sift.get('success') or sift.get('job_uid') != context['source']['job_uid']
            or sift.get('project_uid') != context['source']['project_uid']):
        return []
    ranked = sorted((r['score'], r['class_id']) for r in sift.get('class_scores', [])
                    if type(r.get('score')) in (int, float) and math.isfinite(r['score'])
                    and r['class_id'] in context['class_ids'] and populations.get(r['class_id'], 0) > 0)
    return list(dict.fromkeys(i for _, i in ranked))[:3]


def validate_classification(response, particles, physical_box_A):
    params = response.get('parameters')
    allowed = {'class2D_K', 'class2D_num_full_iter', 'class2D_window_inner_A', 'class2D_window_outer_A'}
    if not isinstance(params, dict) or set(params) - allowed:
        raise ValueError('Use only the advertised classification parameters')
    k = params.get('class2D_K')
    if type(k) is not int or not 2 <= k <= min(100, particles):
        raise ValueError('class2D_K must be an integer in 2..min(100, particle count)')
    if 'class2D_num_full_iter' in params and (type(params['class2D_num_full_iter']) is not int or not 1 <= params['class2D_num_full_iter'] <= 5):
        raise ValueError('Full iterations must be 1..5')
    for name in ('class2D_window_inner_A', 'class2D_window_outer_A'):
        if name in params and (type(params[name]) not in (int, float) or not math.isfinite(params[name]) or not 0 < params[name] <= physical_box_A):
            raise ValueError('Classification window must fit the physical extraction box')
    if all(k in params for k in ('class2D_window_inner_A', 'class2D_window_outer_A')) and params['class2D_window_inner_A'] >= params['class2D_window_outer_A']:
        raise ValueError('Inner window must be smaller than outer window')
    return response


class ObjectivePilotExperiment(PilotExperiment):
    def __init__(self, root):
        manifest = json.loads((Path(root)/'manifest.json').read_text())
        if manifest.get('protocol_version') != VERSION:
            raise ValueError('V2 requires a new V2 protocol/root; do not mutate completed V1 experiments')
        super().__init__(root)
        self.policy = GatePolicy(**self.manifest['gate_policy'])
        self.cryosift_threshold = self.policy.cryosift_threshold
        if self.state.get('protocol_version', VERSION) != VERSION:
            raise ValueError('Incompatible saved experiment state')
        self.state['protocol_version'] = VERSION
        if self.state.get('gate_policy', asdict(self.policy)) != asdict(self.policy):
            raise ValueError('Gate policy changed after checkpoint; start a new experiment')
        self.state['gate_policy'] = asdict(self.policy)

    def feedback(self):
        return [dict(micrographs=t['micrographs'], parameters=t.get('parameters'),
                     extracted_particles=t.get('extracted_particles'), selected_particles=t.get('selected_particles'),
                     inspect_accepted=t.get('inspect_accepted'), raw_picks=t.get('raw_picks'),
                     before_scores=t.get('before_scores'), after_scores=t.get('after_scores'),
                     objective_gate={k:v for k,v in (t.get('objective_gate') or {}).items()
                                     if k not in {'pairwise_shape_ncc','similarity_class_ids','policy'}},
                     outcome=t.get('outcome'))
                for t in self.state['trials'] if t.get('finished')][-3:]

    def model(self, slug, messages, model, validate):
        suffix = slug.rsplit('_', 1)[-1]
        optional = {'picker': {'use_circle', 'use_ellipse', 'use_ring', 'min_distance'},
                    'inspect': set(), 'extract': set()}
        if suffix in optional:
            extras = (
                ' For Blob Picker the parameters object may additionally include use_circle, '
                'use_ellipse, use_ring (exactly one true if supplied together) and min_distance '
                '(positive dimensionless diameter multiplier, not Angstroms).'
            ) if suffix == 'picker' else ''
            messages = [*messages, dict(role='user', content=(
                'V2 experiment: parameters are NOT frozen. Reconsider this step using current '
                'evidence and previous-pilot feedback. You may retain a previous setting with '
                'an evidence-based rationale; you are not required to change values arbitrarily. '
                'Use prior missed picks, low yields and objective failures to revise diameter, '
                'thresholds or box size as appropriate. A failed previous Class 2D or gate does NOT '
                'imply that filtering should become stricter: insufficient sample size, too few '
                'particles per class, missed true particles or another upstream setting may be '
                'responsible. Do not repeatedly raise NCC minima or lower Power maxima merely '
                'because previous trials failed. Treat stricter and more permissive thresholds '
                'as competing hypotheses requiring a matched comparison. Do not issue a promotion/usable vote; '
                'promotion is a deterministic CryoSift + 2D-shape-diversity + count gate.' + extras))]
            original_validate = validate
            def validate(d):
                d = original_validate(d)
                p = d['parameters']
                allowed = {'picker': {'diameter','diameter_max','use_circle','use_ellipse','use_ring','min_distance'},
                           'inspect': {'ncc_score_thresh','lpower_thresh_min','lpower_thresh_max','calibrate_ncc','calibrate_pow','do_auto_cluster'},
                           'extract': {'box_size_pix'}}[suffix]
                if set(p)-allowed:
                    raise ValueError('Use only the advertised step parameters; resource limits and sample size are controlled separately')
                if suffix == 'inspect' and p.get('do_auto_cluster', False) is not False:
                    raise ValueError('This experiment uses explicit raw thresholds, not auto clustering')
                if suffix == 'picker':
                    shapes = ('use_circle', 'use_ellipse', 'use_ring')
                    if any(k in p for k in shapes):
                        if not all(type(p.get(k)) is bool for k in shapes) or sum(p[k] for k in shapes) != 1:
                            raise ValueError('Explicitly supply all three shape booleans with exactly one true')
                    if 'min_distance' in p and (type(p['min_distance']) not in (int, float) or not math.isfinite(p['min_distance']) or not 0 < p['min_distance'] <= 2):
                        raise ValueError('min_distance must be a positive dimensionless multiplier <=2')
                return d
        return super().model(slug, messages, model, validate)

    def stage(self, slug, stage, visual):
        visual = {**visual, 'previous_pilot_feedback': self.feedback(), 'promotion_policy': self.manifest['gate_policy'],
                  'current_trial_parameters': self.state['trials'][-1].get('parameters', {}) if self.state['trials'] else {}}
        # Also show the latest prior class sheet so parameter changes can be
        # grounded in the actual failure morphology, not only scalar scores.
        previous = next((t for t in reversed(self.state['trials']) if t.get('finished') and t.get('after_job')), None)
        if previous:
            path = self.root/'evidence'/f"n{previous['micrographs']}_after"/'context.json'
            if path.exists():
                c = json.loads(path.read_text())
                visual['previous_class_source'] = c['source']
                visual['evidence_images'] = [*(visual.get('evidence_images') or []), dict(
                    data_url=c['contact_sheet']['data_url'], role='previous_pilot_class_averages',
                    description='Previous pilot Class 2D; not the current micrograph. Source '+c['source']['job_uid'])]
        return super().stage(slug, stage, visual)

    def classify_parameters(self, slug, particles, physical_box_A, previous=None):
        from upstream_evidence import metadata
        content = [dict(type='text', text=json.dumps(dict(
            instruction='Choose this Class 2D step parameters. Reconsider K, full iterations and optional physical windows using current counts and prior outcomes. '
                        'Do not infer true angular coverage from a 2D shape group count. Return JSON with parameters and an auditable reason.',
            schema=dict(parameters=dict(class2D_K='required integer', class2D_num_full_iter='optional integer 1..5',
                class2D_window_inner_A='optional positive Angstrom', class2D_window_outer_A='optional positive Angstrom'), reason='string'),
            particles=particles, physical_box_A=physical_box_A, suggested_K=class_count(particles),
            previous_trials=self.feedback(), source_classes=metadata(previous) if previous else None)))]
        if previous:
            content.append(dict(type='image_url', image_url=dict(url=previous['contact_sheet']['data_url'])))
        return self.model(slug, [dict(role='system', content='Choose bounded cryo-EM classification parameters, not a promotion verdict.'),
            dict(role='user', content=content)], 'openai/gpt-6-luna',
            lambda d: validate_classification(d, particles, physical_box_A))['parameters']

    def choose(self, slug, context, previous=None):
        from upstream_evidence import metadata
        from cryosift_adapter import CRYOSIFT_SCORE_GUIDANCE
        messages = [dict(role='system', content='Choose a bounded diagnostic particle subset. Do not decide whether to promote to full data.'),
            dict(role='user', content=[dict(type='text', text=(
                'Return JSON {"selected_class_ids":[integer IDs],"reason":"evidence"}. '
                'Select relatively better, repeatable source classes for one reclassification. '
                'An uncertain nonempty selection is allowed for diagnosis, not proof of quality. '
                'Preserve plausible different shapes rather than preferring only compact views. '
                'No retention quota; class IDs belong only to this source job. ' + CRYOSIFT_SCORE_GUIDANCE
                + '\nCURRENT ' + json.dumps(metadata(context)))),
                dict(type='image_url', image_url=dict(url=context['contact_sheet']['data_url']))])]
        return self.model(slug, messages, 'openai/gpt-6.1-sol',
                          lambda d: selected_ids(d, set(context['class_ids'])))

    def report(self):
        import csv
        rows = []
        for t in self.state['trials']:
            g = t.get('objective_gate') or {}
            rows.append(dict(micrographs=t['micrographs'], before_job=t.get('before_job'), after_job=t.get('after_job'),
                extracted=t.get('extracted_particles'), selected=t.get('selected_particles'),
                qualified_particles=g.get('qualified_particles'), supported_2d_shape_groups=g.get('supported_group_count'),
                objective_passed=g.get('passed'), failed_conditions=';'.join(g.get('failed_conditions', [])), outcome=t.get('outcome')))
        with (self.root/'metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['micrographs'])
            writer.writeheader()
            writer.writerows(rows)
        lines = ['# Parameter-Adaptive Pilot: Objective Promotion Gate', '',
            f"Workspace: {self.project.uid}/{self.workspace.uid}. Status: {self.state['status']}.", '',
            'Every pilot re-decides Picker, Inspect, Extract and both Class 2D parameters. Model subset selection does not control promotion.', '',
            'Policy: '+json.dumps(self.manifest['gate_policy']), '',
            '| Micrographs | After Class 2D | Qualified particles | Supported 2D shape groups | Passed | Outcome |',
            '| ---: | --- | ---: | ---: | --- | --- |']
        for r in rows:
            lines.append('| '+' | '.join(str(r[k]) for k in ('micrographs','after_job','qualified_particles','supported_2d_shape_groups','objective_passed','outcome'))+' |')
        lines += ['', 'The diversity metric is a reference-free 2D shape proxy, NOT measured 3D angular coverage.',
                  'Thresholds are configurable experimental defaults, not calibrated scientific guarantees. No full-data or 3D Job is submitted automatically.']
        (self.root/'REPORT.md').write_text('\n'.join(lines)+'\n')

    def run(self):
        from vision_inputs import build_micrograph_preview_visual_context, build_pick_inspection_visual_context, build_extract_box_visual_context
        from webapp.usage import task_usage
        if self.state.get('status') in {'pilot_usable_for_promotion','pilot_exhausted_no_validated_route'}:
            return
        for scale in getattr(self, 'scales', SCALES):
            trial = next((t for t in self.state['trials'] if t['micrographs'] == scale), None)
            if trial is None:
                trial = dict(micrographs=scale, parameters={})
                self.state['trials'].append(trial)
                self.persist()
            if trial.get('finished'):
                continue
            prefix = f'n{scale}'
            preview = self.service('micrograph evidence', lambda: build_micrograph_preview_visual_context(self.project.uid, SHARED[-1], max_micrographs=3))
            p = self.stage(prefix+'_picker', 'blob_picker_diameter', preview)
            trial['parameters']['picker'] = p
            self.persist()
            picker = self.submit(prefix+'_blob', 'blob_picker_gpu', {'micrographs':(SHARED[-1],'exposures')}, dict(p,num_process=scale))
            trial.update(picker=picker.uid, raw_picks=picker.outputs['particles'].num_items)
            if trial['raw_picks'] == 0:
                trial.update(finished=True,outcome='empty_picker_expand')
                self.persist()
                continue
            visual = self.service('picker evidence', lambda: build_pick_inspection_visual_context(self.project.uid,picker.uid,max_micrographs=3))
            p = self.stage(prefix+'_inspect','inspect_picks_threshold',visual)
            native = {k:p[k] for k in ('ncc_score_thresh','lpower_thresh_min','lpower_thresh_max')}
            trial['parameters']['inspect'] = native
            inspect = self.submit(prefix+'_inspect','inspect_picks_v2',dict(micrographs=(picker.uid,'micrographs'),particles=(picker.uid,'particles')),
                dict(native,calibrate_ncc=False,calibrate_pow=False,do_auto_cluster=False))
            trial.update(inspect=inspect.uid, inspect_accepted=inspect.outputs['particles'].num_items)
            if trial['inspect_accepted'] == 0:
                trial.update(finished=True,outcome='empty_inspect_expand')
                self.persist()
                continue
            visual = self.service('extract evidence',lambda:build_extract_box_visual_context(self.project.uid,picker.uid,particle_job_uid=inspect.uid,max_examples=3))
            p = self.stage(prefix+'_extract','extract_box_size',visual)
            trial['parameters']['extract'] = p
            extracted = self.submit(prefix+'_extract','extract_micrographs_multi',dict(micrographs=(inspect.uid,'micrographs'),particles=(inspect.uid,'particles')),
                dict(box_size_pix=p['box_size_pix'],num_extract=scale,compute_num_gpus=1))
            number = extracted.outputs['particles'].num_items
            trial.update(extract=extracted.uid,extracted_particles=number)
            if number < 100:
                trial.update(finished=True,outcome='too_few_particles_expand')
                self.persist()
                continue
            physical = p['box_size_pix']*.856
            params = self.classify_parameters(prefix+'_class_parameters',number,physical)
            trial['parameters']['classify'] = params
            baseline = self.submit(prefix+'_class','class_2D_new',{'particles':(extracted.uid,'particles')},dict(params,random_seed=26026,compute_num_gpus=1))
            before = self.classes(prefix+'_before',baseline)
            first = self.choose(prefix+'_choose',before)
            ids = first['selected_class_ids']
            if not ids:
                ids = diagnostic_fallback(before)
                trial['selection_fallback'] = dict(class_ids=ids,reason='Lowest finite CryoSift nonempty classes for diagnosis only; cannot authorize promotion')
            trial.update(before_job=baseline.uid,before_scores=before['score_summary'],first_selection=first)
            if not ids:
                trial.update(finished=True,outcome='no_diagnostic_subset_expand')
                self.persist()
                continue
            selected = self.submit(prefix+'_select','select_2D',dict(particles=(baseline.uid,'particles'),templates=(baseline.uid,'class_averages')),
                dict(selected_templates=','.join(map(str,ids))))
            chosen = selected.outputs['particles_selected'].num_items
            trial.update(select=selected.uid,selected_particles=chosen)
            if chosen < 100:
                trial.update(finished=True,outcome='too_few_selected_expand')
                self.persist()
                continue
            params = self.classify_parameters(prefix+'_reclass_parameters',chosen,physical,previous=before)
            trial['parameters']['reclassify'] = params
            reclass = self.submit(prefix+'_reclass','class_2D_new',{'particles':(selected.uid,'particles_selected')},dict(params,random_seed=26027,compute_num_gpus=1))
            after = self.classes(prefix+'_after',reclass)
            images = self.service('objective class images',lambda:load_class_images(self.project,reclass.uid))
            gate = evaluate_gate(after,images,self.policy)
            self.write(self.root/'evidence'/prefix/'objective_gate.json',gate)
            trial.update(after_job=reclass.uid,after_scores=after['score_summary'],objective_gate=gate,
                         credible_particles=gate.get('qualified_particles',0),outcome=next_step(gate,scale),finished=True)
            self.state['usage'] = task_usage(self.root)
            self.persist()
            self.report()
            if trial['outcome'] == 'pilot_usable_for_promotion':
                self.state['status'] = 'pilot_usable_for_promotion'
                self.state['promotion_candidate'] = dict(source_class_job=reclass.uid,
                    class_ids=gate['promotion_class_ids'], qualified_particles=gate['qualified_particles'],
                    parameters=trial['parameters'], objective_gate_file=str(self.root/'evidence'/prefix/'objective_gate.json'))
                break
        else:
            self.state['status'] = 'pilot_exhausted_no_validated_route'
        self.state['usage'] = task_usage(self.root)
        self.state['note'] = 'Objective pilot gate only. No full-data or 3D submission; no subjective model usability veto.'
        self.persist()
        self.report()
        self.write(self.root/'results.json',self.state)
