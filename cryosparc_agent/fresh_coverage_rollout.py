"""Isolated, fresh post-CTF validation of the complete coverage-aware order."""
import copy
import json
import math
import time
from pathlib import Path

import numpy as np

from coverage_rollout import CoverageRollout, POLICY, class_pool, heldout_indices, union_from_source
from pilot_inspect_comparison import validate_options
from pilot_objective_gate import GatePolicy
from pilot_reclassification_experiment import PilotExperiment


def validate_boxes(response):
    boxes = response.get('box_sizes_pix')
    if (not isinstance(boxes, list) or len(boxes) != 2 or
            any(type(b) is not int or not 64 <= b <= 720 or b % 8 for b in boxes) or
            boxes[1] - boxes[0] < 32 or not response.get('reason')):
        raise ValueError('Two ascending native boxes, multiples of 8 in 64..720, gap >=32, with reason required')
    return response


def diameter_contract(messages):
    """The controlled grid needs a finite range, unlike the general backend."""
    messages = copy.deepcopy(messages)
    requirement = ('CONTROLLED PILOT CONTRACT: both parameters.diameter AND parameters.diameter_max '
        'are REQUIRED positive finite numbers in Angstroms, with maximum >= minimum. '
        'They bound an exploratory small-sample experiment, not a measured distribution. '
        'When uncertain, provide a plausible bounded trial range, mark diameter_confidence=uncertain '
        'and explain the uncertainty. Do not omit the maximum or request human input. '
        'This explicit pilot requirement overrides generic optional-maximum wording above.')
    user = messages[-1]
    if user['role'] != 'user':
        raise ValueError('Stage contract must end in user request')
    if isinstance(user['content'], list):
        user['content'].append(dict(type='text', text=requirement))
    else:
        user['content'] += '\n' + requirement
    return messages


class FreshCoverageRollout(CoverageRollout):
    def model(self, slug, messages, model, validate):
        # Stage-specific contracts must remain the final user instruction.
        # The parent appends class-recovery guidance even to diameter requests.
        if slug == 'pilot_picker_parameters':
            messages = diameter_contract(messages)
        if getattr(self, 'api_base', '') == 'http://15.204.107.114:3009/v1':
            model = model.removeprefix('openai/')
        print(json.dumps(dict(event='model_start', stage=slug, model=model)), flush=True)
        started = time.monotonic()
        try:
            result = PilotExperiment.model(self, slug, messages, model, validate)
        except Exception as exc:
            print(json.dumps(dict(event='model_failed', stage=slug,
                elapsed_seconds=round(time.monotonic()-started, 2), error_type=type(exc).__name__)), flush=True)
            raise
        print(json.dumps(dict(event='model_validated', stage=slug,
            elapsed_seconds=round(time.monotonic()-started, 2))), flush=True)
        return result

    def __init__(self, root):
        PilotExperiment.__init__(self, root)
        if (self.manifest.get('protocol') != 'fresh-coverage-v1' or
                self.manifest.get('shared_jobs') != ['J1215', 'J1250', 'J1442']):
            raise ValueError('Fresh experiment may share only declared Import/Motion/CTF')
        self.allowed_samples = (50, 200, self.manifest['full_micrographs'])
        self.policy = GatePolicy()
        self.cryosift_threshold = 3.
        self.api_base = self.manifest.get('api_override', {}).get('base_url', 'https://api.ofox.ai/v1')

    def submit(self, slug, kind, sources, params, wait=True):
        params = copy.deepcopy(params)
        # Parent rollout uses fixed pilot values. Replace them only after this
        # run has measured and chosen its own recipe; never alter the grid.
        if slug.startswith(('confirm_', 'ramp', 'full_')):
            recipe = self.state['pilot_recipe']
            if kind == 'blob_picker_gpu':
                params.update(recipe['picker_parameters'])
            if kind == 'extract_micrographs_multi':
                params['box_size_pix'] = recipe['box_size_pix']
        return super().submit(slug, kind, sources, params, wait)

    def visual_content(self, instruction, visual):
        from upstream_evidence import metadata
        content = [dict(type='text', text=instruction + '\n' + json.dumps(metadata(visual)))]
        images = [visual.get('contact_sheet') or {}, visual.get('score_scatter') or {},
                  *visual.get('evidence_images', [])]
        seen = set()
        for image in images:
            url = image.get('data_url')
            if url and url not in seen:
                seen.add(url)
                content.append(dict(type='image_url', image_url=dict(url=url)))
        return content

    def grid(self):
        from vision_inputs import (build_micrograph_preview_visual_context,
                                   build_pick_inspection_visual_context, build_extract_box_visual_context)
        if self.state.get('pilot_recipe'):
            return self.state['pilot_recipe']
        ctf_uid = self.manifest['shared_jobs'][-1]
        source = self.project.find_job(ctf_uid).load_output('exposures')
        pilot = self.external('pilot50', 'exposure', 'exposures',
                              dict(source=(ctf_uid, 'exposures')), source.take(np.arange(50)))
        grid = self.state.setdefault('pilot_grid', {})
        if 'picker_parameters' not in grid:
            preview = self.service('fresh pilot preview', lambda: build_micrograph_preview_visual_context(
                self.project.uid, pilot.uid, output_name='exposures', max_micrographs=3))
            grid['picker_parameters'] = self.stage('pilot_picker_parameters', 'blob_picker_diameter', preview)
            self.persist()
        picker = self.submit('pilot_picker', 'blob_picker_gpu', dict(micrographs=(pilot.uid, 'exposures')),
                             dict(grid['picker_parameters'], num_process=50))
        if not picker.outputs['particles'].num_items:
            raise RuntimeError('Fresh pilot Blob has no particles; retain failure, do not invent promotion')
        if 'options' not in grid:
            visual = self.service('fresh threshold evidence', lambda: build_pick_inspection_visual_context(
                self.project.uid, picker.uid, max_micrographs=3))
            content = self.visual_content(
                'Design three DISTINCT NESTED raw NCC/Power windows on these same 50 micrographs: '
                'permissive, balanced, strict. Failure can reflect too few samples, not a need for stricter '
                'filtering. Preserve plausible weak views and most luminous signal, exclude obvious junk. '
                'Return JSON {"options":[{"name":"permissive","parameters":{"ncc_score_thresh":number,'
                '"lpower_thresh_min":number,"lpower_thresh_max":number},"reason":"evidence"},'
                '{same fields, name balanced},{same fields, name strict}]}. '
                'NCC and lower Power increase, upper Power decreases across the three windows.', visual)
            grid['options'] = self.model('pilot_threshold_options', [dict(role='system', content=POLICY),
                dict(role='user', content=content)], 'openai/gpt-6.1-sol', validate_options)['options']
            self.persist()
        inputs = dict(micrographs=(picker.uid, 'micrographs'), particles=(picker.uid, 'particles'))
        inspected = {}
        for wait in (False, True):
            for option in grid['options']:
                name = option['name']
                inspected[name] = self.submit('grid_' + name + '_inspect', 'inspect_picks_v2', inputs,
                    dict(option['parameters'], calibrate_ncc=False, calibrate_pow=False, do_auto_cluster=False), wait)
        nonempty = [j for j in inspected.values() if j.outputs['particles'].num_items]
        if not nonempty:
            raise RuntimeError('All pilot windows empty; genuine physical failure, no fabricated class metrics')
        if 'boxes' not in grid:
            evidence = self.service('fresh box evidence', lambda: build_extract_box_visual_context(
                self.project.uid, picker.uid, particle_job_uid=nonempty[0].uid, max_examples=3))
            content = self.visual_content(
                'Choose TWO ascending native Extract box sizes to compare. Use the scale bar and calibrated '
                'native pixel size 0.856 A/px; displayed pixels are NOT native pixels. '
                'Around 1.5-2x the maximum whole-particle diameter is a starting preference, not a guarantee. '
                'One balanced and one more generous box; avoid both clipping and multiple particles per box. '
                'No Fourier cropping. Multiples of 8, 64..720, separated by at least 32 pixels. '
                'Return {"box_sizes_pix":[integer,integer],"reason":"evidence and uncertainty"}.', evidence)
            grid['boxes'] = self.model('pilot_box_options', [dict(role='system', content=POLICY),
                dict(role='user', content=content)], 'openai/gpt-6.1-sol', validate_boxes)['box_sizes_pix']
            self.persist()
        branches = [(o, b) for o in grid['options'] for b in grid['boxes']
                    if inspected[o['name']].outputs['particles'].num_items]
        extracted = {}
        for wait in (False, True):
            for option, box in branches:
                name = option['name'] + '_' + str(box)
                job = inspected[option['name']]
                extracted[name] = self.submit('grid_' + name + '_extract', 'extract_micrographs_multi',
                    dict(micrographs=(job.uid, 'micrographs'), particles=(job.uid, 'particles')),
                    dict(box_size_pix=box, num_extract=50, compute_num_gpus=1), wait)
        rows = grid.setdefault('rows', {})
        for wait in (False, True):
            for option, box in branches:
                name = option['name'] + '_' + str(box)
                job = extracted[name]
                if job.outputs['particles'].num_items < 100:
                    rows[name] = dict(executable=False, reason='Fewer than 100 extracted particles')
                    continue
                classified = self.submit('grid_' + name + '_class', 'class_2D_new',
                    dict(particles=(job.uid, 'particles')),
                    dict(class2D_K=20, class2D_num_full_iter=4, random_seed=26026, compute_num_gpus=1), wait)
                if wait:
                    slug = 'grid_' + name + '_classes'
                    context, gate = self.readout(slug, classified)
                    rows[name] = dict(executable=True, classes=classified.uid, evidence_slug=slug,
                        particles=job.outputs['particles'].num_items, box_size_pix=box,
                        thresholds=option['parameters'], picker_parameters=grid['picker_parameters'],
                        picker=picker.uid, inspect=inspected[option['name']].uid, extract=job.uid,
                        score=context['score_summary'], gate=gate)
                    self.persist()
        viable = {key: value for key, value in rows.items() if value.get('executable')}
        if not viable:
            raise RuntimeError('No physically executable grid branch')
        content = [dict(type='text', text=POLICY + ' Choose ONE route from this matched 3-window x '
            '2-box comparison. Consider intact particles, diversity, yield, lower CryoSift scores and '
            'qualified counts, not only mean score. Relative best does not imply an absolute gate pass. '
            'Return {"route":"one advertised key","reason":"evidence"}.\n' + json.dumps(viable))]
        for name, row in viable.items():
            context = json.loads((self.root / 'evidence' / row['evidence_slug'] / 'context.json').read_text())
            content += [dict(type='text', text=name + ' ' + row['classes']),
                dict(type='image_url', image_url=dict(url=context['contact_sheet']['data_url']))]
        def validate(r):
            if r.get('route') not in viable or not r.get('reason'):
                raise ValueError('Choose one executable grid key with evidence')
            return r
        choice = self.model('pilot_grid_choice', [dict(role='system', content=POLICY),
            dict(role='user', content=content)], 'openai/gpt-6.1-sol', validate)
        self.state['pilot_recipe'] = dict(viable[choice['route']], route=choice['route'], reason=choice['reason'])
        self.persist()
        return self.state['pilot_recipe']

    def bank(self):
        if self.state.get('bank_job'):
            return self.project.find_job(self.state['bank_job'])
        recipe = self.grid()
        source = self.project.find_job(recipe['classes'])
        context = json.loads((self.root / 'evidence' / recipe['evidence_slug'] / 'context.json').read_text())
        ids = self.review('pilot_core_review', context, 'core')
        core = self.select('pilot_core_select', source.uid, ids)
        initial = self.submit('pilot_core_abinit', 'homo_abinit', dict(particles=(core.uid, 'particles_selected')),
                              dict(abinit_K=1))
        homo = self.submit('pilot_core_homo', 'homo_refine_new',
            dict(particles=(core.uid, 'particles_selected'), volume=(initial.uid, 'volume_class_0')),
            dict(refine_symmetry='C1'))
        nu = self.submit('pilot_core_nu', 'nonuniform_refine_new',
            dict(particles=(homo.uid, 'particles'), volume=(homo.uid, 'volume')), dict(refine_symmetry='C1'))
        selections = [core]
        provenance = [dict(source=source.uid, ids=ids, role='core', selection=core.uid)]
        pool = class_pool(context, 'recovery', ids)
        if pool:
            weak = self.select('pilot_weak_select', source.uid, pool)
            n = weak.outputs['particles_selected'].num_items
            if n >= 100:
                recovered = self.submit('pilot_weak_reclass', 'class_2D_new',
                    dict(particles=(weak.uid, 'particles_selected')),
                    dict(class2D_K=min(30, max(5, n // 500)), class2D_num_full_iter=4,
                         random_seed=26027, compute_num_gpus=1))
                c, _ = self.readout('pilot_weak_classes', recovered)
                matched = self.reference('pilot_weak_reference', recovered.uid, nu.uid)
                extra_ids = self.review('pilot_weak_review', c, 'recovered', reference=matched)
                if extra_ids:
                    extra = self.select('pilot_weak_compatible', recovered.uid, extra_ids)
                    selections.append(extra)
                    provenance.append(dict(source=recovered.uid, ids=extra_ids, role='recovered', selection=extra.uid))
        checkpoints = [homo, nu]
        if len(selections) > 1:
            combined = union_from_source(source.load_output('particles'),
                [j.load_output('particles_selected') for j in selections])
            union = self.external('pilot_core_weak_union', 'particle', 'particles',
                                  dict(source=(source.uid, 'particles')), combined)
            expanded = self.submit('pilot_union_nu', 'nonuniform_refine_new',
                dict(particles=(union.uid, 'particles'), volume=(nu.uid, 'volume')),
                dict(refine_symmetry='C1', refine_res_init=30., refine_gs_resplit=True))
            checkpoints.append(expanded)
        datasets = [j.load_output('templates_selected').filter_fields(
            lambda f: f == 'uid' or f.startswith('blob/')) for j in selections]
        data = datasets[0].append(*datasets[1:]) if len(datasets) > 1 else datasets[0]
        bank = self.external('pilot_template_bank', 'template', 'templates',
            {'source_' + str(i): (j.uid, 'templates_selected') for i, j in enumerate(selections)}, data)
        best = min(checkpoints, key=self.resolution)
        self.state.update(bank_job=bank.uid, bank_provenance=provenance, pilot_best_refine=best.uid,
                          pilot_best_resolution_A=self.resolution(best))
        self.persist()
        return bank

    def template_parameters(self):
        if self.state.get('template_parameters'):
            return self.state['template_parameters']
        recipe = self.state['pilot_recipe']
        content = [dict(type='text', text='Estimate whole-particle Template Picker diameter in Angstroms. '
            f'Native extraction box {recipe["box_size_pix"]} pixels at 0.856 A/px; physical field '
            f'{recipe["box_size_pix"] * .856:.4f} A per tile. Displayed tile pixels are NOT native pixels. '
            'Use complete molecular extent, not Blob minimum or a subunit. '
            'Return {"diameter":number,"reason":"evidence and uncertainty"}. '
            'Use native dimensionless min_distance default; do not override it.')]
        for row in self.state['bank_provenance']:
            slug = recipe['evidence_slug'] if row['role'] == 'core' else 'pilot_weak_classes'
            c = json.loads((self.root / 'evidence' / slug / 'context.json').read_text())
            content += [dict(type='text', text=json.dumps(row)),
                dict(type='image_url', image_url=dict(url=c['contact_sheet']['data_url']))]
        def validate(r):
            d = r.get('diameter')
            if type(d) not in (int, float) or not math.isfinite(d) or not 20 <= d <= 600 or not r.get('reason'):
                raise ValueError('Finite diameter 20..600 A and reason required')
            return r
        response = self.model('fresh_template_diameter', [dict(role='system', content=POLICY),
            dict(role='user', content=content)], 'openai/gpt-6.1-sol', validate)
        self.state['template_parameters'] = dict(diameter=response['diameter'])
        self.persist()
        return self.state['template_parameters']

    def run(self):
        if self.state.get('rollout_complete'):
            return self.state['best_refine']
        bank = self.bank()
        ctf_uid = self.manifest['shared_jobs'][-1]
        ctf = self.project.find_job(ctf_uid).load_output('exposures')
        pilot = self.project.find_job(self.state['steps']['pilot50']['job_uid']).load_output('exposures')
        indices = heldout_indices(ctf['uid'], pilot['uid'], 200)
        confirmation = self.external('heldout50', 'exposure', 'exposures',
            dict(source=(ctf_uid, 'exposures')), ctf.take(indices[:50]))
        ramp = self.external('ramp200', 'exposure', 'exposures',
            dict(source=(ctf_uid, 'exposures')), ctf.take(indices))
        self.write(self.root / 'independence.json', dict(pilot_uids=list(map(int, pilot['uid'])),
            confirmation_uids=list(map(int, ctf['uid'][indices[:50]])), ramp_uids=list(map(int, ctf['uid'][indices])),
            confirmation_disjoint_from_pilot=True, confirmation_included_in_ramp=True,
            note='Full data later include pilot images; this is route confirmation, not a held-out outcome benchmark.'))
        routes = self.state.setdefault('confirmation', {})
        for kind in ('blob', 'template'):
            if kind not in routes:
                routes[kind] = self.route('confirm_' + kind, (confirmation.uid, 'exposures'), 50, kind, bank,
                    self.state['pilot_recipe']['thresholds'] if kind == 'blob' else None)
                self.persist()
        if not self.state.get('chosen_route'):
            self.state['chosen_route'] = self.choose_route(routes)
            self.persist()
        chosen = self.state['chosen_route']
        if 'ramp' not in self.state:
            self.state['ramp'] = self.route('ramp', (ramp.uid, 'exposures'), 200, chosen, bank, routes[chosen]['thresholds'])
            self.persist()
        if not self.state['ramp'].get('executable'):
            alternate = 'blob' if chosen == 'template' else 'template'
            if not routes[alternate].get('executable'):
                raise RuntimeError('Neither route can physically continue from ramp')
            if 'ramp_alternate' not in self.state:
                self.state['ramp_alternate'] = self.route('ramp_alternate', (ramp.uid, 'exposures'),
                    200, alternate, bank, routes[alternate]['thresholds'])
                self.persist()
            if not self.state['ramp_alternate'].get('executable'):
                raise RuntimeError('Alternate ramp has no executable particles')
            chosen = alternate
            self.state['chosen_route'] = chosen
            self.state['ramp'] = self.state['ramp_alternate']
            self.persist()
        self.state['promotion_quality_status'] = ('objective_pass_on_confirmation_and_ramp'
            if routes[chosen]['gate'].get('passed') and self.state['ramp']['gate'].get('passed')
            else 'exploratory_relative_best_not_validated')
        self.persist()
        if 'full' not in self.state:
            self.state['full'] = self.route('full', (ctf_uid, 'exposures'), self.manifest['full_micrographs'],
                chosen, bank, self.state['ramp']['thresholds'])
            self.persist()
        if not self.state['full'].get('executable'):
            raise RuntimeError('Full route has no executable particles')
        best = self.reconstruct(self.state['full'])
        self.state.update(rollout_complete=True, status='seeded_for_planner', best_refine=best.uid,
                          best_resolution_A=self.resolution(best))
        self.persist()
        return best.uid
