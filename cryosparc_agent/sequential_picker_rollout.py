"""Opt-in serial visual review, qualified-core reconstruction and promotion."""
import copy
import json
import math
from pathlib import Path

import numpy as np

from coverage_rollout import CoverageRollout, class_pool, heldout_indices, union_from_source
from fresh_coverage_rollout import FreshCoverageRollout
from pilot_objective_gate import GatePolicy, evaluate_gate, load_class_images
from pilot_reclassification_experiment import PilotExperiment, native_thresholds
from extract_scale_context import SCALE_GUIDANCE, scale_memory as calibrated_scale_memory, validate_scale_response


POLICY = (
    'Execute one Inspect window, review actual kept/rejected overlays and raw NCC/Power, '
    'then keep or revise it serially. Execute one native Extract box, inspect actual '
    'particle stacks and matched full-field images, then keep or revise it serially. '
    'No Cartesian grid. CryoSift lower is better. A core requires MORE THAN 2000 '
    'particles inside classes scoring strictly below 3, not total yield or whole-job '
    'average. On failure expand nested micrograph samples with frozen accepted '
    'parameters FIRST. Only after maximum sample failure diagnose whether to change '
    'Inspect thresholds OR the Extract box; change only one factor per round. '
    'failure is not evidence that filtering must become stricter. Once core quality '
    'passes, reconstruct it. If diversity is insufficient, separately reclassify '
    'excluded 3..4.5 classes and screen against the actual core and own lowpassed map. '
    'Full-data promotion requires at least three supported high-quality 2D shape groups '
    'as well as the qualified-particle floor. Shape groups are not measured 3D directions '
    'and own-reference agreement/FSC are not proof of correctness. Never bypass a '
    'failed objective gate with a subjective usable vote. Stop explicitly when bounded '
    'attempts/budget are exhausted, not in a human-input wait.'
)


class ProtocolExhausted(RuntimeError):
    pass


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def inspect_parameters(parameters):
    p = native_thresholds(parameters)
    allowed = {'ncc_score_thresh', 'lpower_thresh_min', 'lpower_thresh_max',
               'calibrate_ncc', 'calibrate_pow', 'do_auto_cluster'}
    if set(p) - allowed:
        raise ValueError('Use only the explicit native NCC/Power threshold fields')
    for key in ('ncc_score_thresh', 'lpower_thresh_min', 'lpower_thresh_max'):
        if not finite_number(p.get(key)):
            raise ValueError('Finite native NCC lower/Power lower/Power upper required')
    if p['lpower_thresh_min'] >= p['lpower_thresh_max'] or p.get('do_auto_cluster') not in (None, False):
        raise ValueError('Ordered Power window and non-interactive explicit thresholds required')
    return {**p, 'calibrate_ncc': False, 'calibrate_pow': False, 'do_auto_cluster': False}


def validate_visual_review(response, stage, current, seen):
    if response.get('action') not in {'keep', 'revise'} or not str(response.get('reason', '')).strip():
        raise ValueError('Return action keep|revise and evidence-based reason')
    if response['action'] == 'revise':
        p = response.get('parameters', {})
        if stage == 'inspect':
            p = inspect_parameters(p)
        else:
            b = p.get('box_size_pix')
            if set(p) != {'box_size_pix'} or type(b) is not int or not 64 <= b <= 720 or b % 8:
                raise ValueError('Native box must be a multiple of 8 in 64..720; no Fourier crop')
        if p == current or p in seen:
            raise ValueError('Revision must change parameters and not repeat a tested setting')
        response = {**response, 'parameters': p}
    return response


def qualified_core(gate):
    if not gate.get('available'):
        raise ValueError('Missing scoring/image evidence is not a failed quality measurement')
    return gate.get('qualified_particles', 0) > 2000


def promotion_pass(gate):
    from webapp.pilot_approval import approved_diversity
    return qualified_core(gate) and (gate.get('supported_group_count', 0) >= 3 or approved_diversity(gate))


def next_pilot_step(gate, count, maximum):
    if qualified_core(gate):
        return 'core_and_recovery'
    return 'expand' if count < maximum else 'adjust_parameters'


def validate_adjustment(response, previous, history=()):
    target = response.get('target')
    if target not in {'inspect', 'extract'} or not isinstance(response.get('reason'), str) or not response['reason'].strip():
        raise ValueError('Choose target inspect|extract and provide evidence-based reason')
    if target == 'inspect':
        parameters = inspect_parameters(response.get('parameters', {}))
        original = inspect_parameters(previous['thresholds'])
        tested = [inspect_parameters(row['thresholds']) for row in history
                  if row.get('box_size_pix') == previous['box_size_pix']]
    else:
        parameters = response.get('parameters', {})
        validate_visual_review(dict(action='revise', reason=response['reason'], parameters=parameters),
                               'extract', dict(box_size_pix=previous['box_size_pix']), [])
        original = dict(box_size_pix=previous['box_size_pix'])
        tested = [dict(box_size_pix=row['box_size_pix']) for row in history
                  if inspect_parameters(row['thresholds']) == inspect_parameters(previous['thresholds'])]
    if parameters == original or parameters in tested:
        raise ValueError('Change the selected factor; do not repeat a tested recipe on this fixed subset')
    return {**response, 'parameters': parameters}


def combine_gate(entries, policy):
    """Separate class ID namespaces; disjoint populations are checked by caller."""
    statistics, scores, images, provenance = [], [], {}, []
    for context, source_images, selected in entries:
        counts = {r['class_id']: r['particle_count'] for r in context['class_statistics']}
        sift = context['cryosift']
        source = context['source']
        if not sift.get('success') or any(sift.get(k) != source.get(k) for k in ('project_uid', 'job_uid')):
            raise ValueError('Combined evidence must retain authoritative source-scoped scores')
        values = {r['class_id']: r['score'] for r in sift['class_scores']}
        for original in selected:
            key = len(statistics)
            statistics.append(dict(class_id=key, particle_count=counts[original]))
            scores.append(dict(class_id=key, score=values[original]))
            images[key] = source_images[original]
            provenance.append(dict(combined_id=key, source=source, source_class_id=original))
    synthetic = dict(source=dict(project_uid='combined', job_uid='combined'), class_statistics=statistics,
        cryosift=dict(success=True, project_uid='combined', job_uid='combined', class_scores=scores))
    gate = evaluate_gate(synthetic, images, policy)
    gate['class_provenance'] = provenance
    return gate


class SequentialPickerRollout(FreshCoverageRollout):
    def __init__(self, root):
        PilotExperiment.__init__(self, root)
        if self.manifest.get('protocol') != 'sequential-picker-v1':
            raise ValueError('Explicit sequential experiment manifest required')
        if self.manifest['shared_jobs'] != ['J1215', 'J1250', 'J1442']:
            raise ValueError('Only completed shared preprocessing is allowed')
        self.scales = tuple(self.manifest['pilot_scales'])
        self.allowed_samples = (*self.scales, self.manifest['full_micrographs'])
        self.policy = GatePolicy(min_particles=2001)
        self.cryosift_threshold = 3.
        self.api_base = self.manifest['api_override']['base_url']

    def submit(self, slug, kind, sources, params, wait=True):
        # Do NOT inherit FreshCoverageRollout's fixed-route substitutions.
        self.guard_budget(slug)
        return CoverageRollout.submit(self, slug, kind, sources, params, wait)

    def guard_budget(self, slug):
        if not self.state['steps'].get(slug, {}).get('job_uid'):
            created = {r.get('job_uid') for r in self.state['steps'].values() if r.get('job_uid')}
            if len(created) >= self.manifest['max_seed_jobs']:
                raise ProtocolExhausted('Structured phase Job budget exhausted; no promotion bypass')

    def external(self, slug, *args, **kwargs):
        self.guard_budget(slug)
        return CoverageRollout.external(self, slug, *args, **kwargs)

    def model(self, slug, messages, model, validate):
        initial_scale = slug.endswith('_box_parameters')
        legacy = (self.root/'decisions'/slug/'decision.json').exists()
        memory = self.extraction_scale_memory() if initial_scale and not legacy else None
        if memory is not None:
            messages = copy.deepcopy(messages)
            messages[-1]['content'].append(dict(type='text',text=
                'Extend this stage JSON contract with the scale_estimate fields below; '
                'these fields are explicitly allowed in addition to decision/parameters/reason. '+
                SCALE_GUIDANCE+'\nSCALE_MEMORY\n'+json.dumps(memory)))
            original_validate = validate
            def sizing_validate(response):
                response = original_validate(response)
                return validate_scale_response(response,memory,response['parameters']['box_size_pix'])
            validate = sizing_validate
        if any(isinstance(m.get('content'), list) and any(b.get('type')=='image_url' for b in m['content'])
               for m in messages):
            model = 'gpt-6.1-sol'
        result = FreshCoverageRollout.model(self, slug, messages, model, validate)
        if initial_scale and result.get('scale_estimate'):
            self.write(self.root/'scale_estimates'/(slug+'.json'),result)
        return result

    def extraction_scale_memory(self, box=None, pixel_size_A=None):
        source = getattr(self,'scale_memory_source_root',self.root)
        scales = getattr(self,'scales',(50,))
        path = source/'decisions'/f'pilot_r0_n{scales[0]}_box_parameters'/'decision.json'
        original = json.loads(path.read_text()) if path.exists() else {}
        paths = sorted((source/'scale_estimates').glob('*.json'),key=lambda p:p.stat().st_mtime)
        previous = [json.loads(p.read_text()) for p in paths[-6:]]
        if pixel_size_A is None:
            task = source/'task.json'
            config = json.loads(task.read_text()).get('config',{}) if task.exists() else {}
            pixel_size_A = config.get('pixel_size_A')
        return calibrated_scale_memory(pixel_size_A,box,self.state.get('picker_parameters',{}),original,previous)

    def review(self, slug, context, role, excluded=(), reference=None):
        if role != 'core':
            skipped = [r for r in self.state.get('reference_outcomes',{}).values()
                       if r.get('classes')==context.get('source',{}).get('job_uid') and
                       r.get('status') in {'empty','tool_failed'}]
            if reference is None and skipped:
                self.write(self.root/'evidence'/slug/'reference_unavailable.json',
                           dict(selected_class_ids=[],reason='No reference-supported recovery evidence',outcomes=skipped))
                return []
            return CoverageRollout.review(self, slug, context, role, excluded, reference)
        values = {r['class_id']:r['score'] for r in context['cryosift']['class_scores']
                  if finite_number(r.get('score'))}
        counts = {r['class_id']:r['particle_count'] for r in context['class_statistics']}
        ids = [i for i in values if values[i] < 3 and counts.get(i,0)>0 and i not in excluded]
        if sum(counts[i] for i in ids) <= 2000:
            raise ProtocolExhausted('No strict qualified core; subjective fallback is disabled')
        path = self.root/'evidence'/slug/'core_context.json'
        self.write(path, context)
        self.state['current_core'] = dict(source=context['source'], class_ids=ids, context_file=str(path))
        self.persist()
        return ids

    def reference(self, slug, classes, volume):
        outcomes = self.state.setdefault('reference_outcomes',{})
        previous = outcomes.get(slug)
        if previous and previous.get('status') in {'empty','tool_failed'}:
            if previous['classes']!=classes or previous['volume']!=volume:
                raise ValueError('Reference source identity changed on resume')
            return None
        try:
            matched = CoverageRollout.reference(self,slug,classes,volume)
        except RuntimeError as exc:
            uid = self.state['steps'].get(slug+'_match',{}).get('job_uid')
            # Only a confirmed native failure of this OPTIONAL matching Job may
            # be skipped. Transport/schema/lowpass failures remain real errors.
            if not uid or str(exc)!=f'Native Job {uid} failed; trajectory retained':
                raise
            job = self.service('verify optional reference failure',lambda:self.project.find_job(uid))
            if job.type!='reference_select_2D' or job.status!='failed':
                raise
            record = dict(status='tool_failed',classes=classes,volume=volume,job_uid=uid,
                error=str(exc),selected_class_ids=[],
                meaning='Native computation failed; zero outputs are NOT an empty successful selection.')
            outcomes[slug]=record
            self.write(self.root/'evidence'/slug/'reference_outcome.json',record)
            self.persist()
            return None
        count = matched.outputs['templates_selected'].num_items
        if count is None:
            raise ValueError('Completed reference job lacks authoritative selected-template count')
        record = dict(status='empty' if count==0 else 'available',classes=classes,volume=volume,
            job_uid=matched.uid,selected_template_count=count,
            meaning='Completed with no reference-supported supplemental classes' if count==0 else 'Reference evidence available')
        outcomes[slug]=record
        self.write(self.root/'evidence'/slug/'reference_outcome.json',record)
        self.persist()
        return None if count==0 else matched

    def review_parameters(self, slug, stage, visual, current, history):
        memory = None
        instruction = (
            'Review the ACTUAL completed result, not a proposed box/window. '
            'Return {"action":"keep|revise","reason":"cite image examples and uncertainty",'
            '"parameters":{replacement numeric parameters ONLY if revise}}. '
        )
        if stage == 'inspect':
            instruction += ('Green circles are retained and magenta rejected; colors are membership, NOT '
                'quality. Compare to unmarked whole micrographs and the raw NCC/Power density. '
                'If revising, supply ncc_score_thresh, lpower_thresh_min, lpower_thresh_max. '
                'Keep most plausible bright-core signal while removing obvious junk; do not equate '
                'higher scores with better particles or blindly tighten after a classification failure. '
                'Do not use auto cluster or score calibration.')
        else:
            facts = json.loads(visual[0]['text']) if isinstance(visual,list) else visual
            memory = self.extraction_scale_memory(current['box_size_pix'],facts['native_pixel_size_A'])
            instruction += ('Review actual extracted particle sheets and matched compressed full-field '
                'micrographs with scale bars. Is the target complete, with reasonable margin, rather '
                'than clipped or multiple particles per box? Native square box, multiples of 8 within '
                '64..720; if revising supply box_size_pix only. No Fourier downsampling. '
                'About 1.5-2x maximum particle diameter is a starting preference, not a measurement. '+
                SCALE_GUIDANCE+'\nSCALE_MEMORY\n'+json.dumps(memory))
        if isinstance(visual, list):
            content = [dict(type='text', text=instruction + '\n' + json.dumps(dict(current=current, history=history))), *visual]
        else:
            content = self.visual_content(instruction + '\n' + json.dumps(dict(current=current, history=history)), visual)
        legacy = (self.root/'decisions'/slug/'decision.json').exists()
        def validate(r):
            r = validate_visual_review(r,stage,current,[h['parameters'] for h in history])
            if memory is not None and (not legacy or r.get('scale_estimate')):
                size = r['parameters']['box_size_pix'] if r['action']=='revise' else current['box_size_pix']
                r = validate_scale_response(r,memory,size)
            return r
        result = self.model(slug, [dict(role='system', content='Review one cryo-EM stage, return the requested JSON only.'),
            dict(role='user', content=content)], 'gpt-6.1-sol',validate)
        if memory is not None and result.get('scale_estimate'):
            self.write(self.root/'scale_estimates'/(slug+'.json'),result)
        return result

    def inspect_serial(self, tag, picker, parameters, fixed_history=()):
        from inspect_review import build_visual_context
        records = self.state.setdefault('inspect_reviews', {}).setdefault(tag, [])
        current = inspect_parameters(parameters)
        for attempt in range(self.manifest['max_stage_reviews']):
            slug = f'{tag}_i{attempt}'
            inspected = self.submit(slug, 'inspect_picks_v2',
                dict(micrographs=(picker.uid, 'micrographs'), particles=(picker.uid, 'particles')), current)
            visual = self.service('Inspect membership ' + inspected.uid,
                lambda: build_visual_context(self.project.uid, inspected.uid, max_micrographs=3))
            self.write(self.root/'evidence'/slug/'review.json', visual)
            response = self.review_parameters(slug+'_review', 'inspect', visual, current, [*fixed_history,*records[:attempt]])
            record = dict(job=inspected.uid, parameters=current, response=response)
            if len(records) <= attempt:
                records.append(record)
            else:
                records[attempt] = record
            self.persist()
            if response['action'] == 'keep':
                if not inspected.outputs['particles'].num_items:
                    raise ProtocolExhausted('Model accepted an empty Inspect result; no fabricated continuation')
                return inspected, current
            current = response['parameters']
        raise ProtocolExhausted('Inspect serial-review budget exhausted without an accepted result')

    def extract_serial(self, tag, picker, inspected, count, box, fixed_history=()):
        from extract_review_evidence import build_evidence
        records = self.state.setdefault('extract_reviews', {}).setdefault(tag, [])
        current = dict(box_size_pix=box)
        for attempt in range(self.manifest['max_stage_reviews']):
            slug = f'{tag}_e{attempt}'
            extracted = self.submit(slug, 'extract_micrographs_multi',
                dict(micrographs=(inspected.uid, 'micrographs'), particles=(inspected.uid, 'particles')),
                dict(current, num_extract=count, compute_num_gpus=1))
            if extracted.outputs['particles'].num_items < 4:
                raise ProtocolExhausted('Fewer than four extracted particles; visual review unavailable')
            content, facts = self.service('actual Extract evidence ' + extracted.uid, lambda: build_evidence(
                self.project.uid, extracted.uid, inspected.uid, self.root/'evidence'/slug/'actual_stacks'))
            response = self.review_parameters(slug+'_review', 'extract', content, current, [*fixed_history,*records[:attempt]])
            record = dict(job=extracted.uid, parameters=current, response=response,
                          sample_ids=[s['sample_id'] for s in facts['samples']])
            if len(records) <= attempt:
                records.append(record)
            else:
                records[attempt] = record
            self.persist()
            if response['action'] == 'keep':
                return extracted, current['box_size_pix']
            current = response['parameters']
        raise ProtocolExhausted('Extract serial-review budget exhausted without an accepted result')

    def sample_picker(self, count):
        from vision_inputs import build_micrograph_preview_visual_context
        ctf = self.manifest['shared_jobs'][-1]
        source = self.project.find_job(ctf).load_output('exposures')
        subset = self.external('sample_n'+str(count), 'exposure', 'exposures',
            dict(source=(ctf, 'exposures')), source.take(np.arange(count)))
        if 'picker_parameters' not in self.state:
            visual = self.service('pilot micrographs', lambda: build_micrograph_preview_visual_context(
                self.project.uid, subset.uid, output_name='exposures', max_micrographs=3))
            self.state['picker_parameters'] = self.stage('pilot_picker_parameters', 'blob_picker_diameter', visual)
            self.persist()
        picker = self.submit('sample_n'+str(count)+'_picker', 'blob_picker_gpu',
            dict(micrographs=(subset.uid, 'exposures')), dict(self.state['picker_parameters'], num_process=count))
        if not picker.outputs['particles'].num_items:
            raise ProtocolExhausted('Blob output empty; no downstream score can be collected')
        return picker

    def request_inspect(self, tag, picker, feedback=None):
        from vision_inputs import build_pick_inspection_visual_context
        visual = self.service('raw pick evidence', lambda: build_pick_inspection_visual_context(
            self.project.uid, picker.uid, max_micrographs=3))
        if feedback is not None:
            raise ValueError('Maximum-sample failures must use joint Inspect/Extract diagnosis')
        return inspect_parameters(self.stage(tag, 'inspect_picks_threshold', visual))

    def request_adjustment(self, tag, previous, history=()):
        from inspect_review import build_visual_context
        from extract_review_evidence import build_evidence
        from upstream_evidence import metadata
        inspection = self.service('maximum-sample Inspect evidence', lambda: build_visual_context(
            self.project.uid, previous['inspect'], max_micrographs=2))
        actual, facts = self.service('maximum-sample extracted particles', lambda: build_evidence(
            self.project.uid, previous['extract'], previous['inspect'], self.root/'evidence'/tag/'actual_stacks'))
        context = (json.loads((self.root/'evidence'/previous['evidence_slug']/'context.json').read_text())
                   if previous.get('evidence_slug') else None)
        memory = self.extraction_scale_memory(previous['box_size_pix'],facts['native_pixel_size_A'])
        instruction = (
            'The MAXIMUM sample has failed quality or shape-diversity validation. Do not expand samples '
            'or immediately blame Inspect. Diagnose the completed Inspect membership and ACTUAL Extract '
            'stacks together with Class 2D/CryoSift. Choose ONE factor for the next controlled trial: '
            '(1) target=inspect for retained junk/empty picks or excluded real/weak-signal particles; '
            'supply a changed ncc_score_thresh, lpower_thresh_min and lpower_thresh_max. Box stays EXACTLY '
            'unchanged; failure does not imply stricter filtering. '
            '(2) target=extract for clipping, excessive blank margin or multiple particles per box; '
            'supply ONLY box_size_pix (native multiple of 8 in 64..720, no Fourier crop). Reuse the SAME '
            'completed Inspect particle population and thresholds. Do not change both factors. '
            'Reason about the bright core, retained/rejected examples, actual particle boundaries and '
            'class images. A few neighbors are acceptable; prioritize whole-particle completeness. '
            'No official references or known target structure are available. '
            'Return {"target":"inspect|extract","parameters":{numeric fields for ONLY that factor},'
            '"reason":"specific evidence, uncertainty and predicted improvement"}. '
            'If target=extract, also return the scale_estimate fields requested below. '+SCALE_GUIDANCE+'\n' +
            json.dumps(dict(previous=previous, tested_recipes=list(history),
                            scale_memory=memory,class_results=metadata(context) if context else None)))
        content = self.visual_content(instruction, inspection)
        # Include both stack sheets and two matched whole fields; avoid uploading
        # every diagnostic image when the model already has the Inspect fields.
        content += [actual[0], *[b for b in actual[1:] if b.get('type')=='image_url'][:4]]
        if context:
            content.append(dict(type='image_url', image_url=dict(url=context['contact_sheet']['data_url'])))
        self.write(self.root/'evidence'/tag/'diagnostic_sources.json', dict(inspect=previous['inspect'],
            extract=previous['extract'], classes=previous.get('classes'), stack_samples=facts['samples'],
            inspected_micrographs=[p['micrograph_uid'] for p in inspection['panels']]))
        legacy = (self.root/'decisions'/tag/'decision.json').exists()
        def validate(r):
            r = validate_adjustment(r,previous,history)
            if r['target']=='extract' and (not legacy or r.get('scale_estimate')):
                r = validate_scale_response(r,memory,r['parameters']['box_size_pix'])
            return r
        result = self.model(tag,[dict(role='system',content=POLICY),dict(role='user',content=content)],
                            'gpt-6.1-sol',validate)
        if result.get('scale_estimate'):
            self.write(self.root/'scale_estimates'/(tag+'.json'),result)
        return result

    def apply_adjustment(self, tag, previous, decision, history=()):
        decision = validate_adjustment(decision, previous, history)
        picker = self.project.find_job(previous['picker'])
        count, box = previous['count'], previous['box_size_pix']
        thresholds = inspect_parameters(previous['thresholds'])
        tested = [*history, previous]
        if decision['target'] == 'extract':
            inspected = self.project.find_job(previous['inspect'])
            fixed = [dict(job=r['extract'],parameters=dict(box_size_pix=r['box_size_pix'])) for r in tested
                     if inspect_parameters(r['thresholds'])==thresholds]
            extracted, box = self.extract_serial(tag, picker, inspected, count,
                decision['parameters']['box_size_pix'],fixed_history=fixed)
            validate_adjustment(dict(decision, parameters=dict(box_size_pix=box)), previous, history)
        else:
            fixed = [dict(job=r['inspect'],parameters=inspect_parameters(r['thresholds'])) for r in tested
                     if r['box_size_pix']==box]
            inspected, thresholds = self.inspect_serial(tag, picker, decision['parameters'],fixed_history=fixed)
            validate_adjustment(dict(decision, parameters=thresholds), previous, history)
            # The selected intervention is Inspect, so keep geometry controlled.
            extracted = self.submit(tag+'_fixed_box_extract', 'extract_micrographs_multi',
                dict(micrographs=(inspected.uid,'micrographs'), particles=(inspected.uid,'particles')),
                dict(box_size_pix=box, num_extract=count, compute_num_gpus=1))
        row = self.classify(tag, extracted, box, picker, inspected, count, thresholds)
        if previous.get('kind'):
            row['kind'] = previous['kind']
        self.write(self.root/'evidence'/tag/'controlled_intervention.json', dict(target=decision['target'],
            previous_picker=previous['picker'], actual_picker=picker.uid, previous_inspect=previous['inspect'],
            actual_inspect=inspected.uid, previous_box=previous['box_size_pix'], actual_box=box,
            previous_thresholds=previous['thresholds'], actual_thresholds=thresholds,
            sample_count=count, reason=decision['reason']))
        return row

    def classify(self, tag, extracted, box, picker, inspected, count, thresholds):
        number = extracted.outputs['particles'].num_items
        if number < 100:
            return dict(executable=False, particles=number, count=count, thresholds=thresholds,
                        box_size_pix=box, picker=picker.uid, inspect=inspected.uid, extract=extracted.uid)
        classified = self.submit(tag+'_class', 'class_2D_new', dict(particles=(extracted.uid, 'particles')),
            dict(class2D_K=min(50, max(10, number//500)), class2D_num_full_iter=4,
                 random_seed=26026, compute_num_gpus=1))
        slug = tag+'_classes'
        context, gate = self.readout(slug, classified)
        qualified_core(gate)  # Validate that unavailable scores cannot trigger expansion/acceptance.
        return dict(executable=True, classes=classified.uid, evidence_slug=slug, particles=number,
            count=count, box_size_pix=box, thresholds=thresholds, picker=picker.uid, inspect=inspected.uid,
            extract=extracted.uid, picker_parameters=self.state['picker_parameters'], score=context['score_summary'], gate=gate)

    def core_refinements(self, tag, core, initial):
        mode = self.manifest.get('pilot_core_refinement', 'homo_then_nu')
        if mode not in {'homo_only', 'homo_then_nu'}:
            raise ValueError('Unsupported pilot_core_refinement: ' + str(mode))
        homo = self.submit(tag+'_homo', 'homo_refine_new',
            dict(particles=(core.uid, 'particles_selected'), volume=(initial.uid, 'volume_class_0')),
            dict(refine_symmetry='C1'))
        checkpoints = [homo]
        if mode == 'homo_then_nu':
            checkpoints.append(self.submit(tag+'_nu', 'nonuniform_refine_new',
                dict(particles=(homo.uid, 'particles'), volume=(homo.uid, 'volume')),
                dict(refine_symmetry='C1')))
        return checkpoints

    def combine_promotion_gate(self, entries):
        return combine_gate(entries, self.policy)

    def core_and_recovery(self, tag, row):
        if not qualified_core(row['gate']):
            raise ValueError('Cannot reconstruct a core below the strict particle-quality floor')
        source = self.project.find_job(row['classes'])
        context = json.loads((self.root/'evidence'/row['evidence_slug']/'context.json').read_text())
        ids = row['gate']['promotion_class_ids']
        core = self.select(tag+'_core_select', source.uid, ids)
        if core.outputs['particles_selected'].num_items != row['gate']['qualified_particles']:
            raise ValueError('Native selected core count differs from scored class population')
        core_path = self.root/'evidence'/tag/'core_context.json'
        self.write(core_path, context)
        self.state['current_core'] = dict(source=context['source'], class_ids=ids, context_file=str(core_path))
        self.persist()
        initial = self.submit(tag+'_abinit', 'homo_abinit', dict(particles=(core.uid, 'particles_selected')), dict(abinit_K=1))
        checkpoints = self.core_refinements(tag, core, initial)
        reference_refine = checkpoints[-1]
        retained = [(core, context, ids)]
        gate = row['gate']
        if not promotion_pass(gate):
            pool = class_pool(context, 'recovery', ids)
            if pool:
                weak = self.select(tag+'_weak_select', source.uid, pool)
                n = weak.outputs['particles_selected'].num_items
                if n >= 100:
                    recovered = self.submit(tag+'_weak_reclass', 'class_2D_new',
                        dict(particles=(weak.uid, 'particles_selected')),
                        dict(class2D_K=min(30, max(5, n//500)), class2D_num_full_iter=4,
                             random_seed=26027, compute_num_gpus=1))
                    c, _ = self.readout(tag+'_weak_classes', recovered)
                    matched = self.reference(tag+'_reference', recovered.uid, reference_refine.uid)
                    extra_ids = self.review(tag+'_weak_review', c, 'recovered', reference=matched)
                    if extra_ids:
                        extra = self.select(tag+'_compatible_select', recovered.uid, extra_ids)
                        if np.intersect1d(core.load_output('particles_selected')['uid'],
                                          extra.load_output('particles_selected')['uid']).size:
                            raise ValueError('Core/recovery overlap would double-count quality support')
                        retained.append((extra, c, extra_ids))
                        gate = self.combine_promotion_gate([(ctx, self.service('class images', lambda ctx=ctx: load_class_images(
                            self.project, ctx['source']['job_uid'])), selected) for _, ctx, selected in retained])
                        self.write(self.root/'evidence'/tag/'combined_gate.json', gate)
                        combined = union_from_source(source.load_output('particles'),
                            [j.load_output('particles_selected') for j, _, _ in retained])
                        union = self.external(tag+'_union', 'particle', 'particles', dict(source=(source.uid, 'particles')), combined)
                        checkpoints.append(self.submit(tag+'_union_nu', 'nonuniform_refine_new',
                            dict(particles=(union.uid, 'particles'), volume=(reference_refine.uid, 'volume')),
                            dict(refine_symmetry='C1', refine_res_init=30., refine_gs_resplit=True)))
        best = min(checkpoints, key=self.resolution)
        result = dict(gate=gate, best_refine=best.uid, best_resolution_A=self.resolution(best),
            selections=[dict(job=j.uid, source=c['source'], ids=i) for j, c, i in retained])
        if promotion_pass(gate):
            data = [j.load_output('templates_selected').filter_fields(lambda f: f=='uid' or f.startswith('blob/'))
                    for j, _, _ in retained]
            bank = self.external(tag+'_bank', 'template', 'templates',
                {'source_'+str(i):(j.uid, 'templates_selected') for i, (j, _, _) in enumerate(retained)},
                data[0].append(*data[1:]) if len(data)>1 else data[0])
            result['bank_job'] = bank.uid
        return result

    def pilot(self):
        from vision_inputs import build_extract_box_visual_context
        if self.state.get('pilot_accepted'):
            if not promotion_pass(self.state['pilot_accepted']['promotion_gate']):
                raise ValueError('Cached pilot does not satisfy the objective promotion gate')
            return self.state['pilot_accepted']
        trials = self.state.setdefault('sequential_trials', {})
        feedback = None
        rounds = self.manifest.get('max_adjustment_rounds', self.manifest.get('max_inspect_rounds', 3))
        for revision in range(rounds):
            scales = self.scales if revision == 0 else self.scales[-1:]
            thresholds, box = None, None
            for count in scales:
                tag = f'pilot_r{revision}_n{count}'
                record = trials.setdefault(tag, {})
                picker = self.sample_picker(count)
                if 'row' not in record:
                    if revision > 0:
                        history = [r['row'] for k,r in trials.items() if k!=tag and r.get('row',{}).get('count')==count]
                        if 'adjustment' not in record:
                            record['adjustment'] = self.request_adjustment(tag+'_diagnosis', feedback, history)
                            self.persist()
                        record['row'] = self.apply_adjustment(tag, feedback, record['adjustment'], history)
                    elif thresholds is None:
                        p = self.request_inspect(tag+'_parameters', picker)
                        inspected, thresholds = self.inspect_serial(tag, picker, p)
                        visual = self.service('initial box evidence', lambda: build_extract_box_visual_context(
                            self.project.uid, picker.uid, particle_job_uid=inspected.uid, max_examples=3))
                        box = self.stage(tag+'_box_parameters', 'extract_box_size', visual)['box_size_pix']
                        extracted, box = self.extract_serial(tag, picker, inspected, count, box)
                    else:
                        # Expansion is a controlled change in sample size ONLY.
                        inspected = self.submit(tag+'_frozen_inspect', 'inspect_picks_v2',
                            dict(micrographs=(picker.uid, 'micrographs'), particles=(picker.uid, 'particles')), thresholds)
                        extracted = self.submit(tag+'_frozen_extract', 'extract_micrographs_multi',
                            dict(micrographs=(inspected.uid, 'micrographs'), particles=(inspected.uid, 'particles')),
                            dict(box_size_pix=box, num_extract=count, compute_num_gpus=1))
                    if 'row' not in record:
                        record['row'] = self.classify(tag, extracted, box, picker, inspected, count, thresholds)
                    self.persist()
                row = record['row']
                thresholds, box = row['thresholds'], row.get('box_size_pix', box)
                feedback = row
                if not row.get('executable'):
                    if count == self.scales[-1]:
                        record['next_step'] = 'adjust_parameters'
                        self.persist()
                    continue
                action = next_pilot_step(row['gate'], count, self.scales[-1])
                record['next_step'] = action
                if action == 'core_and_recovery':
                    if 'reconstruction' not in record:
                        record['reconstruction'] = self.core_and_recovery(tag, row)
                        self.persist()
                    outcome = record['reconstruction']
                    if promotion_pass(outcome['gate']):
                        accepted = dict(row, promotion_gate=outcome['gate'], bank_job=outcome['bank_job'],
                            best_refine=outcome['best_refine'], selections=outcome['selections'], tag=tag)
                        self.state['pilot_accepted'] = accepted
                        self.state['pilot_recipe'] = accepted
                        self.state['bank_job'] = accepted['bank_job']
                        self.persist()
                        return accepted
                    record['next_step'] = 'expand' if count < self.scales[-1] else 'adjust_parameters'
                    feedback = dict(row, recovery_gate=outcome['gate'])
                self.persist()
        raise ProtocolExhausted('Maximum samples and Inspect/Extract adjustment rounds exhausted; objective promotion not met')

    def template_parameters(self):
        if self.state.get('template_parameters'):
            return self.state['template_parameters']
        recipe = self.state['pilot_accepted']
        pixel = getattr(self, 'native_pixel_size_A', .856)
        content = [dict(type='text', text='Estimate whole-particle Template Picker diameter in Angstroms '
            f'from the retained templates. Each tile spans {recipe["box_size_pix"]*pixel:.4f} A '
            f'({recipe["box_size_pix"]} native pixels at {pixel} A/px); rendered pixels are not native. '
            'Return {"diameter":number,"reason":"evidence and uncertainty"}. Use native min_distance default.')]
        for selection in recipe['selections']:
            c = self.classes('template_source_'+selection['source']['job_uid'],
                             self.project.find_job(selection['source']['job_uid']))
            content += [dict(type='text', text=json.dumps(selection)),
                dict(type='image_url', image_url=dict(url=c['contact_sheet']['data_url']))]
        def validate(r):
            if not finite_number(r.get('diameter')) or not 20 <= r['diameter'] <= 600 or not r.get('reason'):
                raise ValueError('Finite whole-particle diameter 20..600 A and reason required')
            return r
        r = self.model('template_diameter', [dict(role='system', content=POLICY), dict(role='user', content=content)],
                       'gpt-6.1-sol', validate)
        self.state['template_parameters'] = dict(diameter=r['diameter'])
        self.persist()
        return self.state['template_parameters']

    def route(self, tag, micrographs, count, kind, bank, thresholds=None, review=False, box=None):
        from vision_inputs import build_pick_inspection_visual_context
        recipe = self.state['pilot_accepted']
        box = recipe['box_size_pix'] if box is None else box
        params = dict(self.template_parameters() if kind=='template' else self.state['picker_parameters'], num_process=count)
        sources = dict(micrographs=micrographs)
        if kind == 'template':
            sources['templates'] = (bank.uid, 'templates')
        picker_slug = f'confirmation_n{count}_{kind}_picker' if tag.startswith('confirm_') else tag+'_picker'
        picker = self.submit(picker_slug, kind+'_picker_gpu', sources, params)
        if not picker.outputs['particles'].num_items:
            return dict(kind=kind, count=count, executable=False, reason='No picks')
        if thresholds is None:
            visual = self.service('confirmation raw picks', lambda: build_pick_inspection_visual_context(
                self.project.uid, picker.uid, max_micrographs=3))
            thresholds = inspect_parameters(self.stage(tag+'_parameters', 'inspect_picks_threshold', visual))
        if review:
            inspected, thresholds = self.inspect_serial(tag, picker, thresholds)
        else:
            inspected = self.submit(tag+'_inspect', 'inspect_picks_v2',
                dict(micrographs=(picker.uid, 'micrographs'), particles=(picker.uid, 'particles')), thresholds)
        if not inspected.outputs['particles'].num_items:
            return dict(kind=kind, count=count, executable=False, reason='Empty Inspect', thresholds=thresholds)
        extracted = self.submit(tag+'_extract', 'extract_micrographs_multi',
            dict(micrographs=(inspected.uid, 'micrographs'), particles=(inspected.uid, 'particles')),
            dict(box_size_pix=box, num_extract=count, compute_num_gpus=1))
        row = self.classify(tag, extracted, box, picker, inspected, count, thresholds)
        return dict(row, kind=kind)

    def choose_confirmation(self, routes, label):
        eligible = {k:r for k,r in routes.items() if r.get('executable') and promotion_pass(r['gate'])}
        if not eligible:
            return None
        content = [dict(type='text', text=POLICY+' Compare qualified Blob/Template routes on the SAME '
            'independent subset, using quality-qualified yield, supported shape groups and quality, NOT '
            'whole-population mean/total yield. Choose one advertised route. '
            'Return {"route":"blob|template","reason":"evidence"}.\n'+json.dumps(eligible))]
        for name,row in eligible.items():
            c=json.loads((self.root/'evidence'/row['evidence_slug']/'context.json').read_text())
            content += [dict(type='text',text=name+' '+row['classes']),
                dict(type='image_url',image_url=dict(url=c['contact_sheet']['data_url']))]
        def validate(r):
            if r.get('route') not in eligible or not r.get('reason'):
                raise ValueError('Select a quantitatively qualified route')
            return r
        chosen = self.model('confirmation_choice_'+label, [dict(role='system',content=POLICY),
            dict(role='user',content=content)], 'gpt-6.1-sol', validate)['route']
        return routes[chosen]

    def confirm(self, recipe):
        ctf = self.manifest['shared_jobs'][-1]
        source = self.project.find_job(ctf).load_output('exposures')
        pilot = self.project.find_job(self.state['steps']['sample_n'+str(recipe['count'])]['job_uid']).load_output('exposures')
        indices = heldout_indices(source['uid'], pilot['uid'], self.scales[-1])
        self.write(self.root/'independence.json', dict(pilot_uids=list(map(int,pilot['uid'])),
            confirmation_max_uids=list(map(int,source['uid'][indices])), disjoint=True,
            note='Final full data include pilot and confirmation; not a held-out outcome benchmark.'))
        bank = self.project.find_job(recipe['bank_job'])
        confirmations = self.state.setdefault('confirmations', {})
        for count in self.scales:
            subset = self.external('confirmation_n'+str(count), 'exposure', 'exposures',
                dict(source=(ctf, 'exposures')), source.take(indices[:count]))
            routes = confirmations.setdefault(str(count), {})
            for kind in ('blob', 'template'):
                if kind not in routes:
                    if count == self.scales[0]:
                        thresholds = recipe['thresholds'] if kind == 'blob' else None
                    else:
                        thresholds = confirmations[str(self.scales[self.scales.index(count)-1])][kind].get('thresholds')
                    routes[kind] = self.route(f'confirm_n{count}_{kind}', (subset.uid,'exposures'), count,
                        kind, bank, thresholds, review=count==self.scales[0])
                    self.persist()
            selected = self.choose_confirmation(routes,'n'+str(count))
            if selected:
                return selected
        for revision in range(self.manifest['max_confirmation_adjustments']):
            revised = confirmations.setdefault('adjustment_'+str(revision), {})
            for kind in ('blob','template'):
                row = routes[kind]
                if kind not in revised and row.get('executable'):
                    tag = f'confirm_adjustment_{revision}_{kind}'
                    history = [r[kind] for key,r in confirmations.items() if key!=('adjustment_'+str(revision))
                               and kind in r and r[kind].get('count')==self.scales[-1]]
                    decision = self.request_adjustment(tag+'_diagnosis',row,history)
                    revised[kind] = self.apply_adjustment(tag,row,decision,history)
                    self.persist()
            selected = self.choose_confirmation(revised,'adjustment_'+str(revision))
            if selected:
                return selected
            routes = revised
        raise ProtocolExhausted('Independent confirmation exhausted without qualified quantity/diversity; no full-data promotion')

    def run(self):
        if self.state.get('rollout_complete'):
            return self.state['best_refine']
        if self.state.get('status') == 'protocol_exhausted':
            return None
        try:
            recipe = self.pilot()
            if not promotion_pass(recipe['promotion_gate']):
                raise ValueError('No qualified pilot; independent/full-data routes are forbidden')
            if 'confirmed_route' not in self.state:
                self.state['confirmed_route'] = self.confirm(recipe)
                self.state['chosen_route'] = self.state['confirmed_route']['kind']
                self.persist()
            confirmed = self.state['confirmed_route']
            if not promotion_pass(confirmed['gate']):
                raise ValueError('Persisted confirmation does not satisfy objective promotion')
            if 'full' not in self.state:
                self.state['full'] = self.route('full', (self.manifest['shared_jobs'][-1],'exposures'),
                    self.manifest['full_micrographs'], confirmed['kind'], self.project.find_job(recipe['bank_job']),
                    confirmed['thresholds'], box=confirmed['box_size_pix'])
                self.persist()
            full = self.state['full']
            if not full.get('executable') or not qualified_core(full['gate']):
                raise ProtocolExhausted('Full-data route has no qualified core; no low-quality fallback reconstruction')
            best = self.reconstruct(full)
            self.state.update(rollout_complete=True, status='seeded_for_planner', best_refine=best.uid,
                              best_resolution_A=self.resolution(best))
            self.persist()
            return best.uid
        except ProtocolExhausted as exc:
            self.state.update(status='protocol_exhausted', stop_reason=str(exc))
            self.persist()
            self.write(self.root/'protocol_outcome.json', dict(status='failed_objective_validation',
                reason=str(exc), full_data_promoted='full' in self.state, awaiting_human_input=False))
            return None
