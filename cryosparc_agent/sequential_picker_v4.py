"""Isolated post-W145 protocol; the running v3 implementation is unchanged."""
import copy
import json
import math
import re

from sequential_picker_rollout import (
    SequentialPickerRollout, ProtocolExhausted, POLICY as LEGACY_POLICY,
    qualified_core, promotion_pass,
)


POLICY = LEGACY_POLICY + (
    ' V4 supersedes the unconditional sample-expansion rule above: expand only '
    'when source-scoped CryoSift evidence supports a signal-limited population. '
    'No credible signal requires Inspect/Extract diagnosis at the CURRENT sample '
    'size; a failure does not automatically justify stricter filtering. '
    'After pilot qualification compare Template once against the existing Blob '
    'result on the SAME pilot images, then promote the selected recipe directly '
    'to full data. This is not held-out validation. '
)
SCALE_CONSISTENCY = (
    ' V4 cross-stage scale contract: Blob bounds are search bounds, not ground '
    'truth. If scale_estimate.diameter_max_A exceeds the upstream Blob diameter_max, '
    'provide picker_bound_revision_reason citing calibrated whole-particle '
    'evidence, confidence, and whether the earlier picker range needs review. '
    'Do not invent a larger extent to justify a preferred box. Low-confidence '
    'estimates favor completeness but are not new measured facts.'
)


def validate_picker_scale(response, picker):
    diameter = (response.get('scale_estimate') or {}).get('diameter_max_A')
    bound = picker.get('diameter_max')
    if (isinstance(diameter, (int, float)) and isinstance(bound, (int, float))
            and diameter > bound and not str(response.get('picker_bound_revision_reason') or '').strip()):
        raise ValueError('Explain the molecular extent exceeding the upstream Blob search maximum')
    return response


def signal_action(context, gate, count, maximum, near_used=False):
    """Heuristic expansion evidence is never a replacement for strict promotion."""
    if qualified_core(gate):
        return 'core_and_recovery', 'strict quality floor reached'
    if count >= maximum:
        return 'adjust_parameters', 'maximum sample without qualified core'
    sift = context.get('cryosift') or {}
    source = context.get('source') or {}
    if (not sift.get('success') or any(sift.get(k) != source.get(k)
                                     for k in ('project_uid', 'job_uid'))):
        raise ValueError('Source-scoped CryoSift evidence required for sample expansion')
    counts = {r['class_id']: r['particle_count'] for r in context['class_statistics']}
    strict, near = 0, 0
    for row in sift['class_scores']:
        score, n = row.get('score'), counts.get(row['class_id'], 0)
        if not isinstance(score, (int, float)) or not math.isfinite(score) or n < 100:
            continue
        if score < 3:
            strict += n
        elif score < 3.5:
            near += n
    if strict >= 200:
        return 'expand', 'supported strict-quality signal but insufficient population'
    if near >= 500 and not near_used:
        return 'expand_near', 'one bounded expansion for supported near-threshold signal'
    return 'adjust_parameters', 'no sufficiently supported signal; diagnose current Inspect/Extract'


def supported_symmetry(evidence):
    target = str(evidence.get('recommended_symmetry') or '').upper()
    group = re.fullmatch(r'([CD])(\d+)', target)
    valid = target in {'T', 'O', 'I'} or (group and 2 <= int(group[2]) <= 60)
    score = evidence.get('average_fsc')
    return bool(evidence.get('success') and valid and isinstance(score, (int, float))
                and math.isfinite(score) and score >= .75)


class SequentialPickerV4(SequentialPickerRollout):
    def __init__(self, root):
        super().__init__(root)
        if self.manifest.get('protocol_revision') != 4:
            raise ValueError('V4 requires an explicit new experiment manifest')

    def model(self, slug, messages, model, validate):
        messages = copy.deepcopy(messages)
        messages.insert(0, dict(role='system', content=POLICY + SCALE_CONSISTENCY))
        # The legacy joint-diagnosis prompt assumes maximum-sample failure.
        # V4 also invokes it early when measured signal support is absent.
        for message in messages:
            content = message.get('content')
            if isinstance(content, str):
                message['content'] = content.replace(LEGACY_POLICY, POLICY)
            if isinstance(content, list):
                for block in content:
                    if block.get('type') == 'text':
                        block['text'] = block['text'].replace(LEGACY_POLICY, POLICY).replace(
                            'The MAXIMUM sample has failed quality or shape-diversity validation.',
                            'The CURRENT sample needs parameter diagnosis; its actual count is recorded below.')
        def checked(response):
            return validate_picker_scale(validate(response), self.state.get('picker_parameters', {}))
        return super().model(slug, messages, model, checked)

    def submit(self, slug, kind, sources, params, wait=True):
        if slug.startswith('full_') and kind in {'homo_refine_new', 'nonuniform_refine_new'}:
            params = dict(params, refine_symmetry=self.state.get('promoted_symmetry', 'C1'))
        return super().submit(slug, kind, sources, params, wait)

    def core_refinements(self, tag, core, initial):
        checkpoints = super().core_refinements(tag, core, initial)
        baseline = checkpoints[0]
        if not math.isfinite(self.resolution(baseline)):
            return checkpoints
        records = self.state.setdefault('early_symmetry', {})
        if baseline.uid not in records:
            from proshade_adapter import analyze_volume_symmetry
            evidence = self.service('early pilot symmetry', lambda: analyze_volume_symmetry(
                self.project.uid, baseline.uid, output_dir=str(self.root/'symmetry'/baseline.uid)))
            records[baseline.uid] = dict(evidence=evidence, baseline=baseline.uid)
            self.persist()
        record = records[baseline.uid]
        evidence = record['evidence']
        if supported_symmetry(evidence):
            if 'decision' not in record:
                target = evidence['recommended_symmetry'].upper()
                def validate(r):
                    if type(r.get('compare')) is not bool or not str(r.get('reason') or '').strip():
                        raise ValueError('Return compare boolean and evidence-based reason')
                    return r
                record['decision'] = self.model(tag+'_symmetry_decision', [dict(role='user', content=
                    'Unknown dataset symmetry. ProSHADE is advisory, not truth. Decide whether to '
                    'compare its proposed '+target+' once with the preserved C1 baseline, using the '
                    'SAME selected particles, initial volume and refinement type. Do not infer known '
                    'target identity. Return {"compare":true|false,"reason":"evidence and uncertainty"}.\n'
                    +json.dumps(evidence))], 'gpt-6.1-sol', validate)
                self.persist()
            if record['decision']['compare']:
                tested = self.submit(tag+'_symmetry_compare', 'homo_refine_new',
                    dict(particles=(core.uid, 'particles_selected'), volume=(initial.uid, 'volume_class_0')),
                    dict(refine_symmetry=evidence['recommended_symmetry'].upper()))
                checkpoints.append(tested)
                self.state.setdefault('symmetry_by_refine', {})[tested.uid] = evidence['recommended_symmetry'].upper()
                record['comparison'] = dict(job_uid=tested.uid, resolution_A=self.resolution(tested),
                    baseline_resolution_A=self.resolution(baseline),
                    interpretation='Nominal FSC comparison, not proof of correct symmetry or structure')
                self.persist()
        return checkpoints

    def pilot(self):
        from vision_inputs import build_extract_box_visual_context
        if self.state.get('pilot_accepted'):
            if not promotion_pass(self.state['pilot_accepted']['promotion_gate']):
                raise ValueError('Cached pilot failed objective gate')
            return self.state['pilot_accepted']
        trials = self.state.setdefault('sequential_trials', {})
        feedback = None
        for revision in range(self.manifest['max_adjustment_rounds']):
            scales = self.scales if revision == 0 else tuple(n for n in self.scales if n >= feedback['count'])
            thresholds, box = None, None
            for count in scales:
                tag = f'pilot_r{revision}_n{count}'
                record = trials.setdefault(tag, {})
                picker = self.sample_picker(count)
                if 'row' not in record:
                    if revision > 0 and thresholds is None:
                        history = [r['row'] for k,r in trials.items()
                                   if k != tag and r.get('row', {}).get('count') == count]
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
                        inspected = self.submit(tag+'_frozen_inspect', 'inspect_picks_v2',
                            dict(micrographs=(picker.uid, 'micrographs'), particles=(picker.uid, 'particles')), thresholds)
                        extracted = self.submit(tag+'_frozen_extract', 'extract_micrographs_multi',
                            dict(micrographs=(inspected.uid, 'micrographs'), particles=(inspected.uid, 'particles')),
                            dict(box_size_pix=box, num_extract=count, compute_num_gpus=1))
                    if 'row' not in record:
                        record['row'] = self.classify(tag, extracted, box, picker, inspected, count, thresholds)
                    self.persist()
                row = record['row']
                feedback, thresholds, box = row, row['thresholds'], row['box_size_pix']
                if not row.get('executable'):
                    action, reason = 'adjust_parameters', 'too few extracted particles to classify'
                else:
                    context = json.loads((self.root/'evidence'/row['evidence_slug']/'context.json').read_text())
                    action, reason = signal_action(context, row['gate'], count, self.scales[-1],
                        any(r.get('next_step') == 'expand_near' for k,r in trials.items() if k != tag))
                record.update(next_step=action, signal_reason=reason)
                if action == 'core_and_recovery':
                    if 'reconstruction' not in record:
                        record['reconstruction'] = self.core_and_recovery(tag, row)
                        self.persist()
                    outcome = record['reconstruction']
                    if promotion_pass(outcome['gate']):
                        accepted = dict(row, promotion_gate=outcome['gate'], bank_job=outcome['bank_job'],
                            best_refine=outcome['best_refine'], selections=outcome['selections'], tag=tag)
                        self.state.update(pilot_accepted=accepted, pilot_recipe=accepted, bank_job=outcome['bank_job'])
                        self.persist()
                        return accepted
                    action = 'expand' if count < self.scales[-1] else 'adjust_parameters'
                    record['next_step'] = action
                self.persist()
                if action == 'adjust_parameters':
                    break
        raise ProtocolExhausted('Signal-aware pilot adjustments exhausted; no qualified pilot')

    def confirm(self, recipe):
        routes = self.state.setdefault('pilot_route_comparison', {})
        if 'blob' not in routes:
            routes['blob'] = dict(recipe, kind='blob', gate=recipe['promotion_gate'])
            self.persist()
        if 'template' not in routes:
            subset = self.project.find_job(self.state['steps']['sample_n'+str(recipe['count'])]['job_uid'])
            try:
                routes['template'] = self.route('pilot_template_comparison', (subset.uid, 'exposures'),
                    recipe['count'], 'template', self.project.find_job(recipe['bank_job']),
                    thresholds=None, review=True, box=recipe['box_size_pix'])
            except ProtocolExhausted as exc:
                # A measured optional route failure must not erase a qualified Blob recipe.
                # Shared Job-budget exhaustion still stops rather than submitting full data.
                if 'Job budget exhausted' in str(exc):
                    raise
                routes['template'] = dict(kind='template', executable=False, reason=str(exc))
            self.persist()
        selected = self.choose_pilot_route(routes)
        if selected is None:
            raise ValueError('Qualified Blob fallback unexpectedly unavailable')
        self.state['promoted_symmetry'] = self.state.get('symmetry_by_refine', {}).get(recipe['best_refine'], 'C1')
        self.write(self.root/'route_comparison_scope.json', dict(same_pilot=True, independent_validation=False,
            count=recipe['count'], blob_classes=recipe['classes'], chosen=selected['kind'],
            note='Existing qualified Blob route retained; one Template comparison, no independent expansion.'))
        self.persist()
        return selected

    def choose_pilot_route(self, routes):
        eligible = {k:r for k,r in routes.items() if r.get('executable') and promotion_pass(r['gate'])}
        if len(eligible) == 1:
            return next(iter(eligible.values()))
        if not eligible:
            return None
        content = [dict(type='text', text='Compare these two qualified picker routes on the SAME pilot '
            'micrographs. Lower CryoSift is better. Consider qualified yield and supported shape groups, '
            'not total picks or average score alone. Blob may include disjoint weak-class recovery: '
            'its source-specific selections/images are attached separately. Return '
            '{"route":"blob|template","reason":"source-scoped evidence and uncertainty"}.\n'+json.dumps(eligible))]
        for kind, row in eligible.items():
            sources = row.get('selections') or [dict(source=dict(job_uid=row['classes']))]
            for source in sources:
                uid = source['source']['job_uid']
                if uid == row['classes']:
                    context = json.loads((self.root/'evidence'/row['evidence_slug']/'context.json').read_text())
                else:
                    context = self.classes('route_comparison_'+uid, self.project.find_job(uid))
                content += [dict(type='text', text=kind+' '+json.dumps(source)),
                    dict(type='image_url', image_url=dict(url=context['contact_sheet']['data_url']))]
        def validate(r):
            if r.get('route') not in eligible or not str(r.get('reason') or '').strip():
                raise ValueError('Choose an advertised qualified route and give a reason')
            return r
        decision = self.model('same_pilot_route_choice', [dict(role='user', content=content)],
                              'gpt-6.1-sol', validate)
        return eligible[decision['route']]
