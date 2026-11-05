"""Opt-in diameter-aware recovery; existing V4 runs are not changed implicitly."""
import copy
import hashlib
import json
import math

import numpy as np

from sequential_picker_v4 import SequentialPickerV4, POLICY as V4_POLICY, signal_action
from sequential_picker_rollout import ProtocolExhausted, promotion_pass, validate_adjustment
from extract_scale_context import scale_memory


POLICY = V4_POLICY + (
    ' Diameter-aware recovery supersedes older Inspect/Extract-only restrictions: '
    'initial Blob search bounds are uncertain hypotheses, '
    'not fixed facts. Failed Inspect-only changes may indicate a wrong diameter '
    'range. You may revise BOTH Blob diameter bounds as one upstream intervention; '
    'then regenerate picks and re-decide Inspect thresholds and the native Extract '
    'box from NEW evidence. Never silently reuse old score thresholds or geometry '
    'after changing picker scale. Prefer a physical box about 1.5-2 times a '
    'credible whole-particle maximum; do not inflate the size estimate to justify '
    'a preferred box. Existing strict quality gates and total Job budget remain.'
)


def validate_diagnosis(response, previous, history):
    if response.get('target') == 'extract':
        # Explanatory scale evidence is not a native CryoSPARC parameter.
        response = copy.deepcopy(response)
        parameters = response.get('parameters') or {}
        for key in ('scale_estimate', 'outside_preference_reason', 'diameter_revision_reason'):
            if key in parameters:
                value = parameters.pop(key)
                if key in response and response[key] != value:
                    raise ValueError('Conflicting top-level and nested scale evidence: ' + key)
                response[key] = value
        response['parameters'] = parameters
    if response.get('target') != 'blob':
        return validate_adjustment(response, previous, history)
    p = response.get('parameters') or {}
    if set(p) != {'diameter', 'diameter_max'}:
        raise ValueError('Blob intervention requires only diameter and diameter_max, in Angstroms')
    if (any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in p.values())
            or p['diameter'] >= p['diameter_max']):
        raise ValueError('Positive finite ordered Blob diameter range required')
    if not isinstance(response.get('reason'), str) or not response['reason'].strip():
        raise ValueError('Evidence-based diameter revision reason required')
    if any(p == r.get('picker_parameters') for r in [previous, *history]):
        raise ValueError('Do not repeat a tested Blob range on the same subset')
    return response


class DiameterAwareRecovery(SequentialPickerV4):
    def model(self, slug, messages, model, validate):
        messages = copy.deepcopy(messages)
        messages.insert(0, dict(role='system', content=POLICY))
        return super().model(slug, messages, 'gpt-6.1-sol', validate)

    def extraction_scale_memory(self, box=None, pixel_size_A=None):
        tag = self.state.get('blob_scale_branch')
        if not tag:
            return super().extraction_scale_memory(box, pixel_size_A)
        task = json.loads((self.root/'task.json').read_text())
        path = self.root/'decisions'/(tag+'_box_parameters')/'decision.json'
        if not path.exists():
            # Keep the previous explicit estimate visible until a revised one is justified.
            return super().extraction_scale_memory(box, pixel_size_A)
        original = json.loads(path.read_text())
        previous = [json.loads(p.read_text()) for p in sorted((self.root/'scale_estimates').glob(tag+'*.json'))]
        return scale_memory(pixel_size_A or task['config']['pixel_size_A'], box,
                            self.state['picker_parameters'], original, previous)

    def sample_picker(self, count):
        params = self.state['picker_parameters']
        ctf = self.manifest['shared_jobs'][-1]
        source = self.project.find_job(ctf).load_output('exposures')
        subset = self.external('sample_n'+str(count), 'exposure', 'exposures',
            dict(source=(ctf, 'exposures')), source.take(np.arange(count)))
        key = hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:12]
        picker = self.submit(f'diameter_{key}_n{count}_picker', 'blob_picker_gpu',
            dict(micrographs=(subset.uid, 'exposures')), dict(params,num_process=count))
        if not picker.outputs['particles'].num_items:
            raise ProtocolExhausted('Revised Blob range produced no picks')
        return picker

    def request_adjustment(self, tag, previous, history=()):
        from inspect_review import build_visual_context
        from extract_review_evidence import build_evidence
        from vision_inputs import build_micrograph_preview_visual_context
        from upstream_evidence import metadata
        ctf = self.manifest['shared_jobs'][-1]
        subset = self.state['steps']['sample_n'+str(previous['count'])]['job_uid']
        fields = self.service('diameter diagnosis whole fields', lambda: build_micrograph_preview_visual_context(
            self.project.uid, subset, output_name='exposures', max_micrographs=3))
        inspect = self.service('diameter diagnosis membership', lambda: build_visual_context(
            self.project.uid, previous['inspect'], max_micrographs=2))
        stacks, facts = self.service('diameter diagnosis particle stacks', lambda: build_evidence(
            self.project.uid, previous['extract'], previous['inspect'], self.root/'evidence'/tag/'stacks'))
        classes = (json.loads((self.root/'evidence'/previous['evidence_slug']/'context.json').read_text())
                   if previous.get('evidence_slug') else None)
        memory = self.extraction_scale_memory(previous['box_size_pix'], facts['native_pixel_size_A'])
        content = self.visual_content(POLICY+' Diagnose current failed route, considering the whole target '
            'rather than only a bright core. Repeated Inspect-only failures are evidence to reconsider '
            'the original diameter hypothesis, not proof of a threshold problem. Return JSON '
            '{"target":"blob|inspect|extract","parameters":{...},"reason":"evidence and uncertainty"}. '
            'For blob provide diameter AND diameter_max in A. For inspect provide explicit NCC lower '
            'and Power lower/upper. For extract parameters MUST contain only box_size_pix; '
            'scale_estimate, outside_preference_reason and diameter_revision_reason are TOP-LEVEL '
            'evidence fields, NEVER native parameters. scale_estimate.confidence must be exactly '
            'confident or uncertain (use uncertain for moderate/limited evidence). Example: '
            '{"target":"extract","parameters":{"box_size_pix":448},'
            '"scale_estimate":{"diameter_max_A":null,"confidence":"uncertain",'
            '"basis":"state the calibrated evidence"},"reason":"explain the intervention"}. '
            'Supply top-level outside_preference_reason/diameter_revision_reason when required by '
            'scale memory; do not invent a diameter. Blob revisions trigger '
            'fresh downstream Inspect/Extract decisions.\n'+json.dumps(dict(previous=previous,
                tested_recipes=list(history),class_results=metadata(classes) if classes else None,scale_memory=memory)), fields)
        content += self.visual_content('Current retained/rejected picks',inspect)
        content += [stacks[0], *[b for b in stacks[1:] if b.get('type')=='image_url'][:3]]
        if classes:
            content.append(dict(type='image_url', image_url=dict(url=classes['contact_sheet']['data_url'])))
        def validate(r):
            from extract_scale_context import validate_scale_response
            r = validate_diagnosis(r,previous,history)
            if r['target'] == 'extract':
                r = validate_scale_response(r,memory,r['parameters']['box_size_pix'])
            return r
        return self.model(tag, [dict(role='user',content=content)], 'gpt-6.1-sol',validate)

    def apply_adjustment(self, tag, previous, decision, history=()):
        if decision['target'] != 'blob':
            return super().apply_adjustment(tag,previous,decision,history)
        from vision_inputs import build_extract_box_visual_context
        validate_diagnosis(decision,previous,history)
        self.state['picker_parameters'] = dict(decision['parameters'])
        self.state['blob_scale_branch'] = tag
        self.persist()
        count = previous['count']
        picker = self.sample_picker(count)
        p = self.request_inspect(tag+'_parameters',picker)
        inspected, thresholds = self.inspect_serial(tag,picker,p)
        visual = self.service('revised diameter box evidence', lambda: build_extract_box_visual_context(
            self.project.uid,picker.uid,particle_job_uid=inspected.uid,max_examples=3))
        box = self.stage(tag+'_box_parameters','extract_box_size',visual)['box_size_pix']
        extracted, box = self.extract_serial(tag,picker,inspected,count,box)
        row = self.classify(tag,extracted,box,picker,inspected,count,thresholds)
        self.write(self.root/'evidence'/tag/'diameter_intervention.json',dict(
            previous_picker=previous['picker'],previous_parameters=previous.get('picker_parameters'),
            new_picker=picker.uid,new_parameters=decision['parameters'],
            inspect=inspected.uid,box_size_pix=box,reason=decision['reason']))
        return row

    def pilot(self):
        if self.state.get('pilot_accepted'):
            return super().pilot()
        old = list(self.state['sequential_trials'].values())[-1]['row']
        trials = self.state.setdefault('diameter_recovery_trials',{})
        previous = old
        for revision in range(self.manifest.get('diameter_recovery_rounds',3)):
            scales = tuple(n for n in self.scales if n >= previous['count'])
            first = True
            for count in scales:
                tag = f'diameter_recovery_r{revision}_n{count}'
                record = trials.setdefault(tag,{})
                if 'row' not in record:
                    if first:
                        history = [r['row'] for r in self.state['sequential_trials'].values()
                                   if r['row']['count']==count]
                        history += [r['row'] for k,r in trials.items()
                                    if k!=tag and r.get('row',{}).get('count')==count]
                        if 'adjustment' not in record:
                            record['adjustment'] = self.request_adjustment(tag+'_diagnosis',previous,history)
                            self.persist()
                        record['row'] = self.apply_adjustment(tag,previous,record['adjustment'],history)
                    else:
                        picker = self.sample_picker(count)
                        inspected = self.submit(tag+'_inspect','inspect_picks_v2',dict(
                            micrographs=(picker.uid,'micrographs'),particles=(picker.uid,'particles')),previous['thresholds'])
                        extracted = self.submit(tag+'_extract','extract_micrographs_multi',dict(
                            micrographs=(inspected.uid,'micrographs'),particles=(inspected.uid,'particles')),
                            dict(box_size_pix=previous['box_size_pix'],num_extract=count,compute_num_gpus=1))
                        record['row'] = self.classify(tag,extracted,previous['box_size_pix'],picker,
                                                     inspected,count,previous['thresholds'])
                    self.persist()
                row = record['row']
                previous, first = row, False
                if row.get('executable'):
                    context = json.loads((self.root/'evidence'/row['evidence_slug']/'context.json').read_text())
                    action,reason = signal_action(context,row['gate'],count,self.scales[-1],
                        any(r.get('next_step')=='expand_near' for k,r in trials.items() if k!=tag))
                else:
                    action,reason = 'adjust_parameters','Too few particles to classify'
                record.update(next_step=action,reason=reason)
                if action=='core_and_recovery':
                    if 'reconstruction' not in record:
                        record['reconstruction'] = self.core_and_recovery(tag,row)
                        self.persist()
                    result = record['reconstruction']
                    if promotion_pass(result['gate']):
                        accepted = dict(row,promotion_gate=result['gate'],bank_job=result['bank_job'],
                            best_refine=result['best_refine'],selections=result['selections'],tag=tag)
                        self.state.update(pilot_accepted=accepted,pilot_recipe=accepted,bank_job=result['bank_job'])
                        self.persist()
                        return accepted
                    action = 'expand' if count<self.scales[-1] else 'adjust_parameters'
                self.persist()
                if action=='adjust_parameters':
                    break
        raise ProtocolExhausted('Diameter-aware recovery rounds exhausted without a qualified pilot')
