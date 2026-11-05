"""Opt-in cross-dataset pilot gate: strict core plus reference-supported views."""
import copy
import json
import math
from pathlib import Path

import numpy as np

from picker_gate_repeat import FreshDiameterGateRepeat
from picker_blob_recovery import POLICY as BASE_POLICY
from pilot_reclassification_experiment import PilotExperiment
from pilot_objective_gate import GatePolicy, evaluate_gate, load_class_images, pairwise_shape_ncc

PROTOCOL = 'two-tier-picker-20261028'
DEFAULT_DUPLICATE_NCC = .90
POLICY = BASE_POLICY + (
    ' EXPERIMENT-SPECIFIC TWO-TIER AMENDMENT supersedes the requirement that ALL '
    'diversity-supporting classes score below 3: the high-quality core still requires '
    'MORE THAN 2000 particles in classes scoring strictly below 3.0. Separately recovered '
    'classes scoring 3.0 to strictly below 4.0 may supplement missing shapes only after '
    'reference-based acceptance and a new-shape test against the core. They never count '
    'toward the strict core particle floor. At least three supported shape groups and '
    'the existing five-percent group-support rule remain required. Do not auto-accept '
    'all classes below 4, interpret shape groups as measured 3D angles, or treat own-map '
    'reference acceptance as proof of correctness. Retain core/recovery provenance.'
)


def scores_and_counts(context):
    source, sift = context['source'], context['cryosift']
    if not sift.get('success') or any(sift.get(k) != source.get(k) for k in ('project_uid', 'job_uid')):
        raise ValueError('Source-scoped successful CryoSift required')
    scores = {r['class_id']: r['score'] for r in sift['class_scores']
              if type(r.get('score')) in (int, float) and math.isfinite(r['score'])}
    counts = {r['class_id']: r['particle_count'] for r in context['class_statistics']}
    return scores, counts


def gate_policy(manifest):
    # Honor pinned historical protocols instead of changing live runs on resume.
    return GatePolicy(min_particles=2001, duplicate_ncc=manifest.get('gate', {}).get(
        'duplicate_ncc', DEFAULT_DUPLICATE_NCC))


def two_tier_gate(entries, verified, policy):
    """Keep strict particle count separate from the combined diversity population."""
    records, values, images, provenance = [], [], {}, []
    core_particles = 0
    core_score_sum = 0.
    seen = set()
    for context, source_images, ids in entries:
        scores, counts = scores_and_counts(context)
        source = context['source']
        scope = source['project_uid'] + '/' + source['job_uid']
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate selected class IDs')
        for i in ids:
            if (scope, i) in seen:
                raise ValueError('Duplicate source class across retained entries')
            seen.add((scope, i))
            score = scores.get(i)
            if score is None or score >= 4 or counts.get(i, 0) <= 0 or i not in source_images:
                raise ValueError('Missing/invalid selected class evidence')
            if score >= 3 and i not in verified.get(scope, []):
                raise ValueError('Supplement lacks verified reference and new-shape evidence')
            key = len(records)
            records.append(dict(class_id=key, particle_count=counts[i]))
            values.append(dict(class_id=key, score=score))
            images[key] = source_images[i]
            if score < 3:
                core_particles += counts[i]
                core_score_sum += score * counts[i]
            provenance.append(dict(combined_id=key, source=source, source_class_id=i,
                score=score, role='strict_core' if score < 3 else 'verified_supplement'))
    synthetic = dict(source=dict(project_uid='combined', job_uid='combined'), class_statistics=records,
        cryosift=dict(success=True, project_uid='combined', job_uid='combined', class_scores=values))
    diversity_policy = GatePolicy(**{**policy.__dict__, 'cryosift_threshold': 4.0})
    gate = evaluate_gate(synthetic, images, diversity_policy)
    if not gate.get('available'):
        return gate
    retained = gate['qualified_particles']
    gate.update(metric_version=PROTOCOL, retained_particles=retained,
        qualified_particles=core_particles, strict_core_particles=core_particles,
        supplemental_particles=retained-core_particles, core_score_threshold=3.0,
        supplemental_score_threshold=4.0, class_provenance=provenance,
        retained_particle_weighted_score=gate['particle_weighted_score'],
        core_particle_weighted_score=core_score_sum/core_particles if core_particles else None)
    gate['failed_conditions'] = []
    if core_particles < policy.min_particles:
        gate['failed_conditions'].append('insufficient_strict_core_particles')
    if gate['supported_group_count'] < policy.min_groups:
        gate['failed_conditions'].append('insufficient_supported_2d_shape_groups')
    gate['passed'] = not gate['failed_conditions']
    return gate


class TwoTierPicker(FreshDiameterGateRepeat):
    def __init__(self, root):
        PilotExperiment.__init__(self, root)
        if self.manifest.get('experimental_gate') != PROTOCOL:
            raise ValueError('Explicit two-tier experiment manifest required')
        dataset = self.manifest.get('dataset')
        if dataset not in {'10228', '10764'} and not (dataset == '11280' and self.manifest.get('cryolo_initial_scale')):
            raise ValueError('Unsupported experimental dataset')
        self.scales = tuple(self.manifest['pilot_scales'])
        self.allowed_samples = (*self.scales, self.manifest['full_micrographs'])
        self.policy = gate_policy(self.manifest)
        self.cryosift_threshold = 3.0
        self.api_base = self.manifest['api_override']['base_url']
        self.acquisition_facts = dict(self.manifest['acquisition_facts'])
        self.native_pixel_size_A = self.acquisition_facts['psize_A']
        self.initial_scale()

    def model(self, slug, messages, model, validate):
        messages = copy.deepcopy(messages)
        messages.insert(0, dict(role='system', content=POLICY))
        return super().model(slug, messages, model, validate)

    def review(self, slug, context, role, excluded=(), reference=None):
        if role == 'core':
            return super().review(slug, context, role, excluded, reference)
        if reference is None:
            return []
        scores, _ = scores_and_counts(context)
        reference_ids = set(map(int, reference.load_output('templates_selected')['blob/idx']))
        allowed = {i for i in reference_ids if i in scores and scores[i] < 4}
        filtered = copy.deepcopy(context)
        filtered['cryosift']['class_scores'] = [r for r in context['cryosift']['class_scores']
                                               if r['class_id'] in allowed]
        ids = super().review(slug, filtered, role, excluded, reference)
        core = self.state['current_core']
        c = json.loads(Path(core['context_file']).read_text())
        core_images = self.service('two-tier core shapes', lambda: load_class_images(self.project, core['source']['job_uid']))
        candidate_images = self.service('two-tier recovered shapes', lambda: load_class_images(self.project, context['source']['job_uid']))
        ordered = [core_images[i] for i in core['class_ids']]
        retained, audit = [], []
        for i in ids:
            matrix = pairwise_shape_ncc([*ordered, candidate_images[i]], self.policy)
            best = float(matrix[:-1, -1].max())
            keep = i in allowed and (scores[i] < 3 or best < self.policy.duplicate_ncc)
            audit.append(dict(class_id=i, score=scores[i], reference_selected=i in reference_ids,
                              best_core_shape_ncc=best, retained=keep))
            if keep:
                retained.append(i)
        scope = context['source']['project_uid'] + '/' + context['source']['job_uid']
        self.state.setdefault('verified_supplement_ids', {})[scope] = retained
        self.persist()
        self.write(self.root/'evidence'/slug/'two_tier_support.json', dict(
            reference_job=reference.uid, source=context['source'], classes=audit,
            selected_class_ids=retained, shape_proxy_not_true_3d_angles=True))
        return retained

    def combine_promotion_gate(self, entries):
        return two_tier_gate(entries, self.state.get('verified_supplement_ids', {}), self.policy)

    def pilot(self):
        amendment = self.manifest.get('pilot_gate_reassessment')
        if not amendment or self.state.get('pilot_accepted'):
            return super().pilot()
        if self.state.get('planner_started') or self.state.get('full'):
            raise ValueError('Pilot gate reassessment cannot replace an active full-data route')
        if self.policy.duplicate_ncc != amendment['duplicate_ncc']:
            raise ValueError('Reassessment threshold differs from pinned gate policy')
        tag = amendment['tag']
        original = self.state[amendment['bucket']][tag]
        row = copy.deepcopy(original['row'])
        job = self.project.find_job(row['classes'])
        _, gate = self.readout(tag+'_gate_reassessment', job)
        from sequential_picker_rollout import promotion_pass
        if not promotion_pass(gate):
            raise ValueError('Requested existing route does not pass the amended gate')
        row.update(gate=gate, evidence_slug=tag+'_gate_reassessment')
        # Same action slugs adopt recorded native Jobs; only a missing bank is created.
        outcome = self.core_and_recovery(tag, row)
        if not promotion_pass(outcome['gate']):
            raise ValueError('Reassessed reconstruction does not pass the amended gate')
        accepted = dict(row, promotion_gate=outcome['gate'], bank_job=outcome['bank_job'],
            best_refine=outcome['best_refine'], selections=outcome['selections'], tag=tag)
        self.state.update(pilot_accepted=accepted, pilot_recipe=accepted,
            bank_job=outcome['bank_job'], gate_reassessment=dict(
                amendment=amendment, result=outcome))
        self.persist()
        return accepted

    def template_parameters(self):
        if self.state.get('template_parameters'):
            return self.state['template_parameters']
        recipe = self.state['pilot_accepted']
        content = [dict(type='text', text='Estimate whole-particle Template Picker diameter in Angstroms. '
            f'Each retained tile spans {recipe["box_size_pix"] * self.native_pixel_size_A:.4f} A '
            f'({recipe["box_size_pix"]} native pixels at {self.native_pixel_size_A} A/px). '
            'Rendered pixels are not native. Return {"diameter":number,"reason":"evidence and uncertainty"}. '
            'Use native min_distance default.')]
        for selection in recipe['selections']:
            context = self.classes('template_source_'+selection['source']['job_uid'],
                                   self.project.find_job(selection['source']['job_uid']))
            content += [dict(type='text', text=json.dumps(selection)),
                dict(type='image_url', image_url=dict(url=context['contact_sheet']['data_url']))]
        def validate(response):
            diameter = response.get('diameter')
            if type(diameter) not in (int, float) or not math.isfinite(diameter) or not 20 <= diameter <= 600 or not response.get('reason'):
                raise ValueError('Finite whole-particle diameter 20..600 A and reason required')
            return response
        response = self.model('template_diameter', [dict(role='system', content=POLICY),
            dict(role='user', content=content)], 'gpt-6.1-sol', validate)
        self.state['template_parameters'] = dict(diameter=response['diameter'])
        self.persist()
        return self.state['template_parameters']
