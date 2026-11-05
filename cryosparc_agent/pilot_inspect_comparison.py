"""One matched Inspect comparison, followed by frozen-threshold sample expansion."""
import copy
import json
import math

from pilot_reclassification_v2 import ObjectivePilotExperiment
from pilot_objective_gate import evaluate_gate, load_class_images


def validate_options(response):
    options = response.get('options')
    if not isinstance(options, list) or len(options) != 3:
        raise ValueError('Provide exactly three raw-score threshold alternatives')
    expected = ['permissive', 'balanced', 'strict']
    for option, label in zip(options, expected):
        if option.get('name') != label or not isinstance(option.get('reason'), str) or not option['reason'].strip():
            raise ValueError('Ordered named alternatives and evidence-based reasons required')
        p = option.get('parameters', {})
        if set(p) != {'ncc_score_thresh', 'lpower_thresh_min', 'lpower_thresh_max'}:
            raise ValueError('Only three explicit raw-score bounds are allowed')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in p.values()):
            raise ValueError('Finite numeric thresholds required')
        if p['lpower_thresh_min'] >= p['lpower_thresh_max']:
            raise ValueError('Power bounds are reversed')
    for a, b in zip(options, options[1:]):
        p, q = a['parameters'], b['parameters']
        if not (p['ncc_score_thresh'] <= q['ncc_score_thresh'] and
                p['lpower_thresh_min'] <= q['lpower_thresh_min'] and
                p['lpower_thresh_max'] >= q['lpower_thresh_max'] and p != q):
            raise ValueError('Options must be distinct nested permissive/balanced/strict windows')
    return response


def comparison_rank(row, min_particles):
    """Relative route choice is not permission to promote poor classes."""
    if not row.get('scoring_available'):
        return None
    gate = row['gate']
    mean = row.get('weighted_score')
    if type(mean) not in (int, float) or not math.isfinite(mean):
        return None
    return (bool(gate.get('passed')), row['particles'] >= min_particles,
            gate.get('supported_group_count', 0), gate.get('qualified_particles', 0),
            -mean, row['particles'])


class InspectComparisonPilot(ObjectivePilotExperiment):
    scales = (50, 100, 200)

    def __init__(self, root):
        super().__init__(root)
        if self.manifest.get('inspect_comparison') != {'micrographs': 50, 'branches': 3, 'freeze_upstream': True}:
            raise ValueError('Prepare a separate matched Inspect-comparison experiment')

    def stage(self, slug, stage, visual):
        frozen = self.state.get('inspect_comparison', {}).get('frozen_parameters')
        if frozen and slug.startswith(('n50_', 'n100_', 'n200_')):
            key = {'blob_picker_diameter': 'picker', 'inspect_picks_threshold': 'inspect',
                   'extract_box_size': 'extract'}[stage]
            return copy.deepcopy(frozen[key])
        return super().stage(slug, stage, visual)

    def classify_parameters(self, slug, particles, physical_box_A, previous=None):
        comparison = self.state.get('inspect_comparison', {})
        if slug == 'n50_class_parameters' and comparison.get('classification'):
            return copy.deepcopy(comparison['classification'])
        return super().classify_parameters(slug, particles, physical_box_A, previous)

    def report(self):
        super().report()
        path = self.root/'REPORT.md'
        text = path.read_text().replace(
            'Every pilot re-decides Picker, Inspect, Extract and both Class 2D parameters. Model subset selection does not control promotion.',
            'One matched Inspect comparison at 50 micrographs; the winner thresholds, picker and extraction box '
            'are frozen across 50/100/200-micrograph trials. Initial 50-micrograph classes are reused. '
            'Model subset selection does not control promotion.')
        path.write_text(text)

    def compare_once(self):
        from vision_inputs import (build_micrograph_preview_visual_context,
                                   build_pick_inspection_visual_context, build_extract_box_visual_context)
        from upstream_evidence import metadata
        from pilot_reclassification_experiment import SHARED
        c = self.state.setdefault('inspect_comparison', {})
        if c.get('finished'):
            return
        preview = self.service('comparison micrographs', lambda: build_micrograph_preview_visual_context(
            self.project.uid, SHARED[-1], max_micrographs=3))
        if 'picker_parameters' not in c:
            c['picker_parameters'] = self.stage('compare_picker', 'blob_picker_diameter', preview)
        picker_params = c['picker_parameters']
        self.persist()
        picker = self.submit('compare_blob', 'blob_picker_gpu', {'micrographs': (SHARED[-1], 'exposures')},
                             dict(picker_params, num_process=50))
        if not picker.outputs['particles'].num_items:
            c.update(finished=True, failure='Shared Blob output contains no particles')
            self.persist()
            return
        visual = self.service('comparison score evidence', lambda: build_pick_inspection_visual_context(
            self.project.uid, picker.uid, max_micrographs=3))
        if 'options' not in c:
            content = [dict(type='text', text=(
                'Design ONE matched Inspect Picks experiment on the SAME 50-micrograph Blob output. '
                'A previous failure is not evidence that stricter filtering is needed: limited '
                'sample size can prevent stable classification. Include a genuinely permissive '
                'alternative retaining plausible weak-signal particles, a balanced alternative, '
                'and a stricter alternative. Use the raw NCC/Power units, not percentiles. '
                'The windows must be nested: permissive NCC/lower Power bounds <= balanced <= '
                'strict; permissive upper Power bound >= balanced >= strict. Bounds must differ. '
                'Use overlays and scatter as evidence, not assumed quality labels. '
                'Return JSON {"options":[{"name":"permissive","parameters":'
                '{"ncc_score_thresh":number,"lpower_thresh_min":number,"lpower_thresh_max":number},'
                '"reason":"evidence"}, {same fields with name balanced}, {same fields with name strict}]}. '
                'This is a relative comparison, not a full-data promotion verdict.\n' +
                json.dumps(metadata(visual))))]
            for image in [visual.get('contact_sheet') or {}, visual.get('score_scatter') or {},
                          *visual.get('evidence_images', [])]:
                if image.get('data_url'):
                    content.append(dict(type='image_url', image_url=dict(url=image['data_url'])))
            c['options'] = self.model('compare_threshold_options', [
                dict(role='system', content='Design a controlled cryo-EM threshold comparison; return JSON only.'),
                dict(role='user', content=content)], 'openai/gpt-6.1-sol', validate_options)['options']
            self.persist()
        sources = dict(micrographs=(picker.uid, 'micrographs'), particles=(picker.uid, 'particles'))
        for option in c['options']:
            self.submit('compare_'+option['name']+'_inspect', 'inspect_picks_v2', sources,
                        dict(option['parameters'], calibrate_ncc=False, calibrate_pow=False, do_auto_cluster=False), wait=False)
        inspected = {}
        for option in c['options']:
            name = option['name']
            inspected[name] = self.submit('compare_'+name+'_inspect', 'inspect_picks_v2', sources,
                dict(option['parameters'], calibrate_ncc=False, calibrate_pow=False, do_auto_cluster=False))
        if 'extract_parameters' not in c:
            # Use the widest population to choose a shared, not branch-specific box.
            visual = self.service('comparison box evidence', lambda: build_extract_box_visual_context(
                self.project.uid, picker.uid, particle_job_uid=inspected['permissive'].uid, max_examples=3))
            c['extract_parameters'] = self.stage('compare_extract', 'extract_box_size', visual)
            self.persist()
        box = c['extract_parameters']['box_size_pix']
        extracted = {}
        for name, job in inspected.items():
            if job.outputs['particles'].num_items:
                self.submit('compare_'+name+'_extract', 'extract_micrographs_multi',
                    dict(micrographs=(job.uid, 'micrographs'), particles=(job.uid, 'particles')),
                    dict(box_size_pix=box, num_extract=50, compute_num_gpus=1), wait=False)
        for name, job in inspected.items():
            if job.outputs['particles'].num_items:
                extracted[name] = self.submit('compare_'+name+'_extract', 'extract_micrographs_multi',
                    dict(micrographs=(job.uid, 'micrographs'), particles=(job.uid, 'particles')),
                    dict(box_size_pix=box, num_extract=50, compute_num_gpus=1))
        eligible = {name: job for name, job in extracted.items() if job.outputs['particles'].num_items >= 100}
        c.setdefault('rows', {})
        if not eligible:
            c.update(finished=True, failure='No branch has 100 extracted particles')
            self.persist()
            return
        if 'classification' not in c:
            c['classification'] = self.classify_parameters('compare_class_parameters',
                min(j.outputs['particles'].num_items for j in eligible.values()), box*.856)
            self.persist()
        for name, job in eligible.items():
            self.submit('compare_'+name+'_class', 'class_2D_new', {'particles': (job.uid, 'particles')},
                        dict(c['classification'], random_seed=26026, compute_num_gpus=1), wait=False)
        for name, job in eligible.items():
            classified = self.submit('compare_'+name+'_class', 'class_2D_new', {'particles': (job.uid, 'particles')},
                dict(c['classification'], random_seed=26026, compute_num_gpus=1))
            context = self.classes('compare_'+name+'_before', classified)
            images = self.service('comparison class images', lambda: load_class_images(self.project, classified.uid))
            gate = evaluate_gate(context, images, self.policy)
            c['rows'][name] = dict(class_job=classified.uid, particles=job.outputs['particles'].num_items,
                scoring_available=context['score_summary']['available'] and gate['available'],
                weighted_score=context['score_summary']['particle_weighted_mean'], gate=gate)
            self.persist()
        ranked = [(comparison_rank(row, self.policy.min_particles), name) for name, row in c['rows'].items()]
        ranked = [(rank, name) for rank, name in ranked if rank is not None]
        if not ranked:
            raise RuntimeError('Comparison scores unavailable; no evidence-based threshold winner')
        _, winner = max(ranked, key=lambda item: (item[0], {'balanced': 2, 'permissive': 1, 'strict': 0}[item[1]]))
        option = next(o for o in c['options'] if o['name'] == winner)
        c.update(winner=winner, finished=True, frozen_parameters=dict(picker=picker_params,
            inspect=option['parameters'], extract=c['extract_parameters']),
            selection_rule='Gate pass, sample floor, supported shape groups, qualified count, lower weighted CryoSift, count; relative winner is not promotion')
        for stage, source in [('blob', 'compare_blob'), ('inspect', 'compare_'+winner+'_inspect'),
                              ('extract', 'compare_'+winner+'_extract'), ('class', 'compare_'+winner+'_class')]:
            self.state['steps']['n50_'+stage] = copy.deepcopy(self.state['steps'][source])
        self.write(self.root/'threshold_comparison.json', c)
        self.persist()

    def run(self):
        if self.state.get('status') in {'pilot_usable_for_promotion', 'pilot_exhausted_no_validated_route'}:
            return
        self.compare_once()
        if not self.state['inspect_comparison'].get('winner'):
            self.state['status'] = 'pilot_exhausted_no_validated_route'
            self.persist()
            return
        super().run()
