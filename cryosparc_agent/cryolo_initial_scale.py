"""Opt-in measured initial scale, followed by Inspect-only controlled trials."""
import copy
import hashlib
import json
import math
from pathlib import Path


def snapshot_scale(probe_root, dataset, pixel_size_A):
    root = Path(probe_root)
    path = root / str(dataset) / 'summary.json'
    raw = path.read_bytes()
    summary = json.loads(raw)
    inputs = json.loads((root / 'manifest.json').read_text())['datasets'][str(dataset)]['images']
    if summary['dataset'] != str(dataset) or not inputs:
        raise ValueError('crYOLO summary/source dataset mismatch')
    if not all(math.isclose(x['pixel_size_A'], pixel_size_A, rel_tol=1e-5) for x in inputs):
        raise ValueError('crYOLO input pixel size differs from current acquisition')
    stats = summary['statistics']['0.5']
    if stats['count'] <= 0:
        raise ValueError('No qualified crYOLO detections')
    diameter = stats['median_A']
    if type(diameter) not in (int, float) or not math.isfinite(diameter) or diameter <= 0:
        raise ValueError('Invalid representative diameter')
    # Stable upward rounding only; this is not Fourier cropping or an FFT claim.
    box = 16 * math.ceil((2 * diameter / pixel_size_A - 1e-9) / 16)
    result = dict(version='cryolo-initial-scale-v1', dataset=str(dataset),
        pixel_size_A=pixel_size_A, diameter_max_A=diameter, box_size_pix=box,
        formula='ceil((2 * median_max_estimated_axis_A / native_pixel_size_A) / 16) * 16',
        confidence_threshold=.5, prediction_count=stats['count'],
        source_summary=str(path), source_summary_sha256=hashlib.sha256(raw).hexdigest(),
        model_sha256=summary['weights_sha256'], measurement=summary['measurement'],
        minimum_diameter_policy='Model chooses a smaller positive minimum once from initial evidence.',
        unlock_policy='Scientific initial pilot ladder exhaustion only; infrastructure/budget errors do not unlock.')
    validate_scale(result, str(dataset), pixel_size_A)
    return result


def validate_scale(value, dataset, pixel_size_A):
    if value.get('version') != 'cryolo-initial-scale-v1' or value.get('dataset') != str(dataset):
        raise ValueError('Wrong crYOLO initial-scale protocol/dataset')
    d, p, box = value.get('diameter_max_A'), value.get('pixel_size_A'), value.get('box_size_pix')
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (d, p)):
        raise ValueError('Positive finite diameter and pixel size required')
    if not math.isclose(p, pixel_size_A, rel_tol=1e-5):
        raise ValueError('Initial-scale pixel size mismatch')
    expected = 16 * math.ceil((2*d/p - 1e-9)/16)
    if type(box) is not int or not 64 <= box <= 720 or box != expected:
        raise ValueError('Computed initial box must be in 64..720; no silent cap or alternate box')
    return value


class CryoloInitialScaleMixin:
    def initial_scale(self):
        value = self.manifest.get('cryolo_initial_scale')
        if value is None:
            return None
        facts = self.manifest['acquisition_facts']
        return validate_scale(value, self.manifest['dataset'], facts['psize_A'])

    def initial_scale_locked(self):
        return (self.initial_scale() is not None and not self.state.get('initial_pilot_exhausted')
                and not self.state.get('pilot_accepted'))

    def model(self, slug, messages, model, validate):
        if not self.initial_scale_locked():
            return super().model(slug, messages, model, validate)
        prior = self.initial_scale()
        messages = copy.deepcopy(messages)
        messages.insert(0, dict(role='system', content=
            'CONTROLLED INITIAL-SCALE AMENDMENT overrides older free-sizing guidance. '
            'Use the measured crYOLO representative maximum-axis diameter as Blob diameter_max '
            'and the calculated native Extract box exactly. Choose a positive smaller Blob minimum '
            'once from initial visual evidence, then freeze the picker range. Only Inspect NCC/Power '
            'thresholds may be revised during this initial pilot ladder. Failed quality does not '
            'automatically mean thresholds should be stricter. Sample expansion, Class 2D, '
            'reconstruction and objective gates remain unchanged. Do not revise diameter or box '
            'until the controller records scientific exhaustion of this Inspect-only ladder. '
            'A transport failure or Job budget exhaustion is not scientific evidence.\n' + json.dumps(prior)))
        def checked(response):
            result = validate(response)
            if slug == 'pilot_picker_parameters':
                p = result['parameters']
                if (not math.isclose(p['diameter_max'], prior['diameter_max_A'], abs_tol=.01, rel_tol=0)
                        or not 0 < p['diameter'] < p['diameter_max']):
                    raise ValueError('Initial Blob maximum must equal measured diameter within 0.01 A; minimum must be smaller')
            if 'target' in result and result['target'] != 'inspect':
                raise ValueError('Initial scale is locked: only target=inspect is allowed before scientific exhaustion')
            return result
        return super().model(slug, messages, model, checked)

    def stage(self, slug, stage, visual):
        if not self.initial_scale_locked() or stage != 'extract_box_size':
            return super().stage(slug, stage, visual)
        prior = self.initial_scale()
        path = self.root/'decisions'/slug/'decision.json'
        decision = dict(parameters=dict(box_size_pix=prior['box_size_pix']),
            reason='Controlled initial trial: twice the crYOLO representative diameter in native pixels, rounded upward to 16.',
            decision_source='measured_formula_not_model_output', cryolo_initial_scale=prior,
            scale_estimate=dict(diameter_max_A=prior['diameter_max_A'], confidence='uncertain',
                                basis='crYOLO learned bounding extent; independent physical accuracy not guaranteed'))
        if path.exists() and json.loads(path.read_text()) != decision:
            raise ValueError('Existing initial sizing decision differs; use a fresh experiment')
        self.write(path, decision)
        self.write(self.root/'evidence'/(slug+'.json'), visual)
        return decision['parameters']

    def review_parameters(self, slug, stage, visual, current, history):
        if not self.initial_scale_locked() or stage != 'extract':
            return super().review_parameters(slug, stage, visual, current, history)
        if current != dict(box_size_pix=self.initial_scale()['box_size_pix']):
            raise ValueError('Initial Extract geometry must remain fixed')
        response = dict(action='keep', reason='Initial controlled geometry is fixed; evaluate Class 2D objective gate before unlocking scale.',
                        decision_source='controlled_policy_not_model_review')
        self.write(self.root/'decisions'/slug/'decision.json', response)
        return response
