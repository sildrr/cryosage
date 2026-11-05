"""Keep picker evidence and automated Inspect Picks thresholds in raw score units."""
from copy import deepcopy
import math

SCORE_SPACE = 'raw_pick_stats'
CALIBRATION_KEYS = ('calibrate_ncc', 'calibrate_pow')
GUIDANCE = (
    'Inspect Picks evidence and thresholds use raw pick_stats scores. '
    'Keep calibrate_ncc=false and calibrate_pow=false: enabling defocus calibration '
    'changes the threshold coordinate system and invalidates these raw plots and statistics. '
    'Heatmap brightness means pick-count density, not higher NCC/Power or verified quality. '
    'An explicit finite lpower_thresh_max is mandatory. Choose an active upper cutoff '
    'around the main bright density core, retaining most of that core while trimming '
    'the sparse high-Power tail more aggressively. Do not default above the observed '
    'maximum simply to avoid filtering. Brightness alone does not establish quality; '
    'cross-check overlays and report uncertainty.'
)


def raw_parameter_template(template):
    result = deepcopy(template)
    for key in CALIBRATION_KEYS:
        result[key] = dict(type='boolean', default=False, const=False,
                           description='Must be false for raw-score visual evidence.')
    result['lpower_thresh_max'] = dict(type='number',required=True,
        description='Explicit raw Power upper cutoff; retain most of the main density core and trim the high-Power tail.')
    return result


def raw_parameters(parameters):
    result = dict(parameters)
    maximum = result.get('lpower_thresh_max')
    minimum = result.get('lpower_thresh_min')
    if isinstance(maximum,bool) or not isinstance(maximum,(int,float)) or not math.isfinite(maximum):
        raise ValueError('Inspect Picks requires an explicit finite lpower_thresh_max')
    if minimum is not None and (isinstance(minimum,bool) or not isinstance(minimum,(int,float))
            or not math.isfinite(minimum) or minimum >= maximum):
        raise ValueError('Inspect Picks requires lpower_thresh_min < lpower_thresh_max')
    for key in CALIBRATION_KEYS:
        if key in result and result[key] is not False:
            raise ValueError(f'{key} must be false for raw-score Inspect Picks evidence')
        result[key] = False
    return result


def raw_candidate(candidate):
    if candidate.get('job_type') != 'inspect_picks_v2':
        return candidate
    result = deepcopy(candidate)
    result['parameter_template'] = raw_parameter_template(result.get('parameter_template') or {})
    result['default_parameters'] = {**(result.get('default_parameters') or {}),
                                    **dict.fromkeys(CALIBRATION_KEYS, False)}
    # A historical ceiling must not silently satisfy the model's new required choice.
    result['default_parameters'].pop('lpower_thresh_max',None)
    result['score_space'] = SCORE_SPACE
    return result
