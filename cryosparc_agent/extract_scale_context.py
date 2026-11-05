"""Calibrated scale memory and soft completeness-first extraction guidance."""
import math


SCALE_GUIDANCE = (
    'Completeness has priority over minimizing background or eliminating neighbor picks. '
    'A few nearby automatic picks inside a large extraction window are acceptable; they '
    'may be false positives, duplicates or peripheral neighbors, not separate complete '
    'molecules. Do NOT shrink merely to remove all extra picked centers. Consider shrinking '
    'for neighbors only when many representative boxes repeatedly contain multiple '
    'clearly resolved complete targets or substantial interfering overlap. Explain how '
    'common and severe that is, not just a few example counts. '
    'All Blob diameter/diameter_max values are ANGSTROMS and are search bounds, NOT pixels '
    'and NOT measured molecular extent. A faint central bright core is not the whole particle. '
    'Retain the original full-field diameter estimate and its uncertainty; do not silently '
    'replace it with a smaller estimate derived from a cropped/filtered bright core. '
    'If revising that estimate, cite new calibrated whole-particle evidence and explain '
    'the disagreement. Use box_width_A=box_size_pix*native_pixel_size_A and prefer '
    'box_width_A/D_max_A around 1.5-2, with the generous end when boundaries are uncertain. '
    'This is a soft preference, not a fixed minimum/maximum box gate. Lack of visible '
    'clipping in noisy crops is NOT positive evidence of completeness. '
    'Return scale_estimate={"diameter_max_A":number or null,"confidence":"confident|uncertain",'
    '"basis":"calibrated evidence and comparison to the original estimate"}. '
    'Use null rather than invent a measurement. For a box outside the 1.5-2 preference '
    'supply outside_preference_reason. Always supply diameter_revision_reason when '
    'claiming a smaller molecular extent than the previous explicit estimate. If the '
    'original diameter is only stated in a legacy reason, also provide '
    'diameter_revision_reason explaining whether you retain or revise that original estimate.'
)


def valid_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def nonempty_text(value):
    return isinstance(value,str) and bool(value.strip())


def scale_memory(pixel_size_A, box_size_pix=None, picker=None, original=None, previous_estimates=()):
    if not valid_number(pixel_size_A):
        raise ValueError('Positive calibrated native pixel size required')
    picker, original = picker or {}, original or {}
    initial = original.get('scale_estimate') or {}
    diameter = initial.get('diameter_max_A')
    if not valid_number(diameter):
        diameter = None
    latest = next((p.get('scale_estimate',{}).get('diameter_max_A') for p in reversed(list(previous_estimates))
                   if valid_number(p.get('scale_estimate',{}).get('diameter_max_A'))),None)
    reference = diameter or latest
    width = box_size_pix*pixel_size_A if box_size_pix is not None else None
    original_box = original.get('parameters', {}).get('box_size_pix')
    return dict(native_pixel_size_A=pixel_size_A, current_box_size_pix=box_size_pix,
        current_box_width_A=width, initial_estimated_max_diameter_A=diameter,
        latest_explicit_estimated_max_diameter_A=latest,
        current_box_over_initial_estimated_diameter=width/diameter if width and diameter else None,
        current_box_over_latest_estimated_diameter=width/latest if width and latest else None,
        preferred_box_range_native_pix=[1.5*reference/pixel_size_A,2*reference/pixel_size_A] if reference else None,
        margin_A_each_side=(width-reference)/2 if width and reference else None,
        original_sizing_decision=original,
        original_box_width_A=original_box*pixel_size_A if valid_number(original_box) else None,
        picker_search_bounds=dict(units='Angstroms; NOT pixels and NOT a measured molecular diameter',
            minimum_A=picker.get('diameter'), maximum_A=picker.get('diameter_max'),
            minimum_native_pix=picker['diameter']/pixel_size_A if valid_number(picker.get('diameter')) else None,
            maximum_native_pix=picker['diameter_max']/pixel_size_A if valid_number(picker.get('diameter_max')) else None),
        previous_explicit_scale_estimates=list(previous_estimates),
        legacy_note='If the original numeric estimate is only in its reason, read that verbatim; do not substitute Blob minimum.',
        advice='Prefer physical box width about 1.5-2x credible whole-particle diameter; uncertainty favors completeness.')


def validate_scale_response(response, memory, box_size_pix):
    estimate = response.get('scale_estimate')
    if (not isinstance(estimate,dict) or estimate.get('confidence') not in {'confident','uncertain'} or
            not isinstance(estimate.get('basis'),str) or not estimate['basis'].strip() or
            'diameter_max_A' not in estimate):
        raise ValueError('Explicit calibrated scale_estimate with confidence/basis required')
    diameter = estimate['diameter_max_A']
    if diameter is not None and not valid_number(diameter):
        raise ValueError('Estimated diameter must be positive finite Angstroms or null')
    if type(box_size_pix) is not int or box_size_pix<=0:
        raise ValueError('Native integer box size required for scale calculation')
    width = box_size_pix*memory['native_pixel_size_A']
    ratio = width/diameter if diameter is not None else None
    if ratio is not None and not 1.5<=ratio<=2 and not nonempty_text(response.get('outside_preference_reason')):
        raise ValueError('Explain a box outside the soft 1.5-2x physical-diameter preference')
    prior = memory.get('initial_estimated_max_diameter_A')
    previous = [p.get('scale_estimate',{}).get('diameter_max_A') for p in memory.get('previous_explicit_scale_estimates',[])]
    legacy = memory.get('original_sizing_decision',{}).get('reason') and not valid_number(prior)
    if (legacy and diameter is not None) or any(valid_number(d) and diameter is not None and diameter<d for d in [prior,*previous]):
        if not nonempty_text(response.get('diameter_revision_reason')):
            raise ValueError('Explain why the earlier molecular diameter estimate is being reduced')
    if diameter is None and estimate['confidence']=='confident':
        raise ValueError('A missing diameter cannot be a confident measurement')
    return {**response, 'scale_evaluation':dict(native_pixel_size_A=memory['native_pixel_size_A'],
        box_size_pix=box_size_pix, box_width_A=width, estimated_max_diameter_A=diameter,
        box_over_estimated_diameter=ratio, outside_soft_preference=ratio is not None and not 1.5<=ratio<=2,
        ratio_is_advice_not_hard_gate=True)}


def neighbor_summary(samples, size):
    counts = [s['extra_picked_centers_by_box_px'].get(str(size),0) for s in samples]
    n = len(counts)
    return dict(sampled_boxes=n, boxes_with_any_extra_pick=sum(c>0 for c in counts),
        boxes_with_more_than_two_extra_picks=sum(c>2 for c in counts),
        fraction_with_any_extra_pick=sum(c>0 for c in counts)/n if n else None,
        meaning='Automatic pick centers, not verified molecules. Occasional neighbors are acceptable; '
                'counts alone do not establish frequent multi-particle contamination or justify shrinking.')
