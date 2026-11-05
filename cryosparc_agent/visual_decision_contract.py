"""Shared prompt and compact output contract for visual parameter decisions."""

from __future__ import annotations

import json
from typing import Any

try:
    from .cryosift_adapter import CRYOSIFT_SCORE_GUIDANCE
except ImportError:
    from cryosift_adapter import CRYOSIFT_SCORE_GUIDANCE


REFINEMENT_PREFERENCE_GUIDANCE = (
    "Planning preferences, not additional hard gates: when the corrected current FSC "
    "resolution is above target by less than 0.1 A, prefer one modern NU-refine "
    "(nonuniform_refine_new) comparison if it has not already been tried on that "
    "particle/volume branch and the map is plausible. Preserve symmetry and comparable "
    "CTF settings; do not repeat the same refinement on unchanged inputs. Otherwise, "
    "when template creation/repicking is eligible under the existing safety gates, "
    "prefer Create Templates followed by Template Picker over reclassification or "
    "routine reselection. After an unsuccessful one-time NU-refine, reconsider eligible "
    "template repicking from the best valid volume rather than repeating refinement. "
    "Keep the best checkpoint when another branch worsens. These preferences do not "
    "guarantee target attainment; explain evidence-backed deviations. "
    "Also prefer a one-time NU comparison on a plausible Homo baseline when the gap "
    "is larger; the near-target rule is not a restriction on trying NU. Before repeated "
    "selection changes at near-atomic resolution, consider the explicit optics_refinement "
    "candidate for defocus plus global tilt/trefoil/Cs fitting. Preserve all best results. "
)

EXTRACT_BOX_PREFERENCE_GUIDANCE = (
    "Box-size preference, not a hard constraint: estimate the maximum credible particle "
    "diameter D_max_A from representative particles in the uncropped full-field images "
    "and their calibrated scale bars, cross-checking the picker diameter range. Do not "
    "size around an ice/carbon contaminant or blindly use an intentionally broad picker "
    "search upper bound. Prefer a native extraction box spanning 1.5-2 times that "
    "maximum particle diameter: box_size_pix approximately (1.5-2) * D_max_A / "
    "native_pixel_size_A. Use native acquisition sampling, not resized display pixels. "
    "Choose a supported FFT-friendly size in or near that range; when uncertain prefer "
    "its upper end for completeness. State the diameter, pixel size, physical box "
    "width and margin in the reason. For a box-size retry, compare nearby supported "
    "sizes before making a large jump (for example, 320 directly to 512); a large "
    "change remains reasonable if the measured diameter or clear clipping supports it. "
    "Treat radial-profile and CTF-aware estimates as cross-checks, not an automatic "
    "override of full-field size evidence. Explain disagreement and uncertainty rather "
    "than blindly copying a conservative CTF recommendation. If no reliable diameter "
    "is available, acknowledge that and use the available evidence or calibration path. "
    "Completeness takes priority over removing every nearby automatic pick. A few "
    "peripheral neighbors are acceptable; extra candidate centers are not verified "
    "molecules. Consider shrinking for overlap only when many representative boxes "
    "repeatedly contain substantial interference or multiple clearly resolved targets. "
    "Do not shrink solely to minimize background or because noisy crops show no obvious clipping. "
)


STAGE_ACTIONS = {
    "blob_picker_diameter": "blob_picker_gpu",
    "template_picker_diameter": "template_picker_gpu",
    "inspect_picks_threshold": "inspect_picks_v2",
    "extract_box_size": "extract_micrographs_multi",
}

STAGE_INSTRUCTIONS = {
    "blob_picker_diameter": (
        "Choose an exploratory Blob Picker diameter range in Angstroms from the "
        "pre-picking micrographs and acquisition facts. `diameter` is the minimum; "
        "use the separate full-field micrographs compressed to 1024-square canvases, "
        "with calibrated Angstrom scale bars and per-image display/native pixel sizes. "
        "These are uncropped full images, not enlarged local crops. Do not confuse "
        "display pixels with native pixels. Include optional `diameter_max` when an upper bound is justified. Do not "
        "claim the observed historical settings are validated optima."
    ),
    "template_picker_diameter": (
        "Choose a Template Picker particle diameter in Angstroms using the supplied "
        "templates, micrographs, and acquisition facts. Treat historical settings as "
        "weak observations rather than validated optima. Blob Picker diameter is a minimum "
        "search bound, not a measured whole-particle diameter; do not simply copy it. "
        "Check template extent and physical scale. A small diameter also makes the native "
        "dimensionless min_distance allow closely spaced, potentially duplicate picks. "
        + CRYOSIFT_SCORE_GUIDANCE
    ),
    "inspect_picks_threshold": (
        "Use raw pick_stats scores with calibrate_ncc=false and calibrate_pow=false. "
        "Heatmap brightness encodes count density, not score magnitude or particle quality. "
        "Choose explicit Inspect Picks thresholds from the NCC/Power scatter and "
        "density plot, score-colored micrograph overlay, and the score-stratified "
        "examples marked on uncropped, compressed full micrographs with scale bars. "
        "Use marked score examples in their full-image context, not as molecular boundary labels. "
        "Determine whether the dense population represents particles at all; high count "
        "density can instead be background, contamination or multiple hits on one particle. "
        "There is no preferred retention fraction and no obligation to retain the bright core "
        "or all marked score examples. Use their image evidence to discriminate signal, "
        "including substantially stronger filtering or an available picker rollback when justified. "
        "You MUST explicitly choose local_power_max "
        "(lpower_thresh_max), above the lower bound. Prefer an active, more aggressive "
        "upper cutoff based on credible particle examples, not automatically P95 or P99. "
        "Do not set it beyond the observed maximum merely to avoid filtering. "
        "Cross-check overlays because dense or high-Power picks are not automatically good particles. "
        "An overlay can show only a subset of picks: compare overlay_count with pick_count "
        "and the total input count before judging spatial density or repeated hits. "
        "Do not use auto clustering. "
        "Important: CryoSPARC `local_power_min` and `local_power_max` are thresholds "
        "on the raw `pick_stats/power` values, not normalized scores, percentiles, "
        "or a 0-30 scale. Read the numeric `score_statistics.power` range and axis "
        "shown in the evidence, and choose bounds in that same raw scale. A bound "
        "below the observed minimum can reject every pick; check the expected "
        "retained count/fraction before returning the decision."
    ),
    "extract_box_size": (
        "Choose `box_size_pix` using acquisition pixel size, upstream picker diameter "
        "and spatially separated, interior NCC/Power-stratified examples marked on uncropped compressed full micrographs "
        "with scale bars and reference box outlines. MID denotes a moderate joint-score sample; HI-NCC has higher NCC but "
        "moderate Power. CONTROL is an explicit boundary/extreme-score contrast, not a typical or recommended particle; "
        "FALLBACK means a requested score region was unavailable. Scores and these labels do not certify particle quality. "
        "Reference box labels and box_size_pix use native pixels, not displayed pixels. The outlines compare "
        "square sampling windows, not circles or particle diameter; they do not mark the molecular boundary or force a choice. "
        "The displayed examples deliberately avoid image edges and overlapping annotations; this does not remove actual picks "
        "and cannot establish that all picks have good geometry. Judge typical samples first, cross-check HI-NCC, "
        "and use CONTROL only to detect artifacts or uncertainty; do not size every box around a contaminant. "
        "Preserve native sampling; do not Fourier-crop. bin_size_pix, if present, means "
        "Fourier-cropped OUTPUT EDGE LENGTH IN PIXELS, not a binning factor. A value of 2 "
        "means 2x2 output, not two-fold downsampling. Leave it unset/blank or match "
        "box_size_pix to preserve sampling; avoid downsampling during resolution optimization "
        "because it can discard required high-frequency information. "
        + EXTRACT_BOX_PREFERENCE_GUIDANCE
    ),
}

STAGE_PARAMETER_FIELDS = {
    "blob_picker_diameter": ("diameter", "diameter_max"),
    "template_picker_diameter": ("diameter",),
    "inspect_picks_threshold": (
        "ncc_score_min", "local_power_min", "local_power_max",
    ),
    "extract_box_size": ("box_size_pix",),
}


def build_stage_decision_messages(
    stage: str,
    context: dict[str, Any],
    *,
    image_paths: list[str] | None = None,
    image_data_urls: list[str] | None = None,
    include_reason: bool = False,
) -> list[dict[str, Any]]:
    """Build backend-neutral messages; backends only materialize image payloads."""
    if stage not in STAGE_ACTIONS:
        raise ValueError(f"Unsupported visual decision stage: {stage!r}")
    fields = STAGE_PARAMETER_FIELDS[stage]
    contract = {
        "decision": STAGE_ACTIONS[stage],
        "parameters": {field: "number" for field in fields},
    }
    if stage == "blob_picker_diameter":
        contract["parameters"]["diameter_max"] = "optional number"
        if include_reason:
            contract['diameter_confidence'] = 'confident | uncertain'
    if include_reason:
        contract["reason"] = (
            "An auditable decision rationale: identify the observed image/state evidence, "
            "justify the chosen parameter values and units, explain the expected quality "
            "effect and any uncertainty. Use several concrete sentences, not a generic "
            "one-line explanation. Do not invent measurements or describe private reasoning."
        )
    trial_actions = [a for a in context.get('candidate_actions', []) if a.get('diameter_trial')]
    trial_review = any((a.get('diameter_trial') or {}).get('role') in {'retry', 'continue'}
                       for a in trial_actions)
    trial_instruction = ''
    if include_reason and stage == 'blob_picker_diameter':
        trial_instruction = (
            'Report diameter_confidence=uncertain when complete particle boundaries cannot '
            'be distinguished from noise, texture or neighboring densities. In that case '
            'the runner will use a bounded small-sample trial, not full-data picking. '
            'Do not call a rough visual guess a measured distribution. Use confident only '
            'when repeated intact particles support the size estimate. '
        )
    if trial_review:
        contract['action_id'] = 'exact available candidate action_id'
        contract['decision'] = 'job_type of the chosen available candidate'
        contract['parameters'] = 'parameters for the chosen job_type, matching its parameter_template'
        trial_instruction += (
            'A small-sample diameter trial is active. Review its evidence before accepting '
            'the upstream diameter. Choose the offered blob_picker_gpu retry with revised '
            'diameter/diameter_max if picks follow fragments/background or particle size is '
            'implausible. Otherwise choose the offered next pilot step and its parameters. '
            'Include the exact action_id; do not silently return a full-data action. '
        )
    user_text = "\n".join((
        STAGE_INSTRUCTIONS[stage],
        trial_instruction,
        (
            "A previous response was rejected by the parameter validator. Correct the specific "
            "reported error and keep every value inside the configured search_space."
            if context.get("previous_invalid_decisions") else ""
        ),
        "Return exactly one JSON object matching this stage contract (no markdown or extra keys): "
        + json.dumps(contract, ensure_ascii=False, separators=(",", ":")),
        "Decision context: " + json.dumps(context, ensure_ascii=False, separators=(",", ":")),
    ))
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for path in image_paths or []:
        content.append({"type": "image", "path": path})
    for data_url in image_data_urls or []:
        content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})
    return [
        {
            "role": "system",
            "content": (
                "You are a cryo-EM workflow parameter decision model. Use the supplied "
                "evidence and return only the requested JSON object. Values from prior "
                "workflows are weak historical observations, not verified optimal labels."
            ),
        },
        {"role": "user", "content": content},
    ]
