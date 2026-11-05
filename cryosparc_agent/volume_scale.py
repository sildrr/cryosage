"""Estimate Template Picker diameter from a completed refinement mask."""
from __future__ import annotations

import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from cryosparc_client import cryosparc_client


MASK_THRESHOLDS = (0.25, 0.50, 0.75)


def measure_mask_extent(mask: Any, pixel_size_A: float) -> dict[str, Any]:
    """Return a robust physical extent from a soft CryoSPARC refinement mask."""
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D refinement mask, got shape {array.shape}.")
    if not math.isfinite(pixel_size_A) or pixel_size_A <= 0:
        raise ValueError("Mask pixel size must be a positive finite number.")

    measurements = []
    for threshold in MASK_THRESHOLDS:
        coordinates = np.where(array >= threshold)
        if not coordinates[0].size:
            continue
        spans_pix = [int(axis.max() - axis.min() + 1) for axis in coordinates]
        spans_A = [span * pixel_size_A for span in spans_pix]
        measurements.append({
            "threshold": threshold,
            "spans_pix_zyx": spans_pix,
            "spans_A_zyx": spans_A,
            "max_projected_diameter_A": max(spans_A),
        })
    if not measurements:
        raise ValueError("No nonempty mask region was found at standard thresholds.")

    diameters = [item["max_projected_diameter_A"] for item in measurements]
    raw_diameter_A = float(median(diameters))
    # The mask bounds the reconstructed density. Add a small picking margin, then
    # round to a stable value instead of exposing voxel-scale noise to the model.
    recommended_A = int(round((raw_diameter_A * 1.05) / 10.0) * 10)
    spread_A = max(diameters) - min(diameters)
    return {
        "success": True,
        "measurement_method": "refinement_mask_bbox_median_with_5_percent_margin",
        "mask_threshold_measurements": measurements,
        "raw_mask_diameter_A": raw_diameter_A,
        "recommended_particle_diameter_A": recommended_A,
        "threshold_spread_A": spread_A,
        "confidence": "high" if spread_A <= raw_diameter_A * 0.10 else "moderate",
        "limitations": [
            "The estimate uses the reconstructed mask extent, not the volume box size.",
            "Use an observed diameter unless direct picking evidence supports an override.",
        ],
    }


def estimate_template_picker_volume_scale(
    project_uid: str,
    template_job_uid: str,
) -> dict[str, Any]:
    """Trace Create Templates to its source refine job and measure ``mask_refine``."""
    try:
        project = cryosparc_client().find_project(project_uid)
        template_job = project.find_job(template_job_uid)
        if template_job.type == 'select_2D':
            return dict(success=False, applicable=False, template_job_uid=template_job_uid,
                        reason='2D-class templates have no source refinement mask. '
                               'Estimate physical diameter from class images and micrograph scale; '
                               'the upstream Blob minimum is not a diameter measurement.')
        parent_uids = list(template_job.doc.get("parents") or [])
        if not parent_uids:
            raise ValueError("Create Templates has no traceable source volume job.")
        source_job = project.find_job(str(parent_uids[0]))
        dataset = source_job.load_output("volume")
        paths = list(dataset["mask_refine/path"])
        pixel_sizes = list(dataset["mask_refine/psize_A"])
        if not paths or not pixel_sizes:
            raise ValueError("Source refinement does not expose mask_refine output metadata.")
        _, mask = source_job.download_mrc(Path(str(paths[0])).name)
        result = measure_mask_extent(mask, float(pixel_sizes[0]))
        result.update({
            "project_uid": project_uid,
            "template_job_uid": template_job_uid,
            "source_refine_job_uid": str(parent_uids[0]),
            "source_refine_job_type": str(source_job.type),
            "mask_pixel_size_A": float(pixel_sizes[0]),
        })
        return result
    except Exception as exc:
        return {
            "success": False,
            "template_job_uid": template_job_uid,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
