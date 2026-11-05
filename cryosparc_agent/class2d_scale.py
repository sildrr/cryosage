"""Estimate particle diameter and extraction box size from Class 2D evidence."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from cryosparc_client import cryosparc_client


DEFAULT_CTF_FIT_MAX_A = 4.0
DEFAULT_BOX_SIZE_MAX_PIX = 720
BOX_SIZE_QUANTUM_PIX = 32
CTF_FIT_PEAK_BIN_WIDTH_A = 0.1


def build_class2d_scale_context(
    project_uid: str,
    job_uid: str,
    max_classes: int = 50,
    target_resolution_A: float | None = None,
    ctf_fit_max_A: float = DEFAULT_CTF_FIT_MAX_A,
    max_box_size_pix: int = DEFAULT_BOX_SIZE_MAX_PIX,
) -> dict[str, Any]:
    """Measure radial class-average diameters and estimate a CTF-aware box."""
    project = cryosparc_client().find_project(project_uid)
    job = project.find_job(job_uid)
    averages = job.load_output("class_averages")
    paths = list(averages["blob/path"])
    if not paths:
        raise ValueError(f"Job {job_uid} has no class average images.")
    _, stack = job.download_mrc(str(paths[0]).split("/")[-1])
    stack = np.asarray(stack)
    if stack.ndim == 2:
        stack = stack[None, ...]
    count = min(len(paths), len(stack), max(1, int(max_classes)))
    pixel_size_A = _pixel_size(averages)
    class_counts = _class_particle_counts(job, count)
    measurements = [
        _measure_radial_class(class_id, stack[class_id], pixel_size_A, class_counts[class_id])
        for class_id in range(count)
    ]
    valid = [item for item in measurements if item["valid"]]
    diameters_5 = [item["diameter_5pct_A"] for item in valid]
    diameters_10 = [item["diameter_10pct_A"] for item in valid]
    weights = [item["particle_count"] for item in valid]
    mean_5 = _weighted_mean(diameters_5, weights)
    mean_10 = _weighted_mean(diameters_10, weights)
    recommended_diameter = (
        (mean_5 + mean_10) / 2.0
        if mean_5 is not None and mean_10 is not None
        else None
    )

    ctf = _ctf_context(project, job, ctf_fit_max_A)
    voltage = ctf.get("accelerating_voltage_kv")
    wavelength = electron_wavelength_A(voltage) if voltage else None
    ctf_term_A = None
    raw_box_A = None
    raw_box_pix = None
    recommended_box_pix = None
    criterion_ratio = None
    d_ctf_peak = _positive_float(ctf.get("ctf_fit_resolution_peak_A"))
    if recommended_diameter and d_ctf_peak and wavelength and ctf.get("defocus_median_plus_1sigma_A"):
        delta_f = float(ctf["defocus_median_plus_1sigma_A"])
        ctf_term_A = 2.0 * wavelength * delta_f / d_ctf_peak
        raw_box_A = recommended_diameter + ctf_term_A
        pixel_size = ctf.get("micrograph_pixel_size_A")
        if pixel_size:
            raw_box_pix = raw_box_A / float(pixel_size)
            recommended_box_pix = round_box_size_up(raw_box_pix, max_box_size_pix)
        criterion_ratio = ctf_term_A / (2.5 * recommended_diameter)

    shape = list(stack[0].shape[-2:]) if count else []
    weighted_valid_fraction = (
        sum(weights) / max(1, sum(class_counts[:count]))
        if class_counts else 0.0
    )
    estimate_usable = (
        len(valid) >= 3
        and weighted_valid_fraction >= 0.20
        and recommended_diameter is not None
    )
    box_usable = (
        estimate_usable
        and recommended_box_pix is not None
        and ctf.get("quality_filter_count", 0) >= 10
        and criterion_ratio is not None
        and criterion_ratio <= 1.0
    )
    return {
        "kind": "class2d_scale",
        "source": {"project_uid": project_uid, "job_uid": job_uid},
        "measurement_method": "particle_count_weighted_annular_radial_profile",
        "thresholds_of_peak": [0.05, 0.10],
        "raw_image_shape_pix": shape,
        "class_average_pixel_size_A": pixel_size_A,
        "class_measurements": measurements,
        "aggregate": {
            "valid_class_count": len(valid),
            "measured_class_count": len(measurements),
            "particle_count_weighted_diameter_5pct_A": mean_5,
            "particle_count_weighted_diameter_10pct_A": mean_10,
            "recommended_particle_diameter_A": recommended_diameter,
            "particle_diameter_range_A": (
                [mean_10, mean_5]
                if mean_5 is not None and mean_10 is not None
                else None
            ),
            "usable_for_picker": estimate_usable,
            "valid_particle_fraction": weighted_valid_fraction,
        },
        "ctf_evidence": ctf,
        "box_size_recommendation": {
            "d_source": "ctf_fit_resolution_distribution_peak",
            "ctf_fit_resolution_peak_A": d_ctf_peak,
            "legacy_target_resolution_A_ignored": _positive_float(target_resolution_A),
            "electron_wavelength_A": wavelength,
            "ctf_delocalization_term_2lambda_deltaF_over_d_A": ctf_term_A,
            "diameter_plus_ctf_term_A": raw_box_A,
            "unrounded_box_size_pix": raw_box_pix,
            "recommended_box_size_pix": recommended_box_pix,
            "max_box_size_pix": max_box_size_pix,
            "box_size_capped": bool(raw_box_pix and raw_box_pix > max_box_size_pix),
            "ctf_term_over_2_5D": criterion_ratio,
            "within_empirical_2_5D_bound": (
                criterion_ratio <= 1.0 if criterion_ratio is not None else None
            ),
            "usable_for_extraction": box_usable,
            "rounding_quantum_pix": BOX_SIZE_QUANTUM_PIX,
        },
        "limitations": [
            "The diameter is an effective 2D class-average footprint, not an atomic or biological boundary.",
            "Only classes with a measurable centered radial profile and sufficient particle counts contribute.",
            "The 4 A CTF-fit cutoff is an operational default and should be checked for each dataset.",
            "The CTF-fit peak is estimated from the complete upstream Patch CTF exposure set, not a sampled pilot subset.",
            "The optional target_resolution_A argument is retained for API compatibility but is not used as d.",
            "The CTF-aware box is a conservative estimate; validate it against extraction and reconstruction results.",
        ],
    }


def electron_wavelength_A(accelerating_voltage_kv: float) -> float:
    """Relativistic electron wavelength in Angstrom for voltage in kV."""
    voltage_v = float(accelerating_voltage_kv) * 1000.0
    if not math.isfinite(voltage_v) or voltage_v <= 0:
        raise ValueError("accelerating voltage must be a positive finite value")
    return 12.2639 / math.sqrt(voltage_v * (1.0 + 0.97845e-6 * voltage_v))


def round_box_size_up(box_size_pix: float, maximum: int = DEFAULT_BOX_SIZE_MAX_PIX) -> int:
    """Round up to a 32-pixel grid, respecting the configured hard maximum."""
    if not math.isfinite(float(box_size_pix)) or box_size_pix <= 0:
        raise ValueError("box size must be positive and finite")
    rounded = int(math.ceil(float(box_size_pix) / BOX_SIZE_QUANTUM_PIX) * BOX_SIZE_QUANTUM_PIX)
    return min(rounded, int(maximum))


def _measure_radial_class(
    class_id: int, image: Any, pixel_size_A: float, particle_count: int,
) -> dict[str, Any]:
    array = np.asarray(image, dtype=np.float32)
    base = {"class_id": class_id, "particle_count": int(particle_count)}
    if array.ndim != 2 or min(array.shape) < 32:
        return {**base, "valid": False, "reason": "invalid_image_shape"}

    height, width = array.shape
    yy, xx = np.indices(array.shape)
    radii = np.floor(np.hypot(xx - (width - 1) / 2.0, yy - (height - 1) / 2.0)).astype(np.int32)
    max_radius = min(height, width) // 2 - 4
    radial_counts = np.bincount(radii.ravel(), minlength=max_radius + 1)
    radial_sums = np.bincount(radii.ravel(), weights=array.ravel(), minlength=max_radius + 1)
    profile = radial_sums[:max_radius + 1] / np.maximum(radial_counts[:max_radius + 1], 1)
    outer_start = max(2, int(min(height, width) * 0.34))
    outer_end = min(max_radius, int(min(height, width) * 0.47))
    if outer_end - outer_start < 4:
        return {**base, "valid": False, "reason": "insufficient_background_annulus"}
    background = float(np.median(profile[outer_start:outer_end]))
    inner_end = max(3, int(min(height, width) * 0.08))
    polarity = 1.0 if float(np.mean(profile[1:inner_end])) >= background else -1.0
    signal = polarity * (profile - background)
    # A short radial smoother suppresses isolated noise rings without erasing the edge.
    smoothed = np.convolve(signal, np.array([0.25, 0.5, 0.25]), mode="same")
    peak_end = max(4, min(max_radius // 2, int(min(height, width) * 0.25)))
    peak_radius = int(np.argmax(smoothed[1:peak_end])) + 1
    peak = float(smoothed[peak_radius])
    if not math.isfinite(peak) or peak <= 0:
        return {**base, "valid": False, "reason": "no_positive_central_radial_signal"}
    if particle_count < 20:
        return {**base, "valid": False, "reason": "too_few_particles", "radial_peak": peak}

    diameters: dict[str, float] = {}
    for threshold in (0.05, 0.10):
        cutoff = peak * threshold
        crossing = _sustained_threshold_crossing(smoothed, peak_radius, cutoff, max_radius)
        if crossing is None:
            return {**base, "valid": False, "reason": "signal_reaches_box_edge", "radial_peak": peak}
        diameters[f"diameter_{int(threshold * 100)}pct_A"] = 2.0 * (crossing + 0.5) * pixel_size_A

    diameter_5 = diameters["diameter_5pct_A"]
    diameter_10 = diameters["diameter_10pct_A"]
    if diameter_5 < diameter_10:
        return {**base, "valid": False, "reason": "nonmonotonic_threshold_estimate", **diameters}
    box_width_A = min(height, width) * pixel_size_A
    if diameter_5 > 0.70 * box_width_A:
        return {**base, "valid": False, "reason": "radial_signal_too_close_to_box_edge", **diameters}
    return {
        **base,
        "valid": True,
        **diameters,
        "radial_peak": peak,
        "background_level": background,
        "signal_polarity": int(polarity),
        "touches_box_edge": False,
    }


def _sustained_threshold_crossing(
    profile: np.ndarray, peak_radius: int, threshold: float, max_radius: int,
) -> int | None:
    """Return the last above-threshold radius before three sustained below bins."""
    below_run = 0
    last_above = None
    for radius in range(peak_radius, max_radius + 1):
        if profile[radius] >= threshold:
            last_above = radius
            below_run = 0
        else:
            below_run += 1
            if below_run >= 3:
                return last_above
    return None


def _class_particle_counts(job: Any, count: int) -> list[int]:
    try:
        particles = job.load_output("particles", slots=["alignments2D"])
        class_ids = np.asarray(particles["alignments2D/class"], dtype=np.int64)
        valid = class_ids[(class_ids >= 0) & (class_ids < count)]
        return np.bincount(valid, minlength=count).astype(int).tolist()
    except Exception:
        # Some older CryoSPARC jobs do not expose class assignments in this output.
        return [1] * count


def _ctf_context(project: Any, class2d_job: Any, fit_max_A: float) -> dict[str, Any]:
    exposures, source_uid = _find_ctf_exposures(project, class2d_job)
    if exposures is None:
        return {"available": False, "reason": "upstream_ctf_exposures_unavailable"}
    keys = set(exposures.keys())
    df1_key = "ctf/df1_A"
    df2_key = "ctf/df2_A"
    fit_key = "ctf_stats/ctf_fit_to_A" if "ctf_stats/ctf_fit_to_A" in keys else "ctf/ctf_fit_to_A"
    if not {df1_key, df2_key, fit_key}.issubset(keys):
        return {"available": False, "reason": "required_ctf_fields_unavailable"}
    df1 = np.asarray(exposures[df1_key], dtype=float)
    df2 = np.asarray(exposures[df2_key], dtype=float)
    fit = np.asarray(exposures[fit_key], dtype=float)
    defocus = np.abs((df1 + df2) / 2.0)
    mask = np.isfinite(defocus) & (defocus > 0) & np.isfinite(fit) & (fit > 0) & (fit <= fit_max_A)
    filtered = defocus[mask]
    if len(filtered) < 10:
        return {
            "available": False,
            "reason": "fewer_than_10_micrographs_pass_ctf_quality_filter",
            "quality_filter_count": int(len(filtered)),
            "ctf_fit_max_A": float(fit_max_A),
        }
    median = float(np.median(filtered))
    sigma = float(np.std(filtered, ddof=0))
    fit_values = fit[mask]
    fit_peak = _histogram_peak(fit_values, fit_max_A, CTF_FIT_PEAK_BIN_WIDTH_A)
    psize_key = "micrograph_blob/psize_A"
    psize = None
    if psize_key in keys:
        values = np.asarray(exposures[psize_key], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        psize = float(np.median(values)) if len(values) else None
    voltage_key = "ctf/accel_kv"
    voltage = None
    if voltage_key in keys:
        values = np.asarray(exposures[voltage_key], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        voltage = float(np.median(values)) if len(values) else None
    return {
        "available": True,
        "source_job_uid": source_uid,
        "source_dataset_count": int(len(defocus)),
        "micrograph_count": int(len(defocus)),
        "quality_filter_count": int(len(filtered)),
        "ctf_fit_field": fit_key,
        "ctf_fit_max_A": float(fit_max_A),
        "ctf_fit_resolution_peak_A": fit_peak,
        "ctf_fit_peak_method": "mode_of_histogram",
        "ctf_fit_peak_bin_width_A": CTF_FIT_PEAK_BIN_WIDTH_A,
        "ctf_fit_peak_sample_count": int(len(fit_values)),
        "defocus_definition": "absolute mean of CTF principal-axis defocus values: abs((df1_A + df2_A) / 2)",
        "defocus_median_A": median,
        "defocus_sigma_A": sigma,
        "defocus_median_plus_1sigma_A": median + sigma,
        "defocus_p90_A": float(np.percentile(filtered, 90)),
        "micrograph_pixel_size_A": psize,
        "accelerating_voltage_kv": voltage,
    }


def _find_ctf_exposures(project: Any, start_job: Any) -> tuple[Any | None, str | None]:
    queue = [start_job]
    visited: set[str] = set()
    datasets: list[tuple[int, int, str, Any]] = []
    while queue:
        job = queue.pop(0)
        uid = str(getattr(job, "uid", ""))
        if not uid or uid in visited:
            continue
        visited.add(uid)
        for name, input_group in (getattr(job, "inputs", {}) or {}).items():
            for connection in getattr(input_group, "connections", []) or []:
                parent_uid = str(getattr(connection, "job_uid", ""))
                output_name = str(getattr(connection, "output", ""))
                if not parent_uid:
                    continue
                try:
                    parent = project.find_job(parent_uid)
                    if output_name in {"exposures", "micrographs"}:
                        dataset = parent.load_output(output_name)
                        fields = set(dataset.keys())
                        if {"ctf/df1_A", "ctf/df2_A"}.issubset(fields):
                            values = dataset["ctf/df1_A"]
                            count = len(values)
                            is_patch_ctf = "patch_ctf" in str(getattr(parent, "type", "")).lower()
                            datasets.append((int(is_patch_ctf), count, parent_uid, dataset))
                    queue.append(parent)
                except Exception:
                    continue
    if not datasets:
        return None, None
    _, _, uid, dataset = max(datasets, key=lambda item: (item[0], item[1]))
    return dataset, uid


def _histogram_peak(values: Any, maximum_A: float, bin_width_A: float) -> float | None:
    """Return the center of the most populated CTF-fit-resolution histogram bin."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data) & (data > 0) & (data <= maximum_A)]
    if not len(data):
        return None
    edges = np.arange(0.0, maximum_A + bin_width_A, bin_width_A)
    counts, edges = np.histogram(data, bins=edges)
    if not len(counts) or int(counts.max()) == 0:
        return None
    index = int(np.argmax(counts))
    return float((edges[index] + edges[index + 1]) / 2.0)


def _pixel_size(averages: Any) -> float:
    values = list(averages.get("blob/psize_A", []))
    finite = [float(value) for value in values if _finite_positive(value)]
    if not finite:
        raise ValueError("Class average pixel size is unavailable.")
    return float(np.median(finite))


def _weighted_mean(values: list[float], weights: list[int]) -> float | None:
    if not values or len(values) != len(weights) or sum(weights) <= 0:
        return None
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def _positive_float(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def _finite_positive(value: Any) -> bool:
    return _positive_float(value) is not None
