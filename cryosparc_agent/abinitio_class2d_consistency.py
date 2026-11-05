"""Produce auditable visual evidence for Ab-initio/Class 2D view consistency."""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from cryosparc_client import cryosparc_client
from vision_inputs import normalize_class_image, vision_cache_dir


OBSERVER_SYSTEM_PROMPT = (
    "You are a cryo-EM WorkflowObserver. Assess only whether the three labelled "
    "Ab-initio projections are supported by the labelled Class 2D tiles selected "
    "upstream. Return exactly one JSON object and no markdown or thinking text."
)


def build_consistency_observer_messages(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a vision-only judgement request from an evidence context."""
    sheet = context.get("contact_sheet") or {}
    image_url = sheet.get("data_url")
    if not isinstance(image_url, str) or not image_url:
        raise ValueError("Ab-initio/Class 2D context has no visual contact sheet.")
    instruction = {
        "task": "abinitio_class2d_view_consistency",
        "source": context.get("source"),
        "protocol": context.get("comparison_protocol"),
        "required_output": {
            "status": "consistent | inconsistent | uncertain",
            "matched_abinitio_views": ["XY projection"],
            "unmatched_abinitio_views": ["XY projection"],
            "evidence": ["short visual observation tied to labels"],
            "suspected_causes": ["truncated_box | off_center_selection | mixed_or_junk_particles | none | uncertain"],
            "recommended_rollback": "new_box_size | reselect_2d | none | uncertain",
            "confidence": 0.0,
        },
        "rules": [
            "Compare each Ab-initio projection only against the green, selected Class 2D tiles.",
            "A ring-like or otherwise distinct Ab-initio projection without a plausible selected Class 2D counterpart is inconsistent.",
            "Do not call a match solely because both images are approximately circular.",
            "Choose new_box_size only when clipping/truncation is visually plausible; choose reselect_2d for off-centre, inconsistent, mixed, or low-quality selected classes.",
            "Do not propose repeating unchanged Ab-initio or refinement.",
        ],
    }
    return [
        {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(instruction, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]


def build_abinitio_class2d_consistency_context(
    project_uid: str,
    abinitio_job_uid: str,
    max_selected_classes: int = 30,
    tile_size: int = 180,
) -> dict[str, Any]:
    """Build a labelled comparison sheet for a completed Ab-initio branch.

    This function deliberately does not assert that two views match. Matching
    3D projections to 2D classes is a visual judgement, so it returns the raw
    evidence and a stable structured provenance record for a vision observer.
    """
    project = cryosparc_client().find_project(project_uid)
    abinitio = project.find_job(abinitio_job_uid)
    if abinitio.type != "homo_abinit":
        raise ValueError(f"{abinitio_job_uid} is {abinitio.type}, expected homo_abinit.")
    select_job = _ancestor_by_type(project, abinitio, "select_2D")
    class_job = _ancestor_by_type(project, abinitio, "class_2D_new")
    if select_job is None or class_job is None:
        raise ValueError("Could not resolve Select 2D and Class 2D ancestors for Ab-initio.")

    selected_ids = _selected_class_ids(select_job)
    volume = _load_volume(abinitio)
    class_stack = _load_class_stack(class_job)
    available = min(len(class_stack), max(selected_ids, default=-1) + 1)
    selected_ids = [class_id for class_id in selected_ids if 0 <= class_id < available]
    if max_selected_classes > 0:
        selected_ids = selected_ids[:max_selected_classes]
    if not selected_ids:
        raise ValueError(f"{select_job.uid} contains no valid selected_templates IDs.")

    image = _compose_comparison(volume, class_stack, selected_ids, tile_size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    data = buffer.getvalue()
    cache = vision_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{project_uid}_{abinitio_job_uid}_abinitio_class2d_consistency.png"
    path.write_bytes(data)
    return {
        "kind": "abinitio_class2d_consistency",
        "success": True,
        "source": {
            "project_uid": project_uid,
            "abinitio_job_uid": abinitio_job_uid,
            "select_2d_job_uid": select_job.uid,
            "class_2d_job_uid": class_job.uid,
        },
        "selected_class_ids": selected_ids,
        "comparison_protocol": {
            "abinitio_views": ["XY projection", "XZ projection", "YZ projection"],
            "class_tiles": "Only classes selected by the upstream Select 2D job, labelled class_id=<integer>.",
            "decision_rule": (
                "For each Ab-initio projection, determine whether at least one selected Class 2D tile "
                "has a plausible corresponding orientation. An unmatched projection is a failure signal, "
                "not a reason to repeat unchanged refinement."
            ),
        },
        "contact_sheet": {
            "mime_type": "image/png",
            "local_path": str(path),
            "data_url": "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
            "tile_size": tile_size,
        },
    }


def build_template_matching_consistency_context(
    project_uid: str,
    abinitio_job_uid: str,
    max_selected_classes: int = 30,
    rotation_step_degrees: int = 15,
    translation_step_pix: int = 4,
    max_translation_pix: int = 8,
    top_k: int = 5,
) -> dict[str, Any]:
    """Quantify agreement between Ab-initio projections and selected Class 2D images.

    This is deliberately a deterministic *evidence* measurement, rather than a
    final quality decision.  It performs masked normalized cross-correlation
    (NCC) after contrast normalization, in-plane rotation, and a small bounded
    translation.  The three canonical volume projections match the projections
    shown to a human observer in ``build_abinitio_class2d_consistency_context``.
    """
    if rotation_step_degrees <= 0 or 360 % rotation_step_degrees:
        raise ValueError("rotation_step_degrees must be a positive divisor of 360.")
    if translation_step_pix <= 0 or max_translation_pix < 0:
        raise ValueError("translation parameters must be non-negative.")
    project = cryosparc_client().find_project(project_uid)
    abinitio = project.find_job(abinitio_job_uid)
    if abinitio.type != "homo_abinit":
        raise ValueError(f"{abinitio_job_uid} is {abinitio.type}, expected homo_abinit.")
    select_job = _ancestor_by_type(project, abinitio, "select_2D")
    class_job = _ancestor_by_type(project, abinitio, "class_2D_new")
    if select_job is None or class_job is None:
        raise ValueError("Could not resolve Select 2D and Class 2D ancestors for Ab-initio.")

    selected_ids = _selected_class_ids(select_job)
    volume = _load_volume(abinitio)
    class_stack = _load_class_stack(class_job)
    selected_ids = [class_id for class_id in selected_ids if 0 <= class_id < len(class_stack)]
    all_selected_ids = list(selected_ids)
    if max_selected_classes > 0:
        selected_ids = selected_ids[:max_selected_classes]
    if not selected_ids:
        raise ValueError(f"{select_job.uid} contains no valid selected_templates IDs.")

    projections = {
        label: _full_projection(volume, axis)
        for label, axis in zip(("XY projection", "XZ projection", "YZ projection"), (0, 1, 2))
    }
    matches = {
        label: _rank_template_matches(
            projection, class_stack, selected_ids,
            rotation_step_degrees=rotation_step_degrees,
            translation_step_pix=translation_step_pix,
            max_translation_pix=max_translation_pix,
            top_k=top_k,
        )
        for label, projection in projections.items()
    }
    # Do not mistake selected classes omitted only by max_selected_classes for
    # excluded classes in the contrast baseline.
    excluded_ids = [class_id for class_id in range(len(class_stack)) if class_id not in all_selected_ids]
    excluded_matches = {
        label: _rank_template_matches(
            projection, class_stack, excluded_ids,
            rotation_step_degrees=rotation_step_degrees,
            translation_step_pix=translation_step_pix,
            max_translation_pix=max_translation_pix,
            top_k=top_k,
        )
        for label, projection in projections.items()
    }
    best_scores = [items[0]["ncc"] for items in matches.values() if items]
    min_best_ncc = float(np.min(best_scores)) if best_scores else None
    selected_minus_excluded = {
        label: round(
            items[0]["ncc"] - excluded_matches[label][0]["ncc"], 6,
        ) if items and excluded_matches[label] else None
        for label, items in matches.items()
    }
    return {
        "kind": "abinitio_class2d_template_matching",
        "success": True,
        "source": {
            "project_uid": project_uid,
            "abinitio_job_uid": abinitio_job_uid,
            "select_2d_job_uid": select_job.uid,
            "class_2d_job_uid": class_job.uid,
        },
        "selected_class_ids": selected_ids,
        "selected_class_ids_total": all_selected_ids,
        "method": {
            "name": "masked_ncc_projection_template_matching_v1",
            "volume_projection": "Full-depth mean projection along each canonical volume axis.",
            "preprocessing": "Per-image percentile clipping, mild Gaussian smoothing, and circular-mask z-score normalization.",
            "search": {
                "in_plane_rotations_degrees": list(range(0, 360, rotation_step_degrees)),
                "translations_pix": list(range(-max_translation_pix, max_translation_pix + 1, translation_step_pix)),
                "score": "masked normalized cross-correlation; sign-invariant maximum is reported.",
            },
            "limitation": (
                "A high NCC measures image agreement only. It is not a validated reconstruction-quality "
                "criterion until calibrated against labelled consistent and inconsistent branches."
            ),
            "selection_contrast": (
                "Each projection is also matched against excluded Class 2D images. A negative selected-minus-"
                "excluded NCC means the excluded set contains the better template for that view, which is direct "
                "evidence that Select 2D may have omitted that orientation."
            ),
        },
        "per_projection_matches": matches,
        "excluded_projection_matches": excluded_matches,
        "summary": {
            "best_ncc_by_projection": {
                label: (items[0]["ncc"] if items else None)
                for label, items in matches.items()
            },
            "mean_best_ncc": float(np.mean(best_scores)) if best_scores else None,
            "min_best_ncc": min_best_ncc,
            "selected_minus_excluded_best_ncc": selected_minus_excluded,
            "min_selected_minus_excluded_best_ncc": min(
                (value for value in selected_minus_excluded.values() if value is not None),
                default=None,
            ),
        },
        "calibration": {
            "reference": "W21 manual labels: J323/J327/J331/J335/J339/J347 aligned; J343/J351/J366 not aligned.",
            "warning_threshold_min_best_ncc": 0.75,
            "w21_calibrated_warning": bool(min_best_ncc is not None and min_best_ncc < 0.75),
            "interpretation": (
                "A warning is evidence for review, not an autonomous failure verdict: score scale may differ "
                "across datasets and reconstruction branches."
            ),
        },
    }


def _ancestor_by_type(project: Any, job: Any, job_type: str) -> Any | None:
    """Walk parent links, preferring the nearest ancestor of the required type."""
    pending = list(job.doc.get("parents") or [])
    seen: set[str] = set()
    while pending:
        uid = str(pending.pop(0))
        if uid in seen:
            continue
        seen.add(uid)
        parent = project.find_job(uid)
        if parent.type == job_type:
            return parent
        pending.extend(parent.doc.get("parents") or [])
    return None


def _selected_class_ids(select_job: Any) -> list[int]:
    raw = str(getattr(select_job.params, "selected_templates", "") or "")
    result = []
    for value in raw.split(","):
        try:
            result.append(int(value.strip()))
        except ValueError:
            continue
    return sorted(set(result))


def _load_volume(job: Any) -> np.ndarray:
    dataset = job.load_output("volume_class_0")
    paths = list(dataset["map/path"])
    if not paths:
        raise ValueError(f"{job.uid} has no volume_class_0 map path.")
    _, volume = job.download_mrc(Path(str(paths[0])).name)
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{job.uid} volume has unexpected shape {array.shape}.")
    return array


def _load_class_stack(job: Any) -> np.ndarray:
    dataset = job.load_output("class_averages")
    paths = list(dataset["blob/path"])
    if not paths:
        raise ValueError(f"{job.uid} has no class averages.")
    _, stack = job.download_mrc(Path(str(paths[0])).name)
    stack = np.asarray(stack, dtype=np.float32)
    return stack[None, ...] if stack.ndim == 2 else stack


def _compose_comparison(volume: np.ndarray, class_stack: np.ndarray, selected_ids: list[int], tile_size: int) -> Image.Image:
    """Render three central-slab volume projections above selected Class 2D tiles."""
    font = ImageFont.load_default()
    label_h = 25
    columns = 5
    projection_labels = ("XY projection", "XZ projection", "YZ projection")
    projections = [_central_projection(volume, axis) for axis in (0, 1, 2)]
    rows = math.ceil(len(selected_ids) / columns)
    width = columns * tile_size
    height = label_h + tile_size + 20 + rows * (label_h + tile_size)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, projection) in enumerate(zip(projection_labels, projections)):
        x = index * tile_size
        draw.rectangle((x, 0, x + tile_size - 1, label_h - 1), fill=(14, 40, 65))
        draw.text((x + 5, 7), label, fill="white", font=font)
        sheet.paste(normalize_class_image(projection, tile_size), (x, label_h))
    draw.text((5, label_h + tile_size + 5), "Selected Class 2D tiles (green = selected upstream)", fill=(25, 107, 36), font=font)
    y0 = label_h + tile_size + 20
    for index, class_id in enumerate(selected_ids):
        x = (index % columns) * tile_size
        y = y0 + (index // columns) * (tile_size + label_h)
        draw.rectangle((x, y, x + tile_size - 1, y + label_h - 1), fill=(25, 107, 36))
        draw.text((x + 5, y + 7), f"class_id={class_id}", fill="white", font=font)
        sheet.paste(normalize_class_image(class_stack[class_id], tile_size), (x, y + label_h))
    return sheet


def _central_projection(volume: np.ndarray, axis: int) -> np.ndarray:
    width = max(3, volume.shape[axis] // 8)
    centre = volume.shape[axis] // 2
    start, stop = max(0, centre - width), min(volume.shape[axis], centre + width)
    return np.mean(np.take(volume, range(start, stop), axis=axis), axis=axis)


def _full_projection(volume: np.ndarray, axis: int) -> np.ndarray:
    """Use a physical full-depth projection, rather than a central slab."""
    return np.mean(volume, axis=axis)


def _rank_template_matches(
    projection: np.ndarray,
    class_stack: np.ndarray,
    selected_ids: list[int],
    *,
    rotation_step_degrees: int,
    translation_step_pix: int,
    max_translation_pix: int,
    top_k: int,
) -> list[dict[str, Any]]:
    target = _preprocess_for_ncc(projection)
    mask = _circular_mask(target.shape)
    rotations = range(0, 360, rotation_step_degrees)
    translations = range(-max_translation_pix, max_translation_pix + 1, translation_step_pix)
    ranked: list[dict[str, Any]] = []
    for class_id in selected_ids:
        template = _preprocess_for_ncc(class_stack[class_id], target.shape)
        best: dict[str, Any] | None = None
        for angle in rotations:
            rotated = _rotate_array(template, angle)
            for shift_y in translations:
                for shift_x in translations:
                    shifted = _translate_with_zeros(rotated, shift_y, shift_x)
                    signed_score = _masked_ncc(target, shifted, mask)
                    candidate = {
                        "class_id": class_id,
                        "ncc": round(abs(signed_score), 6),
                        "signed_ncc": round(signed_score, 6),
                        "rotation_degrees": angle,
                        "translation_pix": {"x": shift_x, "y": shift_y},
                    }
                    if best is None or candidate["ncc"] > best["ncc"]:
                        best = candidate
        if best is not None:
            ranked.append(best)
    return sorted(ranked, key=lambda item: item["ncc"], reverse=True)[:max(1, top_k)]


def _preprocess_for_ncc(image: np.ndarray, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if target_shape is not None and array.shape != target_shape:
        scaled = Image.fromarray(array, mode="F").resize(
            (target_shape[1], target_shape[0]), Image.Resampling.BILINEAR,
        )
        array = np.asarray(scaled, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        high = low + 1.0
    array = np.clip(array, low, high)
    # Mild smoothing makes the score less sensitive to CTF-scale high-frequency noise.
    image8 = Image.fromarray(((array - low) / (high - low) * 255).astype(np.uint8), mode="L")
    return np.asarray(image8.filter(ImageFilter.GaussianBlur(radius=0.7)), dtype=np.float32)


def _circular_mask(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.ogrid[:shape[0], :shape[1]]
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    radius = min(shape) * 0.42
    return ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2)


def _rotate_array(array: np.ndarray, degrees: float) -> np.ndarray:
    image = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(image.rotate(degrees, resample=Image.Resampling.BILINEAR, fillcolor=0.0), dtype=np.float32)


def _translate_with_zeros(array: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    result = np.zeros_like(array)
    source_y = slice(max(0, -shift_y), min(array.shape[0], array.shape[0] - shift_y))
    source_x = slice(max(0, -shift_x), min(array.shape[1], array.shape[1] - shift_x))
    target_y = slice(max(0, shift_y), min(array.shape[0], array.shape[0] + shift_y))
    target_x = slice(max(0, shift_x), min(array.shape[1], array.shape[1] + shift_x))
    result[target_y, target_x] = array[source_y, source_x]
    return result


def _masked_ncc(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    x = left[mask].astype(np.float64)
    y = right[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else 0.0
