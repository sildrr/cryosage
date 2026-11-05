"""Build compact, class-id-labelled visual inputs from CryoSPARC outputs."""
import base64
import io
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cryosparc_client import cryosparc_client
from reference_projection_metrics import (
    compute_reference_projection_metrics,
    resolve_reference_volume,
)


def vision_cache_dir() -> Path:
    """Return a user-writable cache, with an explicit deployment override."""
    configured = os.getenv("CRYOAGENT_VISION_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / "logs" / "vision_inputs"


def build_class_average_visual_context(
    project_uid: str,
    job_uid: str,
    max_classes: int = 50,
    tile_size: int = 192,
    columns: int = 5,
) -> dict[str, Any]:
    """Return a labelled contact sheet and per-class metadata for vision APIs."""
    cs = cryosparc_client()
    project = cs.find_project(project_uid)
    job = project.find_job(job_uid)
    dataset = job.load_output("class_averages")
    paths = list(dataset["blob/path"])
    indices = list(dataset["blob/idx"])
    resolutions = list(dataset.get("blob/res_A", []))
    if not paths:
        raise ValueError(f"Job {job_uid} has no class average images.")

    image_path = str(paths[0])
    filename = Path(image_path).name
    _, stack = job.download_mrc(filename)
    stack = np.asarray(stack)
    if stack.ndim == 2:
        stack = stack[None, ...]
    available_count = min(len(paths), len(stack))
    # A non-positive limit means attach every class so the model can make the
    # actual Select 2D inclusion decision rather than receiving a sampled subset.
    count = available_count if max_classes <= 0 else min(available_count, max_classes)
    selected_ids = list(range(count))
    rows = math.ceil(count / columns)
    label_height = 26
    sheet = Image.new(
        "RGB", (columns * tile_size, rows * (tile_size + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for class_id in selected_ids:
        image = normalize_class_image(stack[class_id], tile_size)
        x = (class_id % columns) * tile_size
        y = (class_id // columns) * (tile_size + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.rectangle((x, y, x + tile_size - 1, y + label_height - 1), fill="black")
        draw.text((x + 5, y + 6), f"class_id={class_id}", fill="white", font=font)

    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG", optimize=True)
    image_bytes = buffer.getvalue()
    cache_dir = vision_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / f"{project_uid}_{job_uid}_class_averages.png"
    cached_path.write_bytes(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    stats = []
    for class_id in selected_ids:
        item = {"class_id": class_id}
        if class_id < len(resolutions):
            item["resolution_A"] = float(resolutions[class_id])
        stats.append(item)
    return {
        "kind": "class_average",
        "source": {"project_uid": project_uid, "job_uid": job_uid},
        "image_count": count,
        "class_ids": selected_ids,
        "class_statistics": stats,
        "contact_sheet": {
            "mime_type": "image/png",
            "local_path": str(cached_path),
            "data_url": f"data:image/png;base64,{encoded}",
            "columns": columns,
            "tile_size": tile_size,
            "label_format": "class_id=<integer>",
        },
    }


def build_class2d_feedback_context(project_uid: str, job_uid: str) -> dict[str, Any]:
    """Extract per-class particle counts and CryoSPARC resolution fields for reward logging."""
    cs = cryosparc_client()
    project = cs.find_project(project_uid)
    job = project.find_job(job_uid)
    averages = job.load_output("class_averages")
    particles = job.load_output("particles", slots=["alignments2D"])
    class_ids = np.asarray(particles["alignments2D/class"], dtype=np.int64)
    resolutions = np.asarray(averages.get("blob/res_A", []), dtype=np.float64)
    class_count = max(len(resolutions), int(np.max(class_ids)) + 1 if class_ids.size else 0)
    counts = np.bincount(class_ids[class_ids >= 0], minlength=class_count)
    classes = []
    for class_id in range(class_count):
        resolution = float(resolutions[class_id]) if class_id < len(resolutions) else None
        if resolution is not None and not np.isfinite(resolution):
            resolution = None
        classes.append({
            "class_id": class_id,
            "particle_count": int(counts[class_id]) if class_id < len(counts) else 0,
            "resolution_A": resolution,
        })
    return {
        "kind": "class2d_feedback",
        "source": {"project_uid": project_uid, "job_uid": job_uid},
        "input_particle_count": int(len(class_ids)),
        "class_count": class_count,
        "resolution_field": "class_averages.blob/res_A",
        "classes": classes,
    }


def build_class2d_reference_projection_context(
    project_uid: str,
    job_uid: str,
    empiar_id: str | None = None,
    emdb_id: str | None = None,
    reference_map_path: str | None = None,
) -> dict[str, Any]:
    """Compare completed 2D averages to projections of an offline reference map."""
    reference = resolve_reference_volume(empiar_id, emdb_id, reference_map_path)
    if not reference.get("available"):
        return {"status": "unavailable", "reference": reference}
    try:
        cs = cryosparc_client()
        job = cs.find_project(project_uid).find_job(job_uid)
        averages = job.load_output("class_averages")
        paths = list(averages["blob/path"])
        if not paths:
            raise ValueError("Class 2D job has no class average image paths")
        _, stack = job.download_mrc(Path(str(paths[0])).name)
        statistics = build_class2d_feedback_context(project_uid, job_uid)["classes"]
        pixel_sizes = averages.get("blob/psize_A", [])
        if isinstance(pixel_sizes, np.ndarray):
            pixel_sizes = pixel_sizes.tolist()
        pixel_size = float(pixel_sizes[0]) if len(pixel_sizes) else None
        result = compute_reference_projection_metrics(
            reference["path"], np.asarray(stack), statistics,
            class_pixel_size_A=pixel_size,
        )
        result["reference"] = reference
        result["source"] = {"project_uid": project_uid, "job_uid": job_uid}
        return result
    except Exception as exc:
        return {
            "status": "failed",
            "reference": reference,
            "source": {"project_uid": project_uid, "job_uid": job_uid},
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def build_micrograph_preview_visual_context(
    project_uid: str,
    job_uid: str,
    max_micrographs: int = 6,
    tile_size: int = 512,
    micrograph_root: str | None = None,
    output_name: str | None = None,
    micrograph_uids: list[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic preview, optionally restricted to a bound UID subset."""
    cs = cryosparc_client()
    project = cs.find_project(project_uid)
    job = project.find_job(job_uid)
    micrographs = (
        job.load_output(output_name)
        if output_name
        else load_first_available_output(job, ("micrographs", "exposures", "imported_micrographs"))
    )
    paths = [str(path) for path in micrographs.get("micrograph_blob/path", [])]
    if not paths:
        raise ValueError(f"Job {job_uid} has no micrograph paths for visual preview.")
    all_uids = [str(uid) for uid in micrographs.get("uid", [])]
    if micrograph_uids is not None:
        wanted = set(map(str, micrograph_uids))
        bound = [(index, path) for index, path in enumerate(paths)
                 if index < len(all_uids) and all_uids[index] in wanted]
        if len(bound) != len(wanted):
            raise ValueError(
                f"Requested {len(wanted)} bound micrograph UIDs but found {len(bound)} in {job_uid}/{output_name}."
            )
        count = min(max(1, max_micrographs), len(bound))
        sample_positions = np.linspace(0, len(bound) - 1, count).round().astype(int)
        selected = [bound[index] for index in sorted(set(sample_positions))]
    else:
        count = min(max(1, max_micrographs), len(paths))
        indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
        selected = [(int(index), paths[int(index)]) for index in sorted(set(indices))]
    if not selected:
        raise ValueError(f"Job {job_uid}/{output_name} has no selected micrographs for preview.")
    panels = []
    evidence_images = []
    cache_dir = vision_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for panel_index, (source_index, source_path) in enumerate(selected):
        raw_image = load_micrograph_image(project, source_path, micrograph_root)
        pixel_size_A = micrograph_pixel_size(micrographs, source_index)
        image, display = render_full_micrograph_preview(raw_image, pixel_size_A)
        uid_label = all_uids[source_index] if source_index < len(all_uids) else str(source_index)
        evidence_images.append(_save_visual_evidence(
            image, cache_dir, f"{project_uid}_{job_uid}_micrograph_full_{panel_index + 1}_v4.png",
            "full_micrograph_with_scale", f"Full micrograph uid={uid_label}, 1024-square canvas, no crop. Native pixel size {pixel_size_A:g} A; bar={display['scale_bar']['length_A']:.1f} A. Display pixels are not native pixels.",
        ))
        panels.append({"sample_index": source_index, "source_path": source_path,
                       "image_shape_pixels": list(raw_image.shape), "display_processing": display})
    psize_values = np.asarray(micrographs.get("micrograph_blob/psize_A", []), dtype=np.float64)
    psize_values = psize_values[np.isfinite(psize_values) & (psize_values > 0)]
    pixel_size_A = float(np.median(psize_values)) if psize_values.size else None
    return {
        "kind": "micrograph_preview",
        "source": {"project_uid": project_uid, "job_uid": job_uid},
        "image_count": len(panels),
        "micrograph_count": len(paths),
        "selected_micrograph_uids": [all_uids[index] for index, _ in selected if index < len(all_uids)],
        "pixel_size_A_median": pixel_size_A,
        "visual_evidence_version": "full_field_v4",
        "display_processing": "uncropped full field, float area reduction, mean +/-3 SD, Y upward, calibrated scale bar; no Gaussian filter",
        "panels": panels,
        "image_calibrations": [p["display_processing"] for p in panels],
        "evidence_images": evidence_images,
        "evidence_image_descriptions": [item["description"] for item in evidence_images],
    }


def normalize_class_image(array: Any, size: int) -> Image.Image:
    """Convert one MRC image to a contrast-normalized square PNG tile."""
    image = np.asarray(array, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        scaled = np.zeros(image.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((image - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    tile = Image.fromarray(scaled, mode="L").convert("RGB")
    return tile.resize((size, size), Image.Resampling.BILINEAR)


def _normalized_gray(array: Any) -> Image.Image:
    image = np.asarray(array, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            high = low + 1.0
        image = np.clip((image - low) / (high - low) * 255, 0, 255)
    else:
        image = np.zeros(image.shape, dtype=np.float32)
    return Image.fromarray(image.astype(np.uint8), mode="L").convert("RGB")


def render_micrograph_display(array: Any, max_size: int = 1024):
    """Independent implementation of CryoSPARC's binned, +/-3 SD preview."""
    image = np.asarray(array, dtype=np.float32)
    if image.ndim != 2 or max_size < 1:
        raise ValueError("Expected a 2D micrograph and positive display size")
    finite = np.isfinite(image)
    if not finite.all():
        image = np.where(finite, image, np.median(image[finite]) if finite.any() else 0)
    factor = max(1, int(math.ceil(max(image.shape) / max_size)))
    height, width = (dimension // factor for dimension in image.shape)
    if not height or not width:
        raise ValueError("Display binning exceeds a micrograph dimension")
    # Keep the summation order and float32 arithmetic used by block binning.
    blocks = image[:height * factor, :width * factor].reshape(height * factor, width, factor).sum(axis=-1)
    binned = blocks.reshape(height, factor, width).sum(axis=1)
    mean, std = np.float32(binned.mean()), np.float32(binned.std())
    low, high = mean - 3 * std, mean + 3 * std
    pixels = np.clip((binned - low) / (high - low) * 255, 0, 255).astype(np.uint8) if high > low else np.full(binned.shape, 127, dtype=np.uint8)
    return Image.fromarray(pixels[::-1].copy()).convert("RGB"), {
        "method": "block_sum_then_mean_plus_minus_3std", "binfactor": factor,
        "native_extent_pixels": [height * factor, width * factor],
        "display_shape_pixels": [height, width], "y_axis": "upward",
        "additional_gaussian_filter": False,
    }


def _overview_tile(array, size):
    preview, metadata = render_micrograph_display(array, size)
    tile = Image.new("RGB", (size, size), "black")
    offset_x, offset_y = (size - preview.width) // 2, (size - preview.height) // 2
    tile.paste(preview, (offset_x, offset_y))
    metadata["offset_pixels"] = [offset_x, offset_y]
    return tile, metadata


def render_full_micrograph_preview(array, pixel_size_A, size=1024):
    """Full-field area reduction without discarding binning remainder pixels."""
    raw = np.asarray(array, dtype=np.float32)
    if raw.ndim != 2 or size < 32 or not pixel_size_A or not np.isfinite(pixel_size_A) or pixel_size_A <= 0:
        raise ValueError("Expected a 2D image, positive pixel size, and canvas >=32 px")
    native_height, native_width = raw.shape
    ratio = min(1.0, size / max(raw.shape))
    width, height = max(1, round(native_width * ratio)), max(1, round(native_height * ratio))
    finite = np.isfinite(raw)
    if not finite.all():
        raw = np.where(finite, raw, np.median(raw[finite]) if finite.any() else 0)
    reduced = np.asarray(Image.fromarray(raw).resize((width, height), Image.Resampling.BOX))
    mean, std = float(reduced.mean()), float(reduced.std())
    low, high = mean - 3 * std, mean + 3 * std
    pixels = np.clip((reduced - low) / (high - low) * 255, 0, 255).astype(np.uint8) if high > low else np.full((height, width), 127, dtype=np.uint8)
    view = Image.fromarray(pixels[::-1].copy()).convert("RGB")
    # Fractional area reduction keeps even the rightmost/bottommost input pixels.
    display_psize_x = pixel_size_A * native_width / width
    target_A = min(1000.0, width * display_psize_x / 4)
    bar_pixels = max(1, int(round(target_A / display_psize_x)))
    measured_A = bar_pixels * display_psize_x
    draw = ImageDraw.Draw(view)
    x0, y0 = 18, height - 20
    draw.rectangle((x0 - 6, y0 - 24, x0 + bar_pixels + 6, y0 + 6), fill="black")
    draw.line((x0, y0, x0 + bar_pixels, y0), fill="white", width=4)
    draw.text((x0, y0 - 19), f"{measured_A:.0f} A ({measured_A / 10:.1f} nm)", fill="white", font=ImageFont.load_default())
    canvas = Image.new("RGB", (size, size), "black")
    offset_x, offset_y = (size - width) // 2, (size - height) // 2
    canvas.paste(view, (offset_x, offset_y))
    return canvas, {
        "method": "full_field_float_area_downsample_then_mean_plus_minus_3std",
        "source_shape_pixels": list(raw.shape), "image_shape_pixels": [height, width],
        "canvas_shape_pixels": [size, size], "offset_pixels": [offset_x, offset_y],
        "full_field_retained": True, "y_axis": "upward", "additional_gaussian_filter": False,
        "native_pixel_size_A": pixel_size_A, "display_pixel_size_x_A": display_psize_x,
        "display_pixel_size_y_A": pixel_size_A * native_height / height,
        "scale_bar": {"length_pixels": bar_pixels, "length_A": measured_A},
    }


def micrograph_pixel_size(micrographs, index):
    values = micrographs.get("micrograph_blob/psize_A", [])
    value = float(values[index]) if index < len(values) else None
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError("Micrograph pixel size is missing; refusing to fabricate a scale bar")
    return value


def redraw_full_field_scale_bar(image, metadata):
    offset_x, offset_y = metadata["offset_pixels"]
    height, _ = metadata["image_shape_pixels"]
    x0, y0 = offset_x + 18, offset_y + height - 20
    bar = metadata["scale_bar"]
    draw = ImageDraw.Draw(image)
    draw.rectangle((x0 - 6, y0 - 24, x0 + bar["length_pixels"] + 6, y0 + 6), fill="black")
    draw.line((x0, y0, x0 + bar["length_pixels"], y0), fill="white", width=4)
    draw.text((x0, y0 - 19), f"{bar['length_A']:.0f} A ({bar['length_A'] / 10:.1f} nm)", fill="white", font=ImageFont.load_default())


def full_field_pick_position(x_frac, y_frac, metadata):
    height, width = metadata["image_shape_pixels"]
    offset_x, offset_y = metadata["offset_pixels"]
    return offset_x + float(x_frac) * (width - 1), offset_y + (1 - float(y_frac)) * (height - 1)


def _overview_pick_position(x_frac, y_frac, shape, metadata):
    factor = metadata["binfactor"]
    height, _ = metadata["display_shape_pixels"]
    offset_x, offset_y = metadata["offset_pixels"]
    return offset_x + float(x_frac) * (shape[1] - 1) / factor, offset_y + height - 1 - float(y_frac) * (shape[0] - 1) / factor


def _crop_bounds(shape, center_x, center_y, side):
    height, width = shape[-2:]
    side = max(1, min(int(side), width, height))
    left = max(0, min(width - side, int(round(center_x - side / 2))))
    top = max(0, min(height - side, int(round(center_y - side / 2))))
    return left, top, side


def _crop_views(raw, center_x, center_y, side, pixel_size_A):
    side = min(int(side), min(raw.shape[-2:]))
    if side >= 128:
        side = side // 128 * 128
    left, top, side = _crop_bounds(raw.shape, center_x, center_y, side)
    cropped = raw[top:top + side, left:left + side]
    scale = 512 / side
    views, factors = [], []
    for max_size in (512, 128):
        preview, info = render_micrograph_display(cropped, max_size)
        views.append(preview.resize((512, 512), Image.Resampling.NEAREST))
        factors.append(info["binfactor"])
    # Multiples of 128 keep both views on identical native-pixel extents.
    return views, (center_x - left) * scale, (side - 1 - (center_y - top)) * scale, scale, factors


def _join_display_views(views, factors):
    sheet = Image.new("RGB", (1024, 540), "white")
    draw = ImageDraw.Draw(sheet)
    for index, view in enumerate(views):
        sheet.paste(view, (index * 512, 28))
        label = f"CryoSPARC-style: {factors[index]}x{factors[index]} bin, mean +/-3 SD, Y upward"
        draw.text((index * 512 + 5, 8), label, fill="black", font=ImageFont.load_default())
    return sheet


def _scale_bar(draw: ImageDraw.ImageDraw, size: tuple[int, int], pixel_size_A: float | None) -> dict[str, Any] | None:
    if not pixel_size_A or not np.isfinite(pixel_size_A) or pixel_size_A <= 0:
        return None
    width, height = size
    target_A = min(100.0, max(20.0, (width * pixel_size_A) / 4))
    bar_px = max(1, min(width - 24, int(round(target_A / pixel_size_A))))
    x0, y0 = 12, height - 18
    draw.rectangle((x0 - 4, y0 - 8, x0 + bar_px + 4, y0 + 5), fill="black")
    draw.line((x0, y0, x0 + bar_px, y0), fill="white", width=3)
    draw.text((x0, y0 - 7), f"{bar_px * pixel_size_A:.0f} A", fill="white", font=ImageFont.load_default())
    return {"length_A": round(bar_px * pixel_size_A, 2), "length_pixels": bar_px}


def _save_visual_evidence(
    image: Image.Image, cache_dir: Path, filename: str, role: str, description: str,
) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    payload = buffer.getvalue()
    path = cache_dir / filename
    path.write_bytes(payload)
    encoded = base64.b64encode(payload).decode("ascii")
    return {
        "role": role,
        "description": description,
        "mime_type": "image/png",
        "local_path": str(path),
        "data_url": f"data:image/png;base64,{encoded}",
    }


def _crop_square(image: Image.Image, center_x: float, center_y: float, side: int) -> Image.Image:
    side = max(32, min(side, image.width, image.height))
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = max(0, min(image.width - side, left))
    top = max(0, min(image.height - side, top))
    return image.crop((left, top, left + side, top + side))



def build_pick_inspection_visual_context(
    project_uid: str,
    job_uid: str,
    max_micrographs: int = 8,
    max_picks_per_micrograph: int = 400,
    tile_size: int = 512,
    micrograph_root: str | None = None,
) -> dict[str, Any]:
    """Build a micrograph contact sheet with sampled Blob Picker overlays."""
    cs = cryosparc_client()
    project = cs.find_project(project_uid)
    job = project.find_job(job_uid)
    micrographs = load_first_available_output(job, ("micrographs", "exposures", "imported_micrographs"))
    particles = load_first_available_output(job, ("particles", "particles_accepted"))
    mic_paths = list(micrographs["micrograph_blob/path"])
    mic_uids = [int(value) for value in micrographs["uid"]]
    particle_uids = [int(value) for value in particles["location/micrograph_uid"]]
    x_frac = np.asarray(particles["location/center_x_frac"], dtype=np.float32)
    y_frac = np.asarray(particles["location/center_y_frac"], dtype=np.float32)
    ncc = np.asarray(particles["pick_stats/ncc_score"], dtype=np.float32)
    power_available = "pick_stats/power" in particles
    power = np.asarray(particles.get("pick_stats/power", np.full(len(ncc), np.nan)), dtype=np.float32)
    if not mic_paths:
        raise ValueError(f"Job {job_uid} has no micrograph output.")

    by_uid: dict[int, list[int]] = {uid: [] for uid in mic_uids}
    for index, uid in enumerate(particle_uids):
        if uid in by_uid:
            by_uid[uid].append(index)
    ncc_rank = percentile_ranks(ncc)
    power_rank = percentile_ranks(power) if power_available else np.full(len(ncc), np.nan)
    # Cover the whole pick-count range instead of selecting only the densest images.
    ranked_mics = sorted(by_uid, key=lambda uid: len(by_uid[uid]))
    sample_count = max(1, min(max_micrographs, len(ranked_mics)))
    sample_indices = np.linspace(0, len(ranked_mics) - 1, sample_count).round().astype(int)
    chosen_uids = [ranked_mics[index] for index in sorted(set(sample_indices))]
    mic_index = {uid: index for index, uid in enumerate(mic_uids)}
    font = ImageFont.load_default()
    panels = []
    cache_dir = vision_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    overview_images = []
    for panel_index, uid in enumerate(chosen_uids):
        source_path = str(mic_paths[mic_index[uid]])
        raw = load_micrograph_image(project, source_path, micrograph_root)
        image, display_info = render_full_micrograph_preview(raw, micrograph_pixel_size(micrographs, mic_index[uid]))
        draw = ImageDraw.Draw(image)
        picks = by_uid[uid]
        if len(picks) > max_picks_per_micrograph:
            positions = np.linspace(0, len(picks) - 1, max_picks_per_micrograph).astype(int)
            picks = [picks[position] for position in positions]
        for particle_index in picks:
            px, py = full_field_pick_position(x_frac[particle_index], y_frac[particle_index], display_info)
            px, py = int(round(px)), int(round(py))
            radius = 4
            if power_available and np.isfinite(ncc_rank[particle_index]) and np.isfinite(power_rank[particle_index]):
                color = score_rank_color(ncc_rank[particle_index], power_rank[particle_index])
            else:
                color = (255, 50, 30)
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                outline=color, width=2,
            )
        draw.rectangle((0, 0, 1024, 24), fill="black")
        draw.text(
            (5, 7),
            f"micrograph_id={panel_index} uid={uid} picks={len(by_uid[uid])}",
            fill="white", font=font,
        )
        redraw_full_field_scale_bar(image, display_info)
        overview_images.append(_save_visual_evidence(
            image, cache_dir, f"{project_uid}_{job_uid}_pick_full_{panel_index + 1}_v4.png",
            "full_micrograph_pick_overlay", f"Full micrograph uid={uid} with calibrated scale bar and sampled score-colored pick locations. No cropping; colors show score percentiles, not verified quality.",
        ))
        scores = ncc[by_uid[uid]]
        panels.append({
            "micrograph_id": panel_index,
            "micrograph_uid": uid,
            "source_path": source_path,
            "display_processing": display_info,
            "pick_count": len(by_uid[uid]),
            "overlay_count": len(picks),
            "ncc_score_mean": float(np.mean(scores)) if len(scores) else None,
            "ncc_score_min": float(np.min(scores)) if len(scores) else None,
            "ncc_score_max": float(np.max(scores)) if len(scores) else None,
        })

    score_scatter = build_pick_score_scatter(ncc, power)
    scatter_buffer = io.BytesIO()
    score_scatter.save(scatter_buffer, format="PNG", optimize=True)
    scatter_bytes = scatter_buffer.getvalue()
    scatter_path = cache_dir / f"{project_uid}_{job_uid}_ncc_power_scatter.png"
    scatter_path.write_bytes(scatter_bytes)
    scatter_encoded = base64.b64encode(scatter_bytes).decode("ascii")
    cache_dir.mkdir(parents=True, exist_ok=True)
    def summary(values: np.ndarray) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"count": 0}
        q = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
        return {
            "count": int(finite.size),
            "min": float(np.min(finite)),
            "p01": float(q[0]), "p05": float(q[1]), "p25": float(q[2]),
            "p50": float(q[3]), "p75": float(q[4]), "p95": float(q[5]),
            "p99": float(q[6]), "max": float(np.max(finite)),
        }

    score_examples = []
    evidence_images = overview_images[1:]
    psize_values = np.asarray(micrographs.get("micrograph_blob/psize_A", []), dtype=np.float64)
    psize_values = psize_values[np.isfinite(psize_values) & (psize_values > 0)]
    pixel_size_A = float(np.median(psize_values)) if psize_values.size else None
    if power_available:
        targets = (
            ("low_ncc_low_power", 0.15, 0.15),
            ("low_ncc_high_power", 0.15, 0.85),
            ("high_ncc_low_power", 0.85, 0.15),
            ("high_ncc_high_power", 0.85, 0.85),
        )
        valid_scores = np.isfinite(ncc) & np.isfinite(power)
        valid_indices = np.flatnonzero(valid_scores)
        for example_index, (role, target_ncc, target_power) in enumerate(targets):
            if not valid_indices.size:
                break
            distance = (ncc_rank[valid_indices] - target_ncc) ** 2 + (power_rank[valid_indices] - target_power) ** 2
            particle_index = int(valid_indices[int(np.argmin(distance))])
            uid = particle_uids[particle_index]
            source_index = mic_index.get(uid)
            if source_index is None:
                continue
            raw = load_micrograph_image(project, str(mic_paths[source_index]), micrograph_root)
            image, display_info = render_full_micrograph_preview(raw, micrograph_pixel_size(micrographs, source_index))
            cx, cy = full_field_pick_position(x_frac[particle_index], y_frac[particle_index], display_info)
            label = (f"{role} | NCC={ncc[particle_index]:.4g} (P{ncc_rank[particle_index]*100:.0f}) | "
                     f"Power={power[particle_index]:.4g} (P{power_rank[particle_index]*100:.0f})")
            draw = ImageDraw.Draw(image)
            draw.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), outline=(255, 50, 30), width=2)
            draw.rectangle((0, 0, 1024, 28), fill="black")
            draw.text((5, 8), label[:130], fill="white", font=font)
            redraw_full_field_scale_bar(image, display_info)
            evidence = _save_visual_evidence(
                image, cache_dir, f"{project_uid}_{job_uid}_score_example_full_{example_index + 1}_v4.png",
                "score_stratified_full_micrograph", label + "; full compressed micrograph with scale bar; red marker identifies this score example, not its molecular boundary",
            )
            evidence_images.append(evidence)
            score_examples.append({
                "role": role, "particle_index": particle_index, "micrograph_uid": uid,
                "ncc_score": float(ncc[particle_index]), "ncc_percentile": float(ncc_rank[particle_index]),
                "power_score": float(power[particle_index]), "power_percentile": float(power_rank[particle_index]),
                "center_display_pixels": [cx, cy], "display_processing": display_info,
            })

    micrograph_flags = list(micrographs.get("micrograph_blob/is_background_subtracted", []))
    input_is_denoised = bool(micrograph_flags) and all(bool(value) for value in micrograph_flags)
    valid_score_pairs = np.isfinite(ncc) & np.isfinite(power)
    clipped_fraction = None
    if power_available and np.any(valid_score_pairs):
        ncc_bounds = np.percentile(ncc[valid_score_pairs], [1, 99])
        power_bounds = np.percentile(power[valid_score_pairs], [1, 99])
        clipped_fraction = float(np.mean(
            (ncc[valid_score_pairs] < ncc_bounds[0]) | (ncc[valid_score_pairs] > ncc_bounds[1]) |
            (power[valid_score_pairs] < power_bounds[0]) | (power[valid_score_pairs] > power_bounds[1])
        ))
    return {
        "kind": "pick_inspection",
        "visual_evidence_version": "full_field_v4",
        "source": {"project_uid": project_uid, "job_uid": job_uid},
        "image_count": len(panels),
        "input_particle_count": int(len(particle_uids)),
        "input_micrograph_count": int(len(mic_uids)),
        "representative_sampling": "pick_count_quantiles",
        "panels": panels,
        "image_calibrations": [p["display_processing"] for p in panels],
        "score_space": "raw_pick_stats",
        "required_inspect_calibration": {"calibrate_ncc": False, "calibrate_pow": False},
        "density_brightness_note": "Brightness encodes pick-count density, not score magnitude or verified particle quality.",
        "score_statistics": {"ncc_score": summary(ncc), "power": summary(power)},
        "power_score_available": power_available,
        "score_examples": score_examples,
        "evidence_images": evidence_images,
        "evidence_image_descriptions": [item["description"] for item in evidence_images],
        "score_direction_note": "These are raw within-job score percentiles; inspect image evidence before interpreting a score tail.",
        "input_is_denoised": input_is_denoised,
        "auto_cluster_supported": input_is_denoised,
        "auto_cluster_warning": (
            None if input_is_denoised else
            "Input micrographs are not marked denoised; prefer explicit NCC/Power thresholds over auto clustering."
        ),
        "retention_guard": {
            "require_nonempty_selection": True,
            "instruction": (
                "Preserve plausible dense particle populations; heatmap brightness is count density, not score or quality. "
                "Always choose an explicit finite upper Power cutoff. Keep most of the main bright "
                "density core, not a fixed fraction of all picks; more aggressively trim sparse high-Power tails. "
                "Do not choose a nonrestrictive upper bound merely because quality is uncertain."
            )
        },
        "overlay": {
            "marker": "circles; when Power is available, color encodes the within-job mean NCC/Power percentile (red lower, green higher)",
            "color_is_not_a_validated_quality_label": True,
            "max_picks_per_micrograph": max_picks_per_micrograph,
        },
        "contact_sheet": {**overview_images[0], "tile_size": 1024, "columns": 1},
        "score_scatter": {
            "mime_type": "image/png",
            "local_path": str(scatter_path),
            "data_url": f"data:image/png;base64,{scatter_encoded}",
            "x_axis": "NCC score",
            "y_axis": "Power score",
            "point_count": int(len(ncc)),
            "display_range": "p01_to_p99; outliers are clipped to the plot boundary",
            "clipped_fraction": clipped_fraction,
        },
    }


def extract_score_strata(candidates, ncc_rank, power_rank, count=4):
    """Candidate queues by score region; these labels are not quality judgments."""
    candidates = np.asarray(candidates, dtype=int)
    nr, pr = ncc_rank[candidates], power_rank[candidates]
    joint = np.isfinite(pr)
    slots = ["typical", "typical", "high_ncc_moderate_power", "contrast"]
    for slot in range(count):
        role = slots[slot] if slot < len(slots) else "typical"
        target_ncc = .65 if slot == 1 else .5
        if role == "typical":
            mask = (nr >= .25) & (nr <= .8) & joint & (pr >= .25) & (pr <= .75)
            distance = (nr-target_ncc)**2 + (pr-.5)**2
        elif role == "high_ncc_moderate_power":
            mask = (nr >= .75) & (nr <= .95) & joint & (pr >= .2) & (pr <= .8)
            distance = (nr-.85)**2 + (pr-.5)**2
        else:
            mask = joint & ((pr <= .1) | (pr >= .9) | (nr <= .1) | (nr >= .98))
            distance = np.minimum.reduce([(nr-.5)**2+(pr-.95)**2, (nr-.5)**2+(pr-.05)**2,
                                          (nr-.98)**2+(pr-.5)**2, (nr-.05)**2+(pr-.5)**2])
        preferred = candidates[mask][np.argsort(distance[mask], kind="stable")]
        # Missing score regions are never silently relabeled as typical or good.
        fallback_distance = (nr-.5)**2 + np.where(joint, (pr-.5)**2, .25)
        fallback = candidates[np.argsort(fallback_distance, kind="stable")]
        yield role, preferred.tolist(), fallback.tolist()


def build_extract_box_visual_context(
    project_uid: str,
    job_uid: str,
    max_examples: int = 3,
    micrograph_root: str | None = None,
    particle_job_uid: str | None = None,
    examples_per_micrograph: int = 4,
) -> dict[str, Any]:
    """Show spaced interior score-stratified picks per calibrated full micrograph."""
    cs = cryosparc_client()
    project = cs.find_project(project_uid)
    job = project.find_job(job_uid)
    micrographs = load_first_available_output(job, ("micrographs", "exposures", "imported_micrographs"))
    particle_job = project.find_job(particle_job_uid) if particle_job_uid else job
    particles = load_first_available_output(particle_job, ("particles", "particles_accepted"))
    mic_paths = [str(path) for path in micrographs["micrograph_blob/path"]]
    mic_uids = [int(value) for value in micrographs["uid"]]
    particle_uids = np.asarray(particles["location/micrograph_uid"], dtype=np.int64)
    x_frac = np.asarray(particles["location/center_x_frac"], dtype=np.float32)
    y_frac = np.asarray(particles["location/center_y_frac"], dtype=np.float32)
    if not len(particle_uids) or not mic_paths:
        raise ValueError(f"Job {job_uid} has no particles/micrographs for extraction-box evidence.")

    psize_values = np.asarray(micrographs.get("micrograph_blob/psize_A", []), dtype=np.float64)
    psize_values = psize_values[np.isfinite(psize_values) & (psize_values > 0)]
    pixel_size_A = float(np.median(psize_values)) if psize_values.size else None
    params = job.params.model_dump() if hasattr(job.params, "model_dump") else dict(job.params or {})
    picker_parameters = {
        key: params.get(key) for key in ("diameter", "diameter_min", "diameter_max", "min_distance")
        if params.get(key) is not None
    }
    try:
        configured = [int(value) for value in os.getenv(
            "CRYOAGENT_BOX_SIZE_TRIALS", "320,384,440,504,576,640,720"
        ).split(",") if value.strip()]
    except ValueError:
        configured = [320, 384, 440, 504, 576, 640, 720]
    reference_sizes = sorted({value for value in configured if value > 0})
    if not reference_sizes:
        reference_sizes = [320, 384, 440, 504, 576, 640, 720]
    # Draw a readable representative subset; the full ladder remains in structured context.
    positions = np.linspace(0, len(reference_sizes) - 1, min(4, len(reference_sizes))).round().astype(int)
    drawn_sizes = [reference_sizes[index] for index in sorted(set(positions))]
    colors = [(40, 230, 255), (255, 70, 220), (255, 210, 40), (80, 255, 100)]

    ncc = np.asarray(particles.get("pick_stats/ncc_score", []), dtype=np.float32)
    if len(ncc) != len(particle_uids):
        raise ValueError("NCC pick scores are missing; cannot build score-stratified examples")
    power = np.asarray(particles.get("pick_stats/power", []), dtype=np.float32)
    power_available = len(power) == len(ncc) and bool(np.isfinite(power).any())
    if len(power) != len(ncc):
        power = np.full(len(ncc), np.nan)
    ncc_rank, power_rank = percentile_ranks(ncc), percentile_ranks(power)
    mic_index_by_uid = {uid: index for index, uid in enumerate(mic_uids)}
    eligible = np.flatnonzero(np.isfinite(ncc) & np.isfinite(x_frac) & np.isfinite(y_frac)
                             & (x_frac >= 0) & (x_frac <= 1) & (y_frac >= 0) & (y_frac <= 1))
    ranked = eligible[np.argsort(-ncc[eligible], kind="stable")]
    groups = {}
    representative_distance = (ncc_rank[eligible]-.5)**2 + np.where(
        np.isfinite(power_rank[eligible]), (power_rank[eligible]-.5)**2, .25)
    # Select micrographs by representative scores, not their most extreme NCC.
    for index in eligible[np.argsort(representative_distance, kind="stable")]:
        uid = int(particle_uids[index])
        if uid in mic_index_by_uid:
            groups.setdefault(uid, []).append(int(index))
    max_images = min(6, max(1, int(max_examples)))
    per_image = min(8, max(1, int(examples_per_micrograph)))
    largest_box = max(drawn_sizes)
    margin = largest_box / 2 + 1
    minimum_separation = largest_box * 1.05
    cache_dir = vision_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    evidence_images = []
    examples = []
    excluded_edge_count = excluded_nearby_count = 0
    evaluated_micrographs = []
    global_ranks = {int(index): rank + 1 for rank, index in enumerate(ranked)}
    for uid, candidates in groups.items():
        if len(evidence_images) >= max_images:
            break
        mic_index = mic_index_by_uid[uid]
        source_path = mic_paths[mic_index]
        raw = load_micrograph_image(project, source_path, micrograph_root)
        evaluated_micrographs.append(uid)
        height, width = raw.shape
        # Reserve display space for the legend/scale bar without covering a box.
        required_margin = margin + 48 * max(height, width) / 1024
        interior = []
        positions_by_index = {}
        for particle_index in candidates:
            x = float(x_frac[particle_index]) * (width - 1)
            y = float(y_frac[particle_index]) * (height - 1)
            if min(x, y, width - 1 - x, height - 1 - y) < required_margin:
                excluded_edge_count += 1
                continue
            interior.append(particle_index)
            positions_by_index[particle_index] = (x, y)
        selected = []
        annotations = {}
        nearby_rejected = set()
        for requested_role, preferred, fallback in extract_score_strata(interior, ncc_rank, power_rank, per_image):
            chosen = None
            for is_fallback, queue in ((False, preferred), (True, fallback)):
                for particle_index in queue:
                    if particle_index in annotations:
                        continue
                    x, y = positions_by_index[particle_index]
                    if any(max(abs(x-p[1]), abs(y-p[2])) < minimum_separation for p in selected):
                        nearby_rejected.add(particle_index)
                        continue
                    chosen = particle_index
                    selected.append((particle_index, x, y))
                    annotations[particle_index] = {"requested_stratum": requested_role,
                        "score_stratum": "unstratified_fallback" if is_fallback else requested_role,
                        "sampling_fallback": is_fallback,
                        "fallback_reason": "requested score region absent or spatially unavailable; no Power when missing" if is_fallback else None}
                    break
                if chosen is not None:
                    break
        excluded_nearby_count += len(nearby_rejected)
        if not selected:
            continue
        native_psize = micrograph_pixel_size(micrographs, mic_index)
        image, display = render_full_micrograph_preview(raw, native_psize)
        scale_x = display["image_shape_pixels"][1] / raw.shape[1]
        scale_y = display["image_shape_pixels"][0] / raw.shape[0]
        draw = ImageDraw.Draw(image)
        labels = []
        for label_index, (particle_index, _, _) in enumerate(selected, 1):
            label = f"E{len(evidence_images) + 1}.{label_index}"
            labels.append(label)
            center_x, center_y = full_field_pick_position(x_frac[particle_index], y_frac[particle_index], display)
            boxes = []
            for box_size, color in zip(drawn_sizes, colors):
                half_x, half_y = box_size * scale_x / 2, box_size * scale_y / 2
                bounds = [center_x - half_x, center_y - half_y, center_x + half_x, center_y + half_y]
                draw.rectangle(bounds, outline=color, width=1)
                boxes.append({"native_box_size_pix": box_size, "bounds_display_pixels": bounds,
                              "crosses_image_edge": False})
            annotation = annotations[particle_index]
            tag = {"typical": "MID", "high_ncc_moderate_power": "HI-NCC", "contrast": "CONTROL",
                   "unstratified_fallback": "FALLBACK"}[annotation["score_stratum"]]
            draw.text((boxes[-1]["bounds_display_pixels"][0], max(30, boxes[-1]["bounds_display_pixels"][1] - 14)),
                      f"{label} {tag}", fill="white", stroke_fill="black", stroke_width=2,
                      font=ImageFont.load_default())
            examples.append({
                "label": label, "particle_index": particle_index, "micrograph_uid": uid,
                "micrograph_path": source_path, "center_fraction": [float(x_frac[particle_index]), float(y_frac[particle_index])],
                "image_shape_pixels": list(raw.shape), "scale_bar": display["scale_bar"],
                "ncc_score": float(ncc[particle_index]), "score_rank": global_ranks[particle_index],
                "power_score": float(power[particle_index]) if np.isfinite(power[particle_index]) else None,
                "ncc_percentile": float(ncc_rank[particle_index])*100,
                "power_percentile": float(power_rank[particle_index])*100 if np.isfinite(power_rank[particle_index]) else None,
                **annotation,
                "center_display_pixels": [center_x, center_y], "candidate_boxes": boxes,
                "display_processing": display,
                "required_edge_margin_native_pix": required_margin,
            })
        draw.rectangle((0, 0, 1024, 28), fill="black")
        for index, (box_size, color) in enumerate(zip(drawn_sizes, colors)):
            draw.text((5 + index * 125, 8), f"{box_size} native px", fill=color, font=ImageFont.load_default())
        draw.text((530, 8), f"{len(selected)} spaced interior picks; uid={uid}", fill="white", font=ImageFont.load_default())
        redraw_full_field_scale_bar(image, display)
        evidence_images.append(_save_visual_evidence(
            image, cache_dir, f"{project_uid}_{job_uid}_{particle_job_uid or job_uid}_extract_box_strata_{len(evidence_images) + 1}_v6.png",
            "extract_box_size_comparison",
            f"Full micrograph uid={uid}; {len(selected)} spatially separated interior score-stratified picks ({', '.join(labels)}). MID=numerically typical, HI-NCC=higher NCC with moderate Power, CONTROL=score-tail contrast, FALLBACK=requested region unavailable. These are not verified quality labels. Square boxes {drawn_sizes} are native extraction windows, not diameter circles; all fit inside the image.",
        ))
    if not evidence_images:
        raise ValueError("No interior scored picks fit all candidate extraction boxes; edge examples are not substituted")
    return {
        "kind": "extract_box_comparison",
        "visual_evidence_version": "full_field_v6_extract_stratified",
        "example_selection": "2 typical joint-score samples, 1 higher-NCC moderate-Power sample, 1 boundary/extreme control per micrograph when available; geometry-safe median-nearest fallback explicitly labeled",
        "power_score_available": power_available,
        "score_sampling_protocol": {
            "percentile_population": "finite scores in supplied particle output, Inspect-accepted when bound",
            "typical": {"ncc_percentile": [25, 80], "power_percentile": [25, 75], "target_percentiles": [[50, 50], [65, 50]]},
            "high_ncc_moderate_power": {"ncc_percentile": [75, 95], "power_percentile": [20, 80], "target_percentiles": [85, 50]},
            "contrast": "Power <=P10 or >=P90, or NCC <=P10 or >=P98; nearest to a score-tail target, not necessarily the maximum",
            "quality_labels": False, "missing_region_behavior": "explicit unstratified_fallback, no fabricated Power",
        },
        "sampling": {"max_micrographs": max_images, "max_examples_per_micrograph": per_image,
                     "largest_candidate_box_pix": largest_box, "minimum_edge_margin_native_pix": margin,
                     "overlay_annotation_padding_display_pix": 48,
                     "minimum_center_separation_chebyshev_pix": minimum_separation,
                     "evaluated_micrograph_uids": evaluated_micrographs,
                     "excluded_edge_count": excluded_edge_count, "excluded_nearby_count": excluded_nearby_count,
                     "counts_scope": "edge candidates and distinct nearby-rejected candidates in evaluated micrographs, not global pick statistics",
                     "affects_job_particle_selection": False},
        "source": {"project_uid": project_uid, "job_uid": job_uid,
                   "particle_job_uid": particle_job_uid or job_uid},
        "image_count": len(evidence_images),
        "pixel_size_A_median": pixel_size_A,
        "upstream_blob_picker_parameters": picker_parameters,
        "reference_box_sizes_pix": reference_sizes,
        "drawn_box_sizes_pix": drawn_sizes,
        "box_overlay_colors_rgb": {str(size): list(color) for size, color in zip(drawn_sizes, colors)},
        "examples": examples,
        "evidence_images": evidence_images,
        "evidence_image_descriptions": [item["description"] for item in evidence_images],
        "interpretation_note": "Extract cuts square boxes, not diameter circles. MID/HI-NCC are score-region labels, not quality certification. CONTROL can be contamination or a real particle; never infer box size solely from it. FALLBACK is not a typical-sample claim. These interior display samples cannot estimate global edge/junk rates or guarantee optimal box size.",
    }


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if np.any(finite):
        selected = values[finite]
        unique, inverse, counts = np.unique(selected, return_inverse=True, return_counts=True)
        if len(selected) > 1:
            starts = np.cumsum(np.r_[0, counts[:-1]])
            ranks_by_value = (starts + (counts - 1) / 2) / (len(selected) - 1)
        else:
            ranks_by_value = np.asarray([0.5])
        ranks = ranks_by_value[inverse]
        result[finite] = ranks
    return result


def score_rank_color(ncc_rank: float, power_rank: float) -> tuple[int, int, int]:
    quality = max(0.0, min(1.0, (ncc_rank + power_rank) / 2))
    return (int(255 * (1 - quality)), int(80 + 150 * quality), int(30 + 40 * (1 - quality)))


def build_pick_score_scatter(ncc: np.ndarray, power: np.ndarray) -> Image.Image:
    """Render the real NCC/Power distribution for multimodal Inspect Picks decisions."""
    width, height, margin = 720, 560, 70
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    finite = np.isfinite(ncc) & np.isfinite(power)
    x_values, y_values = ncc[finite], power[finite]
    plot_left, plot_top = margin, 35
    plot_right, plot_bottom = width - 25, height - margin
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="black", width=2)
    if x_values.size:
        x_low, x_high = np.percentile(x_values, [1, 99])
        y_low, y_high = np.percentile(y_values, [1, 99])
        if x_high <= x_low:
            x_high = x_low + 1.0
        if y_high <= y_low:
            y_high = y_low + 1.0
        histogram, _, _ = np.histogram2d(
            x_values, y_values, bins=(100, 100), range=((x_low, x_high), (y_low, y_high))
        )
        density = np.log1p(histogram).T[::-1, :]
        if density.max() > 0:
            density = density / density.max()
        heat = np.zeros((100, 100, 3), dtype=np.uint8)
        heat[..., 0] = np.clip(density * 255, 0, 255).astype(np.uint8)
        heat[..., 1] = np.clip(np.sqrt(density) * 190, 0, 190).astype(np.uint8)
        heat[..., 2] = np.clip((1 - density) * 40 + density * 40, 0, 255).astype(np.uint8)
        density_image = Image.fromarray(heat, mode="RGB").resize(
            (plot_right - plot_left, plot_bottom - plot_top), Image.Resampling.NEAREST
        )
        canvas.paste(density_image, (plot_left, plot_top))
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="black", width=2)
        draw.text((plot_left, 10), f"NCC vs Power density ({x_values.size} picks; axes p01-p99)", fill="black", font=font)
        draw.text((plot_left, plot_bottom + 12), f"NCC: {x_low:.3g} to {x_high:.3g}", fill="black", font=font)
        draw.text((plot_left + 260, plot_bottom + 12), f"Power: {y_low:.3g} to {y_high:.3g}", fill="black", font=font)
    else:
        draw.text((plot_left, plot_top + 20), "No finite NCC/Power scores available", fill="black", font=font)
    draw.text((width // 2 - 35, height - 20), "NCC score", fill="black", font=font)
    draw.text((8, height // 2), "Power", fill="black", font=font)
    return canvas


def load_first_available_output(job: Any, output_names: tuple[str, ...]) -> Any:
    """Load a compatible output across Blob Picker and Junk Detector naming variants."""
    errors = []
    for output_name in output_names:
        try:
            return job.load_output(output_name)
        except Exception as exc:
            errors.append(f"{output_name}: {exc}")
    raise ValueError(f"Job {job.uid} has no compatible output among {output_names}: {errors}")


def load_micrograph_image(project: Any, source_path: str, micrograph_root: str | None) -> np.ndarray:
    """Load a micrograph from a shared root when possible, otherwise via API."""
    candidates = []
    if micrograph_root:
        candidates.append(Path(micrograph_root) / Path(source_path).name)
    if Path(source_path).is_absolute():
        candidates.append(Path(source_path))
    for path in candidates:
        if path.exists():
            from cryosparc.mrc import read
            _, image = read(str(path))
            image = np.asarray(image)
            return image[0] if image.ndim == 3 else image
    _, image = project.download_mrc(source_path)
    image = np.asarray(image)
    return image[0] if image.ndim == 3 else image


def normalize_micrograph_image(array: Any, size: int) -> np.ndarray:
    """Aspect-preserving CryoSPARC-style preview in a padded square tile."""
    return np.asarray(_overview_tile(array, size)[0])
