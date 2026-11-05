"""Read-only extraction review: sample real boxes, optionally probe the visual API."""

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryosparc_client import cryosparc_client
from model_direct_runner import resolve_api_key, run_openai_compatible_model


def read_slice(path, index):
    with path.open("rb") as stream:
        header = np.frombuffer(stream.read(1024), dtype="<i4")
    nx, ny, nz, mode = map(int, header[:4])
    dtype = {0: "i1", 1: "<i2", 2: "<f4", 6: "<u2", 12: "<f2"}[mode]
    if not 0 <= index < nz:
        raise ValueError("Invalid particle stack index")
    offset = 1024 + int(header[23]) + index * nx * ny * np.dtype(dtype).itemsize
    with path.open("rb") as stream:
        stream.seek(offset)
        image = np.frombuffer(stream.read(nx * ny * np.dtype(dtype).itemsize), dtype=dtype)
    return image.reshape(ny, nx).astype(np.float32)


def render_sheet(images, ids, psize, lowpass_A=None):
    tile, top, bottom = 320, 28, 32
    sheet = Image.new("RGB", (tile * 4, (tile + top + bottom) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(zip(images, ids)):
        a = np.asarray(image, dtype=np.float32)
        if lowpass_A:
            sigma = lowpass_A / (2.355 * psize)
            fy, fx = np.fft.fftfreq(a.shape[0]), np.fft.rfftfreq(a.shape[1])
            filt = np.exp(-2 * np.pi**2 * sigma**2 * (fy[:, None]**2 + fx[None, :]**2))
            a = np.fft.irfft2(np.fft.rfft2(a) * filt, s=a.shape).real.astype(np.float32)
        # Area-average intensities before display scaling; retain the entire box.
        display = np.asarray(Image.fromarray(a).resize((tile, tile), Image.Resampling.BOX))
        mean, std = float(np.nanmean(display)), float(np.nanstd(display))
        pixels = np.clip((display - mean) / max(std * 6, 1e-8) + 0.5, 0, 1)
        x, y = (i % 4) * tile, (i // 4) * (tile + top + bottom)
        draw.text((x + 4, y + 5), f"{label} | box={a.shape[1]} px | {a.shape[1]*psize:.1f} A", fill="black")
        sheet.paste(Image.fromarray((pixels * 255).astype(np.uint8)).convert("RGB"), (x, y + top))
        bar = round(100 / (a.shape[1] * psize / tile))
        draw.line((x + 8, y + top + tile + 8, x + 8 + bar, y + top + tile + 8), fill="black", width=3)
        draw.text((x + 12 + bar, y + top + tile + 3), "100 A", fill="black")
    return sheet


def build_evidence(args):
    project = cryosparc_client().find_project(args.project)
    job = project.find_job(args.extract_job)
    if job.doc.get("status") != "completed":
        raise ValueError("Source extraction must be completed")
    particles = job.load_output("particles")
    picks = project.find_job(args.source_picks).load_output("particles")
    uid = np.asarray(particles["location/micrograph_uid"])
    pick_uid = np.asarray(picks["location/micrograph_uid"])
    groups = np.unique(uid)
    groups = groups[np.linspace(0, len(groups) - 1, min(6, len(groups))).astype(int)]
    ncc = np.asarray(particles["pick_stats/ncc_score"], dtype=float)
    power = np.asarray(particles["pick_stats/power"], dtype=float)
    images, rows = [], []
    psize = float(np.median(particles["blob/psize_A"]))
    project_dir = Path(str(project.dir()))
    for micrograph in groups:
        indices = np.flatnonzero((uid == micrograph) & np.isfinite(ncc) & np.isfinite(power))
        if len(indices) < 2:
            continue
        nr = np.argsort(np.argsort(ncc[indices])).astype(float) / max(1, len(indices) - 1)
        pr = np.argsort(np.argsort(power[indices])).astype(float) / max(1, len(indices) - 1)
        used = set()
        for target in (0.4, 0.7):
            order = np.argsort((nr - target)**2 + (pr - 0.5)**2)
            index = next(int(indices[v]) for v in order if int(indices[v]) not in used)
            used.add(index)
            path = project_dir / str(particles["blob/path"][index])
            image = read_slice(path, int(particles["blob/idx"][index]))
            if args.comparison_box >= image.shape[0] or args.comparison_box % 2:
                raise ValueError("Comparison box must be smaller and even")
            images.append(image)
            x, y = float(particles["location/center_x_frac"][index]), float(particles["location/center_y_frac"][index])
            matching = np.flatnonzero(pick_uid == micrograph)
            shape = np.asarray(picks["location/micrograph_shape"][matching[0]], dtype=float)
            dx = (np.asarray(picks["location/center_x_frac"])[matching] - x) * shape[1]
            dy = (np.asarray(picks["location/center_y_frac"])[matching] - y) * shape[0]
            distance = np.hypot(dx, dy)
            rows.append({
                "sample_id": f"S{len(rows)+1:02d}", "particle_row": index,
                "stack_path": str(path), "stack_index": int(particles["blob/idx"][index]),
                "micrograph_uid": int(micrograph), "center_fraction": [x, y],
                "ncc": float(ncc[index]), "power": float(power[index]),
                "other_picked_centers_inside_original_box": int(np.sum((np.abs(dx) < image.shape[1]/2) & (np.abs(dy) < image.shape[0]/2) & (distance > 1))),
                "other_picked_centers_inside_comparison_box": int(np.sum((np.abs(dx) < args.comparison_box/2) & (np.abs(dy) < args.comparison_box/2) & (distance > 1))),
                "note": "Other pick centers are coordinate evidence, not verified molecular identities.",
            })
    if len(images) != 12:
        raise ValueError("This pilot requires twelve samples from six micrographs")
    ids = [r["sample_id"] for r in rows]
    cropped = []
    for image in images:
        start = (image.shape[0] - args.comparison_box) // 2
        cropped.append(image[start:start+args.comparison_box, start:start+args.comparison_box])
    for name, samples, lp in [
        ("original_raw.png", images, None), ("original_lowpass8A.png", images, 8),
        ("comparison_raw.png", cropped, None), ("comparison_lowpass8A.png", cropped, 8),
    ]:
        render_sheet(samples, ids, psize, lp).save(args.output_dir / name)
    metadata = {
        "project": args.project, "extract_job": args.extract_job, "source_picks": args.source_picks,
        "native_pixel_size_A": psize, "original_box_px": images[0].shape[0],
        "comparison_box_px": args.comparison_box,
        "sampling": "Six evenly spaced micrograph UIDs, two moderate joint-score picks per micrograph; no selection by crowding or appearance.",
        "comparison_is": "A centered geometric crop of the same original extracted particle, NOT another job's output.",
        "lowpass": "Gaussian FWHM 8 A, diagnostic display only; original stack unchanged.",
        "submitted_jobs": False, "samples": rows,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="P6")
    parser.add_argument("--extract-job", default="J606")
    parser.add_argument("--source-picks", default="J605")
    parser.add_argument("--comparison-box", type=int, default=440)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submit-api", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else build_evidence(args)
    if (meta["project"], meta["extract_job"], meta["source_picks"], meta["comparison_box_px"]) != (
        args.project, args.extract_job, args.source_picks, args.comparison_box,
    ):
        raise ValueError("Existing evidence belongs to different arguments")
    if not args.submit_api:
        print(json.dumps({"evidence_directory": str(args.output_dir), "sample_count": len(meta["samples"])}))
        return
    from webapp.manage import apply_runtime_profile
    from webapp.proxy import configure_compute_proxy
    apply_runtime_profile()
    configure_compute_proxy()
    base = os.environ["OPENAI_BASE_URL"]
    model = os.environ["OPENAI_MODEL"]
    key = resolve_api_key()
    common = (
        "Independently assess real, unaligned cryo-EM particle extraction images, not Class 2D averages. "
        "Every tile is a complete extracted box, resized for display with a 100 A scale bar and sample ID. "
        "The raw and low-pass sheets show the SAME particles: do not count them twice. The low-pass "
        "view uses an 8 A FWHM Gaussian only to aid perception; filtering can merge neighboring density. "
        "Do not mistake noise/CTF rings for distinct particles. Distinguish a centered single intact "
        "target, additional separate particle-like objects, off-centering, truncation, and insufficient "
        "signal. A neighboring object alone does not prove extraction is wrong: preserving molecular "
        "extent and CTF-related margin also matters. Use visible evidence, not a presumption that a "
        "particular box size is good or bad. Return JSON only, with descriptive strings in Chinese: "
        "per_sample (sample_id, assessment, visible_particle_like_object_count or null, "
        "centered_target_complete or null, observation, confidence), overall_assessment, "
        "box_size_recommendation (maintain|decrease|increase|insufficient_evidence), "
        "recommended_box_px or null, rationale, and limitations."
    )
    for label, names, extra in [
        ("blind_original", ["original_raw.png", "original_lowpass8A.png"],
         f"Only the current extraction is shown: {meta['original_box_px']} native px at {meta['native_pixel_size_A']} A/px. No later outcome or expected particle diameter is provided."),
        ("paired_comparison", ["original_raw.png", "original_lowpass8A.png", "comparison_raw.png", "comparison_lowpass8A.png"],
         f"The first two sheets use {meta['original_box_px']} native px. The last two are centered geometric {meta['comparison_box_px']} px crops of the SAME samples, not separately extracted outputs. The display tiles have equal pixel dimensions but different physical fields of view. Compare completeness and extra objects; it is acceptable to prefer either or find evidence insufficient."),
    ]:
        if (args.output_dir / (label + "_result.json")).exists():
            raise ValueError(f"Refusing to repeat an already saved paid API probe: {label}")
        content = [{"type": "text", "text": common + "\n" + extra}]
        for name in names:
            image = base64.b64encode((args.output_dir / name).read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + image, "detail": "high"}})
        (args.output_dir / (label + "_prompt.txt")).write_text(common + "\n" + extra)
        start = time.monotonic()
        result = run_openai_compatible_model([{"role": "user", "content": content}], base, key, model,
                                             max_new_tokens=6144, timeout_seconds=180, max_retries=0)
        result["elapsed_seconds"] = round(time.monotonic() - start, 2)
        (args.output_dir / (label + "_result.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (args.output_dir / (label + "_output.txt")).write_text(result["raw_text"])
        print(json.dumps({"probe": label, "elapsed_seconds": result["elapsed_seconds"],
                          "finish_reason": result["raw_response"]["choices"][0].get("finish_reason"),
                          "output": result["raw_text"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
