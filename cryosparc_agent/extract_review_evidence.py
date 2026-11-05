"""Read completed pilot stacks and render full-field, coordinate-matched evidence."""

import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cryosparc_client import cryosparc_client
from vision_inputs import render_full_micrograph_preview, full_field_pick_position
from extract_review import PILOT_TITLE_PREFIX
from extract_scale_context import neighbor_summary


def read_stack_slice(path, index):
    with Path(path).open("rb") as stream:
        header = np.frombuffer(stream.read(1024), dtype="<i4")
        nx, ny, nz, mode = map(int, header[:4])
        dtype = np.dtype({0: "i1", 1: "<i2", 2: "<f4", 6: "<u2", 12: "<f2"}[mode])
        if not 0 <= index < nz:
            raise ValueError("Invalid stack index")
        stream.seek(1024 + int(header[23]) + index * nx * ny * dtype.itemsize)
        data = stream.read(nx * ny * dtype.itemsize)
    if len(data) != nx * ny * dtype.itemsize:
        raise ValueError("Incomplete particle stack; refusing partial data")
    return np.frombuffer(data, dtype=dtype).reshape(ny, nx).astype(np.float32)


def particle_sheet(images, labels, psize, lowpass=True):
    tile, stride = 320, 380
    canvas = Image.new("RGB", (4 * tile, ((len(images) + 3) // 4) * stride), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (array, label) in enumerate(zip(images, labels)):
        a = np.asarray(array, dtype=np.float32)
        smooth = a
        if lowpass:
            sigma = 8 / (2.355 * psize)
            fy, fx = np.fft.fftfreq(a.shape[0]), np.fft.rfftfreq(a.shape[1])
            filt = np.exp(-2*np.pi**2*sigma**2*(fy[:, None]**2+fx[None, :]**2))
            smooth = np.fft.irfft2(np.fft.rfft2(a)*filt, s=a.shape).real.astype(np.float32)
        b = np.asarray(Image.fromarray(smooth[::-1].copy()).resize((tile, tile), Image.Resampling.BOX))
        b = np.clip((b-float(b.mean()))/max(6*float(b.std()), 1e-8)+0.5, 0, 1)
        x, y = (i % 4) * tile, (i // 4) * stride
        canvas.paste(Image.fromarray((255*b).astype(np.uint8)).convert("RGB"), (x, y+28))
        view = "8A Gaussian" if lowpass else "raw"
        draw.text((x+4, y+6), f"{label}: {a.shape[1]}px = {a.shape[1]*psize:.1f} A; {view}", fill="black")
        bar_A = min(100, a.shape[1]*psize/3)
        bar = round(bar_A/(a.shape[1]*psize/tile))
        draw.line((x+8, y+358, x+8+bar, y+358), fill="black", width=3)
        draw.text((x+12+bar, y+353), f"{bar_A:.0f} A", fill="black")
    return canvas


def build_evidence(project_uid, pilot_uid, source_uid, directory, alternative_uid=None, history=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    project = cryosparc_client().find_project(project_uid)
    root = Path(str(project.dir()))
    original_job = project.find_job(pilot_uid)
    if original_job.doc.get("status") != "completed":
        raise ValueError("Only completed pilot stacks may be inspected")
    particles = original_job.load_output("particles")
    picks = project.find_job(source_uid).load_output("particles")
    alternate = None
    if alternative_uid:
        job = project.find_job(alternative_uid)
        if job.doc.get("status") != "completed":
            raise ValueError("Comparison pilot is not completed")
        alternate = job.load_output("particles")
    mic_ids = np.asarray(particles["location/micrograph_uid"])
    pick_ids = np.asarray(picks["location/micrograph_uid"])
    psize = float(np.median(particles["blob/psize_A"]))
    size = int(particles["blob/shape"][0][0])
    alt_size = int(alternate["blob/shape"][0][0]) if alternate is not None and len(alternate) else None
    groups = np.unique(mic_ids)
    groups = groups[np.linspace(0, len(groups)-1, min(6, len(groups))).astype(int)]
    samples, images, alternate_images, micrographs = [], [], [], []
    for uid in groups:
        indices = np.flatnonzero(mic_ids == uid)
        mic_picks = np.flatnonzero(pick_ids == uid)
        if not len(mic_picks):
            continue
        if not np.isclose(float(picks["location/micrograph_psize_A"][mic_picks[0]]), psize):
            raise ValueError("Review requires native-sampled particle/micrograph agreement")
        source_path = root / str(picks["location/micrograph_path"][mic_picks[0]])
        mic = read_stack_slice(source_path, 0)
        overview, info = render_full_micrograph_preview(mic, psize)
        draw = ImageDraw.Draw(overview)
        points = np.column_stack((np.asarray(picks["location/center_x_frac"])[mic_picks]*mic.shape[1],
                                  np.asarray(picks["location/center_y_frac"])[mic_picks]*mic.shape[0]))
        # Moderate joint NCC/Power ranks, never the extreme high-score tail.
        ncc = np.asarray(particles["pick_stats/ncc_score"])[indices]
        power = np.asarray(particles["pick_stats/power"])[indices]
        finite = np.isfinite(ncc) & np.isfinite(power)
        indices, ncc, power = indices[finite], ncc[finite], power[finite]
        if not len(indices):
            continue
        nr = np.argsort(np.argsort(ncc)) / max(1, len(indices)-1)
        pr = np.argsort(np.argsort(power)) / max(1, len(indices)-1)
        selected = []
        for target in (0.4, 0.7):
            choices = indices[np.argsort((nr-target)**2 + (pr-0.5)**2)]
            choice = next((int(i) for i in choices if int(i) not in selected), None)
            if choice is not None:
                selected.append(choice)
        for index in selected:
            center = [float(particles["location/center_x_frac"][index]), float(particles["location/center_y_frac"][index])]
            alt_index = None
            if alternate is not None:
                match = np.flatnonzero((np.asarray(alternate["location/micrograph_uid"]) == uid)
                    & (np.abs(np.asarray(alternate["location/center_x_frac"])-center[0]) < 1e-7)
                    & (np.abs(np.asarray(alternate["location/center_y_frac"])-center[1]) < 1e-7))
                if len(match) != 1:
                    continue
                alt_index = int(match[0])
            label = f"S{len(samples)+1:02d}"
            image = read_stack_slice(root / str(particles["blob/path"][index]), int(particles["blob/idx"][index]))
            images.append(image)
            if alt_index is not None:
                if not np.isclose(float(alternate["blob/psize_A"][alt_index]), psize):
                    raise ValueError("Comparison particle sampling differs")
                alternate_images.append(read_stack_slice(root / str(alternate["blob/path"][alt_index]), int(alternate["blob/idx"][alt_index])))
            delta = points - np.asarray(center)*np.array(mic.shape[::-1])
            distances = np.linalg.norm(delta, axis=1)
            other = distances > 1
            sizes = [size] + ([alt_size] if alt_size else [])
            counts = {str(s): int(np.sum(other & (np.max(np.abs(delta), axis=1) < s/2))) for s in sizes}
            samples.append({"sample_id": label, "micrograph_uid": int(uid), "center_fraction": center,
                "extra_picked_centers_by_box_px": counts,
                "nearest_other_center_distance_px": float(distances[other].min()) if other.any() else None})
            cx, cy = full_field_pick_position(*center, info)
            h, w = info["image_shape_pixels"]
            for s, color in zip(sizes, ("magenta", "cyan")):
                hx, hy = s*w/mic.shape[1]/2, s*h/mic.shape[0]/2
                draw.rectangle((cx-hx, cy-hy, cx+hx, cy+hy), outline=color, width=1)
            near = other & (np.max(np.abs(delta), axis=1) < max(sizes)/2)
            for position in points[near]:
                px, py = full_field_pick_position(position[0]/mic.shape[1], position[1]/mic.shape[0], info)
                draw.line((px-2, py, px+2, py), fill="yellow")
            draw.ellipse((cx-2, cy-2, cx+2, cy+2), outline="white")
            draw.text((cx+size*w/mic.shape[1]/2+3, cy), label, fill="white", stroke_width=1, stroke_fill="black")
        name = f"micrograph_{int(uid)}.png"
        overview.save(directory/name)
        micrographs.append(name)
    if len(samples) < 4:
        raise ValueError("Fewer than four matched samples; review cannot support an automatic size change")
    labels = [s["sample_id"] for s in samples]
    names = []
    for filtered, suffix in [(False, "raw"), (True, "filtered")]:
        name = f"current_particles_{suffix}.png"
        particle_sheet(images, labels, psize, filtered).save(directory/name)
        names.append(name)
    if alternate_images:
        for filtered, suffix in [(False, "raw"), (True, "filtered")]:
            name = f"alternative_particles_{suffix}.png"
            particle_sheet(alternate_images, labels, psize, filtered).save(directory/name)
            names.append(name)
    names.extend(micrographs)
    facts = {"current_box_size_pix": size, "alternative_box_size_pix": alt_size,
             "native_pixel_size_A": psize, "samples": samples, "historical_baselines": history or [],
             "neighbor_summary": neighbor_summary(samples,size),
             "limitations": ["Neighbor picks are not verified molecular identities; their counts alone do not determine size.",
                 "Different-population historical FSC does not prove optimality.",
                 "Particle sheets use Gaussian FWHM 8 A for display; full micrographs are uncropped area-downsampled images. All images Y upward.",
                 "Magenta=current box, cyan=actual tested alternative when present, white=target, yellow=other automatic pick.",
                 "A soft-mask-based picker diameter may already include margin and is not a direct molecular boundary measurement."],
             "image_order": names, "pilot_job_uid": pilot_uid, "alternative_job_uid": alternative_uid}
    (directory/"evidence.json").write_text(json.dumps(facts, indent=2))
    content = [{"type": "text", "text": json.dumps(facts)}]
    for name in names:
        encoded = base64.b64encode((directory/name).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,"+encoded, "detail": "high"}})
    return content, facts
