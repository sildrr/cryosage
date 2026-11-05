"""Create a small, auditable pre-extraction box-size visual probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cryosparc.mrc import read as read_mrc

from cryosparc_client import cryosparc_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--box-size-pix", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--micrographs", type=int, default=4)
    parser.add_argument("--picks-per-micrograph", type=int, default=4)
    return parser.parse_args()


def choose_micrographs(micrographs, particles, count: int) -> list[int]:
    ids, counts = np.unique(np.asarray(particles["location/micrograph_uid"]), return_counts=True)
    count_by_uid = dict(zip(ids.tolist(), counts.tolist()))
    uid = np.asarray(micrographs["uid"])
    df1 = np.asarray(micrographs["ctf/df1_A"], dtype=float)
    df2 = np.asarray(micrographs["ctf/df2_A"], dtype=float)
    defocus = (df1 + df2) / 2
    eligible = np.flatnonzero(np.isfinite(defocus) & (defocus > 0))
    eligible = [i for i in eligible if count_by_uid.get(int(uid[i]), 0) >= 20]
    eligible.sort(key=lambda i: defocus[i])
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} micrographs have usable defocus and picks")
    positions = np.linspace(0.1, 0.9, count)
    return [eligible[int(round(position * (len(eligible) - 1)))] for position in positions]


def select_picks(particles, uid: int, shape: tuple[int, int], crop_size: int, count: int) -> list[int]:
    all_uid = np.asarray(particles["location/micrograph_uid"])
    indices = np.flatnonzero(all_uid == uid)
    x = np.asarray(particles["location/center_x_frac"])[indices] * shape[1]
    y = np.asarray(particles["location/center_y_frac"])[indices] * shape[0]
    score = np.asarray(particles["pick_stats/ncc_score"])[indices]
    radius = crop_size // 2 + 2
    good = np.isfinite(score) & (x >= radius) & (x < shape[1] - radius)
    good &= (y >= radius) & (y < shape[0] - radius)
    indices, x, y, score = indices[good], x[good], y[good], score[good]
    if len(indices) < count:
        raise ValueError(f"Micrograph {uid} has only {len(indices)} interior picks")
    selected: list[int] = []
    for quantile in np.linspace(0.25, 0.9, count):
        target = np.quantile(score, quantile)
        for candidate in np.argsort(np.abs(score - target)):
            if all(np.hypot(x[candidate] - x[old], y[candidate] - y[old]) >= crop_size for old in selected):
                selected.append(int(candidate))
                break
    return indices[selected].tolist()


def render_patch(data: np.ndarray, center_x: int, center_y: int, crop_size: int, box_size: int,
                 tile_size: int) -> Image.Image:
    half = crop_size // 2
    patch = np.asarray(data[center_y-half:center_y+half, center_x-half:center_x+half], dtype=np.float32)
    low, high = np.percentile(patch, (1, 99))
    if high <= low:
        high = low + 1
    scaled = np.uint8(np.clip((patch - low) * 255 / (high - low), 0, 255))
    image = Image.fromarray(scaled, mode="L").resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    box_half = box_size * tile_size / crop_size / 2
    center = tile_size / 2
    draw.rectangle((center-box_half, center-box_half, center+box_half, center+box_half),
                   outline=(255, 74, 38), width=3)
    return image


def main() -> None:
    args = parse_args()
    if args.box_size_pix <= 0 or args.micrographs <= 0 or args.picks_per_micrograph <= 0:
        raise ValueError("Box size and sample counts must be positive")
    crop_size = max(640, int(np.ceil(args.box_size_pix * 1.5 / 2) * 2))
    project = cryosparc_client().find_project(args.project)
    job = project.find_job(args.job)
    micrographs = job.load_output("micrographs")
    particles = job.load_output("particles")
    chosen_mics = choose_micrographs(micrographs, particles, args.micrographs)
    tile_size, label_height = 320, 38
    sheet = Image.new("RGB", (args.picks_per_micrograph * tile_size,
                               args.micrographs * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    records = []
    for row, mic_index in enumerate(chosen_mics):
        uid = int(micrographs["uid"][mic_index])
        source = str(micrographs["micrograph_blob/path"][mic_index])
        path = args.project_dir / source
        if not path.is_file():
            raise FileNotFoundError(path)
        _, data = read_mrc(str(path))
        if data.ndim == 3:
            data = data[0]
        shape = tuple(data.shape)
        picked = select_picks(particles, uid, shape, crop_size, args.picks_per_micrograph)
        for col, pick_index in enumerate(picked):
            x = int(round(float(particles["location/center_x_frac"][pick_index]) * shape[1]))
            y = int(round(float(particles["location/center_y_frac"][pick_index]) * shape[0]))
            tile = render_patch(data, x, y, crop_size, args.box_size_pix, tile_size)
            left, top = col * tile_size, row * (tile_size + label_height)
            sheet.paste(tile, (left, top + label_height))
            panel = row * args.picks_per_micrograph + col + 1
            draw.rectangle((left, top, left + tile_size - 1, top + label_height - 1), fill="black")
            draw.text((left + 6, top + 10), f"#{panel}  defocus={float(micrographs['ctf/df1_A'][mic_index])/1e4:.2f} um  NCC={float(particles['pick_stats/ncc_score'][pick_index]):.3f}",
                      fill="white", font=ImageFont.load_default())
            records.append({"panel": panel, "micrograph_uid": uid, "micrograph_path": source,
                            "pick_uid": int(particles["uid"][pick_index]), "center_xy_pix": [x, y],
                            "ncc_score": float(particles["pick_stats/ncc_score"][pick_index]),
                            "defocus_A": float((micrographs["ctf/df1_A"][mic_index] + micrographs["ctf/df2_A"][mic_index]) / 2)})
    pixel_sizes = [float(micrographs["micrograph_blob/psize_A"][i]) for i in chosen_mics]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "j4_pick_box_probe.png"
    sheet.save(image_path, optimize=True)
    metadata = {"project_uid": args.project, "job_uid": args.job, "image_path": str(image_path),
                "box_size_pix": args.box_size_pix, "crop_size_pix": crop_size,
                "pixel_size_A": float(np.median(pixel_sizes)), "box_width_A": float(args.box_size_pix * np.median(pixel_sizes)),
                "panels": records}
    (args.output_dir / "sample_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "panels"}, indent=2))


if __name__ == "__main__":
    main()
