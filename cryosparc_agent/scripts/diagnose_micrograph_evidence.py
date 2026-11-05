"""Read-only comparison of raw and low-pass display crops, without model calls."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vision_inputs import _normalized_gray, load_micrograph_image
from cryosparc_client import cryosparc_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = cryosparc_client().find_project(args.project)
    raw = load_micrograph_image(project, args.path, None)
    sheet = Image.new("RGB", (1024, 1080), "white")
    draw = ImageDraw.Draw(sheet)
    for row, y in enumerate((args.y, 1 - args.y)):
        left = max(0, min(raw.shape[1] - 768, round(args.x * (raw.shape[1] - 1) - 384)))
        top = max(0, min(raw.shape[0] - 768, round(y * (raw.shape[0] - 1) - 384)))
        crop = np.asarray(raw[top:top + 768, left:left + 768], dtype=np.float32)
        fy = np.fft.fftfreq(crop.shape[0])[:, None]
        fx = np.fft.rfftfreq(crop.shape[1])[None, :]
        filtered = np.fft.irfft2(np.fft.rfft2(crop) * np.exp(-2 * np.pi**2 * 25 * (fx**2 + fy**2)), s=crop.shape)
        for col, image in enumerate((crop, filtered)):
            rendered = _normalized_gray(image).resize((512, 512), Image.Resampling.LANCZOS)
            sheet.paste(rendered, (col * 512, row * 540 + 28))
            draw.text((col * 512 + 8, row * 540 + 8),
                      f"{'raw' if col == 0 else 'Gaussian sigma=5 native px'} | y={'direct' if row == 0 else 'flipped'}", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print({"shape": raw.shape, "finite_fraction": float(np.isfinite(raw).mean()), "output": str(args.output)})


if __name__ == "__main__":
    main()
