"""Compare a read-only Import image event with our independently rendered preview."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cryosparc_client import cryosparc_client
from vision_inputs import load_micrograph_image, render_micrograph_display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--import-job", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cs = cryosparc_client()
    project = cs.find_project(args.project)
    events = cs.api.jobs.get_event_logs(args.project, args.import_job)
    event = next(e for e in events if e.type == "image" and e.text.startswith("Raw data "))
    source = event.text[len("Raw data "):]
    asset = next(f for f in event.imgfiles if f.filetype == "png")
    original = Image.open(io.BytesIO(cs.api.assets.download(asset.fileid).dumps())).convert("RGB")
    raw = load_micrograph_image(project, source, None)
    rendered, metadata = render_micrograph_display(raw, 1024)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original.save(args.output_dir / "cryosparc_original.png")
    rendered.save(args.output_dir / "agent_aligned.png")
    result = {"source": source, "processing": metadata, "original_size": original.size,
              "agent_size": rendered.size}
    if original.size == rendered.size:
        difference = np.abs(np.asarray(original).astype(np.int16) - np.asarray(rendered).astype(np.int16))
        result.update(mean_absolute_pixel_error=float(difference.mean()),
                      max_absolute_pixel_error=int(difference.max()), exact_pixel_fraction=float((difference == 0).mean()))
    sheet = Image.new("RGB", (original.width + rendered.width, max(original.height, rendered.height) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "CryoSPARC original Import preview", fill="black")
    draw.text((original.width + 8, 8), "Agent: aligned binning / contrast / orientation", fill="black")
    sheet.paste(original, (0, 30))
    sheet.paste(rendered, (original.width, 30))
    sheet.save(args.output_dir / "comparison.png")
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
