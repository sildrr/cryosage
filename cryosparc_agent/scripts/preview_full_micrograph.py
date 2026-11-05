"""Save a calibrated, uncropped full micrograph for human visual review."""
import argparse
import json
from pathlib import Path

from cryosparc_client import cryosparc_client
from vision_inputs import load_micrograph_image, render_full_micrograph_preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--pixel-size-A", type=float, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = load_micrograph_image(cryosparc_client().find_project(args.project), args.path, None)
    preview, metadata = render_full_micrograph_preview(raw, args.pixel_size_A, args.size)
    metadata["source"] = {"project": args.project, "path": args.path}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(args.output, optimize=True)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size, **metadata}))


if __name__ == "__main__":
    main()
