"""Render model-reported approximate measurement boxes as exact SVG overlays."""
import argparse
import base64
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    estimate = json.loads(json.loads(args.result.read_text())["raw_text"])
    encoded = base64.b64encode(args.image.read_bytes()).decode("ascii")
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1064" viewBox="0 0 1024 1064">',
             '<rect width="100%" height="100%" fill="black"/>',
             '<text x="12" y="25" fill="white" font-family="monospace" font-size="15">Model-reported approximate measurements (NOT verified particle boundaries)</text>',
             f'<image x="0" y="40" width="1024" height="1024" href="data:image/png;base64,{encoded}"/>']
    for index, example in enumerate(estimate.get("examples", []), 1):
        x, y = example["x_display_px"], example["y_display_px"]
        width, height = example["width_display_px"], example["height_display_px"]
        lines.extend([
            f'<rect x="{x-width/2}" y="{y+40-height/2}" width="{width}" height="{height}" fill="none" stroke="#ff713c" stroke-width="2"/>',
            f'<text x="{x+width/2+4}" y="{y+40}" fill="#ff713c" font-family="monospace" font-size="13">{index}: {width}x{height} px</text>',
        ])
    lines.append('</svg>')
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
