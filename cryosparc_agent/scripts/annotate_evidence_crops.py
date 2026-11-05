"""Exact SVG coordinate overlays on a read-only micrograph preview."""
import argparse
import base64
from pathlib import Path

from cryosparc_client import cryosparc_client
from vision_inputs import _crop_bounds, load_micrograph_image, render_micrograph_display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = "J481/imported/007676200404011108911_14sep05c_00024sq_00003hl_00005es_c.mrc"
    project = cryosparc_client().find_project("P6")
    raw = load_micrograph_image(project, source, None)
    preview, metadata = render_micrograph_display(raw, 1024)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "extract_source_micrograph.png"
    preview.save(png)
    encoded = base64.b64encode(png.read_bytes()).decode("ascii")
    factor = metadata["binfactor"]
    extent_y = metadata["native_extent_pixels"][0]
    margin, header = 30, 95
    width, height = preview.width + 60, preview.height + header + 130
    fragments = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 '<rect width="100%" height="100%" fill="#f8faf5"/>',
                 '<g font-family="monospace" fill="#183c31">',
                 '<text x="30" y="28" font-size="20">EMPIAR-10025: actual Extract evidence crop locations</text>',
                 '<text x="30" y="53" font-size="12">P6/J483 picks | source J481 ...00024sq_00003hl_00005es_c.mrc</text>',
                 '<text x="30" y="76" font-size="12">Same source image; 8x8 binning, mean +/-3 SD, Y upward</text>',
                 f'<image x="{margin}" y="{header}" width="{preview.width}" height="{preview.height}" href="data:image/png;base64,{encoded}"/>']
    regions = [("A: old edge crop", .7086206674575806, .02666666731238365, 912, "#ff7148"),
               ("B: new interior crop", .49827587604522705, .4983333349227905, 896, "#17d8cf")]
    for index, (label, x_frac, y_frac, side, color) in enumerate(regions):
        center_x, center_y = x_frac * (raw.shape[1] - 1), y_frac * (raw.shape[0] - 1)
        left, top, side = _crop_bounds(raw.shape, center_x, center_y, side)
        x, y = margin + left / factor, header + (extent_y - top - side) / factor
        cx, cy = margin + center_x / factor, header + (extent_y - 1 - center_y) / factor
        fragments.extend([
            f'<rect x="{x:.3f}" y="{y:.3f}" width="{side / factor:.3f}" height="{side / factor:.3f}" fill="none" stroke="{color}" stroke-width="3"/>',
            f'<path d="M{cx-7:.3f},{cy:.3f}h14 M{cx:.3f},{cy-7:.3f}v14" stroke="{color}" stroke-width="2"/>',
            f'<rect x="{x:.3f}" y="{y-24:.3f}" width="220" height="22" fill="#152e27"/>',
            f'<text x="{x+5:.3f}" y="{y-8:.3f}" fill="{color}" font-size="12">{label} ({side} native px)</text>',
            f'<text x="30" y="{header+preview.height+30+index*24}" font-size="12">{label}: native crop x={left}, y={top}, side={side}; pick=({center_x:.1f}, {center_y:.1f})</text>'
        ])
    fragments.extend([f'<text x="30" y="{height-24}" font-size="12">Rectangles = exact crop bounds; crosses = pick centers. No original MRC modification.</text>', '</g></svg>'])
    target = args.output_dir / "extract_crop_locations.svg"
    target.write_text("\n".join(fragments))
    print(target)


if __name__ == "__main__":
    main()
