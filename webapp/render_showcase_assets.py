"""CPU-only scientific previews; requires the existing CryoSPARC worker environment."""
import json
import argparse
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from PIL import Image
from skimage.measure import marching_cubes

from webapp.core import REPO, SHOWCASE
from cryosparc_agent.reference_projection_metrics import read_mrc_volume

BACKGROUND = "#f4f6ee"
GREEN = "#205e4d"


@njit(cache=True)
def rasterize_surface(points, faces, shades, width, height):
    depth = np.full((height, width), -np.inf, dtype=np.float32)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = 244
    image[:, :, 1] = 246
    image[:, :, 2] = 238
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denominator) < 1e-10:
            continue
        x0 = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        x1 = min(width-1, int(np.ceil(max(a[0], b[0], c[0]))))
        y0 = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        y1 = min(height-1, int(np.ceil(max(a[1], b[1], c[1]))))
        for y in range(y0, y1+1):
            for x in range(x0, x1+1):
                u = ((b[1]-c[1])*(x+.5-c[0]) + (c[0]-b[0])*(y+.5-c[1])) / denominator
                v = ((c[1]-a[1])*(x+.5-c[0]) + (a[0]-c[0])*(y+.5-c[1])) / denominator
                w = 1-u-v
                if min(u, v, w) < -1e-6:
                    continue
                z = u*a[2] + v*b[2] + w*c[2]
                if z > depth[y, x]:
                    depth[y, x] = z
                    gray = int(255 * (u*shades[face[0]] + v*shades[face[1]] + w*shades[face[2]]))
                    image[y, x, :] = gray
    return image


def draw_fsc(curves, resolution, target):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": GREEN, "axes.labelcolor": GREEN,
                         "xtick.color": "#6e7b73", "ytick.color": "#6e7b73"})
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=170, facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    for index, curve in enumerate(curves):
        ax.plot(curve["frequency_inv_A"], curve["values"],
                color=(GREEN, "#88a594", "#b9c5aa")[index % 3],
                linewidth=2.5 if index == 0 else 1.5,
                linestyle="-" if index == 0 else "--", label=curve["label"])
    ax.axhline(.143, color="#b66a3e", linewidth=1, linestyle=(0, (4, 4)))
    ax.text(.99, .175, "FSC = 0.143", transform=ax.get_yaxis_transform(),
            ha="right", fontsize=9, color="#b66a3e")
    ax.axvline(1 / resolution, color=GREEN, alpha=.25, linewidth=1)
    ax.set(xlim=(0, max(max(c["frequency_inv_A"]) for c in curves)), ylim=(-.08, 1.05),
           xlabel="Spatial frequency (1/Å)", ylabel="FSC")
    ax.set_title(f"{resolution:.2f} Å", fontfamily="DejaVu Serif", fontsize=22, loc="left", pad=16)
    ax.grid(axis="y", color="#d6ded4", linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#bdcbbd")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout(pad=1.6)
    fig.savefig(target, facecolor=BACKGROUND)
    plt.close(fig)


def draw_shape(map_file, target):
    volume, voxel = read_mrc_volume(map_file)
    original_shape = volume.shape
    level = float(volume.mean() + 3.0 * volume.std())
    vertices, faces, normals, _ = marching_cubes(volume, level=level, step_size=1, allow_degenerate=False)
    del volume
    vertices = vertices[:, ::-1]  # MRC z,y,x to plotting x,y,z.
    normals = normals[:, ::-1]
    # Show the broad face rather than an accidental edge-on view. This rotation
    # is for presentation only; preserve handedness and never modify the map.
    vertices -= vertices.mean(axis=0)
    _, basis = np.linalg.eigh(np.cov(vertices.T))
    basis = basis[:, [1, 0, 2]]
    if np.linalg.det(basis) < 0:
        basis[:, 1] *= -1
    vertices = vertices @ basis
    normals = normals @ basis
    elevation, azimuth = np.deg2rad([18, -75])
    eye = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    up = np.cross(eye, right)
    camera = np.column_stack((right, up, eye))
    points = vertices @ camera
    view_normals = normals @ camera
    light = np.array([-.35, .5, 1]); light /= np.linalg.norm(light)
    fill = np.array([.8, -.1, .6]); fill /= np.linalg.norm(fill)
    shades = np.clip(.25 + .48*np.abs(view_normals @ light) + .12*np.abs(view_normals @ fill), 0, 1)
    width, height, supersampling = 1600, 1080, 2
    low, high = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    center = (low + high) / 2
    scale = min(width*.84 / max(high[0]-low[0], 1), height*.84 / max(high[1]-low[1], 1))
    points[:, 0] = ((points[:, 0]-center[0])*scale + width/2)*supersampling
    points[:, 1] = (height/2 - (points[:, 1]-center[1])*scale)*supersampling
    pixels = rasterize_surface(points.astype(np.float32), faces, shades.astype(np.float32), width*supersampling, height*supersampling)
    image = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
    temporary = target.with_name(f".{uuid.uuid4().hex}.png")
    try:
        image.save(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"original_shape": original_shape, "voxel_A": voxel,
            "quality_version": "native_voxel_v2", "display_max_voxels": max(original_shape), "gaussian_sigma_display_voxels": 0,
            "native_voxels_preserved": True, "mesh_decimation": False, "triangle_count": len(faces),
            "isolevel": level, "isolevel_rule": "mean + 3 SD of original native sharpened map (display contour only)",
            "disconnected_component_min_fraction": 0, "elevation": 18, "azimuth": -75,
            "shading": "Native density-gradient vertex normals, barycentric Gouraud interpolation, CPU z-buffer, 2x image supersampling",
            "presentation_rotation": "Right-handed PCA basis; longest axis vertical, broad face toward camera",
            "surface_color": "neutral gray; equal RGB components with directional lighting",
            "note": "Display-only isosurface, not biological orientation or structural validation; source map unchanged."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=[case["id"] for case in SHOWCASE])
    parser.add_argument("--shapes-only", action="store_true")
    args = parser.parse_args()
    directory = REPO / "webapp/static/showcase"
    sources = {row["dataset_id"]: row for row in json.loads((directory / "sources.json").read_text())}
    existing = directory / "rendering.json"
    settings = {row["dataset_id"]: row for row in json.loads(existing.read_text())} if existing.exists() else {}
    for case in SHOWCASE:
        dataset = case["id"]
        if args.dataset and dataset not in args.dataset:
            continue
        if not args.shapes_only:
            curves = json.loads((directory / f"{dataset}-curves.json").read_text())
            draw_fsc(curves, case["resolution"], directory / f"{dataset}-fsc.png")
        settings[dataset] = {"dataset_id": dataset, **draw_shape(sources[dataset]["map_cache"], directory / f"{dataset}-shape.png")}
        existing.write_text(json.dumps(list(settings.values()), indent=2))
        print(f"Rendered {dataset}: themed FSC and single-view density isosurface", flush=True)


if __name__ == "__main__":
    main()
