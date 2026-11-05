#!/usr/bin/env python3
"""Make centered, low-pass MRC maps for an unbiased rigid-body map comparison."""

import argparse
import json
from pathlib import Path

import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def prepare(source: Path, destination: Path, grid_size: int, spacing: float) -> dict:
    with mrcfile.open(source, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
        voxel_size = np.array(
            [mrc.voxel_size.z, mrc.voxel_size.y, mrc.voxel_size.x], dtype=float
        )
        if not np.all(np.isfinite(voxel_size)) or np.any(voxel_size <= 0):
            raise ValueError(f"Invalid voxel size in {source}: {voxel_size}")
        threshold = np.percentile(data, 90)
        weight = np.maximum(data - threshold, 0)
        center = np.array(
            np.unravel_index(np.argmax(data), data.shape), dtype=float
        )
        if weight.sum() > 0:
            center = np.array(
                [np.sum(weight, axis=tuple(j for j in range(3) if j != i)).dot(
                    np.arange(data.shape[i])
                ) / weight.sum() for i in range(3)]
            )
        sigma = np.maximum(0.0, 4.0 / voxel_size)
        smooth = gaussian_filter(data, sigma=sigma, mode="constant")
        axis = (np.arange(grid_size, dtype=np.float32) - (grid_size - 1) / 2) * spacing
        coordinates = np.meshgrid(*[center[i] + axis / voxel_size[i] for i in range(3)], indexing="ij")
        sample = map_coordinates(smooth, coordinates, order=1, mode="constant", cval=0)
        sample = sample.astype(np.float32)
        # Suppress the noisy solvent background without changing the particle envelope.
        cutoff = np.percentile(sample[sample > 0], 35) if np.any(sample > 0) else 0
        sample[sample < cutoff] = 0
        with mrcfile.new(destination, overwrite=True) as output:
            output.set_data(sample)
            output.voxel_size = spacing
        return {
            "path": str(source),
            "shape_zyx": list(data.shape),
            "voxel_size_zyx_A": voxel_size.tolist(),
            "density_center_zyx": center.tolist(),
            "prepared_map": str(destination),
            "positive_fraction": float(np.mean(sample > 0)),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("moving", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--spacing", type=float, default=3.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "moving": prepare(args.moving, args.output_dir / "moving_lowpass.mrc", args.grid_size, args.spacing),
        "reference": prepare(args.reference, args.output_dir / "reference_lowpass.mrc", args.grid_size, args.spacing),
        "comparison_grid_size": args.grid_size,
        "comparison_spacing_A": args.spacing,
        "lowpass_sigma_A": 4.0,
    }
    (args.output_dir / "map_metadata.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
