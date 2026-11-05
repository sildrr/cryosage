"""Reference-map projection matching for offline Class 2D feedback."""
from __future__ import annotations

import math
import os
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MAPDB_ROOT = Path("/hdd2/cryosparc/mapDB")


def _numeric_id(value: Any) -> str | None:
    match = re.search(r"\d+", str(value or ""))
    return match.group(0) if match else None


def resolve_reference_volume(
    empiar_id: str | None,
    emdb_id: str | None,
    explicit_path: str | None = None,
    mapdb_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve an explicit map, direct EMDB map, or matching mapDB reconstruction."""
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if path.is_file():
            return {"available": True, "path": str(path), "source": "explicit"}
        return {"available": False, "reason": "explicit_reference_map_missing", "path": str(path)}

    emdb = _numeric_id(emdb_id)
    empiar = _numeric_id(empiar_id)
    if not emdb:
        return {"available": False, "reason": "missing_emdb_id"}
    root = Path(mapdb_root or os.getenv("CRYOAGENT_MAPDB_ROOT", str(DEFAULT_MAPDB_ROOT)))
    direct_candidates = [root / "reproduce3D" / f"{emdb}.mrc", root / "reproduce3D" / f"{emdb}.map"]
    direct = next((path for path in direct_candidates if path.is_file()), None)
    if direct:
        return {"available": True, "path": str(direct), "source": "reproduce3D_emdb_id"}

    target_particles: int | None = None
    target_resolution: float | None = None
    recon_index = root / "recon.tsv"
    if empiar and recon_index.is_file():
        for line in recon_index.read_text(errors="replace").splitlines()[1:]:
            fields = line.split("\t")
            if len(fields) < 8 or fields[0] != empiar:
                continue
            if fields[1] != emdb:
                return {"available": False, "reason": "empiar_emdb_index_mismatch", "indexed_emdb_id": fields[1]}
            try:
                target_resolution = float(fields[5])
                target_particles = int(float(fields[7]))
            except (ValueError, TypeError):
                pass
            break

    candidates: list[tuple[float, int, Path]] = []
    emdb_dir = root / "byEMDB" / emdb
    if emdb_dir.is_dir():
        for path in emdb_dir.glob("*.mrc"):
            match = re.match(r"([0-9]+(?:\.[0-9]+)?)_(\d+)_", path.name)
            if not match:
                continue
            try:
                candidates.append((float(match.group(1)), int(match.group(2)), path))
            except ValueError:
                continue
    if not candidates:
        return {"available": False, "reason": "no_reference_volume_found", "emdb_id": emdb}

    exact_count = [item for item in candidates if target_particles and item[1] == target_particles]
    pool = exact_count or candidates
    # Prefer the best resolution among maps produced from the full dataset when
    # that count is indexed; otherwise prefer the best-resolution local map.
    resolution, particle_count, selected = min(pool, key=lambda item: (item[0], -item[1], item[2].name))
    return {
        "available": True,
        "path": str(selected),
        "source": "byEMDB_matching_full_particle_count" if exact_count else "byEMDB_best_available",
        "emdb_id": emdb,
        "target_particle_count": target_particles,
        "map_particle_count": particle_count,
        "map_resolution_A": resolution,
        "target_resolution_A": target_resolution,
        "candidate_count": len(candidates),
        "used_exact_particle_count_match": bool(exact_count),
    }


def read_mrc_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read common MRC modes without requiring an optional mrcfile package."""
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(1024)
        if len(header) != 1024:
            raise ValueError(f"MRC header is truncated: {path}")
        candidates = []
        for endian in ("<", ">"):
            nx, ny, nz, mode = struct.unpack_from(endian + "4i", header, 0)
            if 0 < nx <= 32768 and 0 < ny <= 32768 and 0 < nz <= 32768 and mode in (0, 1, 2, 6):
                candidates.append((endian, nx, ny, nz, mode))
        if not candidates:
            raise ValueError(f"Unsupported MRC header dimensions or mode: {path}")
        endian, nx, ny, nz, mode = candidates[0]
        mx, my, mz = struct.unpack_from(endian + "3i", header, 28)
        cell_x, cell_y, cell_z = struct.unpack_from(endian + "3f", header, 40)
        mapc, mapr, maps = struct.unpack_from(endian + "3i", header, 64)
        nsymbt = struct.unpack_from(endian + "i", header, 92)[0]
        dtype = {0: "i1", 1: endian + "i2", 2: endian + "f4", 6: endian + "u2"}[mode]
        stream.seek(1024 + max(0, nsymbt))
        data = np.fromfile(stream, dtype=np.dtype(dtype), count=nx * ny * nz)
    if data.size != nx * ny * nz:
        raise ValueError(f"MRC data is truncated: expected {nx * ny * nz}, found {data.size}")
    volume = data.reshape((nz, ny, nx)).astype(np.float32, copy=False)
    # Convert section,row,column axes to z,y,x when the MRC axis mapping is set.
    if sorted((mapc, mapr, maps)) == [1, 2, 3] and (mapc, mapr, maps) != (1, 2, 3):
        axis_for_world = {maps: 0, mapr: 1, mapc: 2}
        volume = np.transpose(volume, tuple(axis_for_world[i] for i in (3, 2, 1)))
    voxel = (
        cell_x / mx if mx > 0 and cell_x > 0 else 1.0,
        cell_y / my if my > 0 and cell_y > 0 else 1.0,
        cell_z / mz if mz > 0 and cell_z > 0 else 1.0,
    )
    return volume, voxel


def _sample_trilinear(volume: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    nz, ny, nx = volume.shape
    valid = (x >= 0) & (x < nx - 1) & (y >= 0) & (y < ny - 1) & (z >= 0) & (z < nz - 1)
    result = np.zeros(x.shape, dtype=np.float32)
    if not np.any(valid):
        return result
    xv, yv, zv = x[valid], y[valid], z[valid]
    x0, y0, z0 = xv.astype(np.int32), yv.astype(np.int32), zv.astype(np.int32)
    dx, dy, dz = xv - x0, yv - y0, zv - z0
    c000 = volume[z0, y0, x0]
    c100 = volume[z0, y0, x0 + 1]
    c010 = volume[z0, y0 + 1, x0]
    c110 = volume[z0, y0 + 1, x0 + 1]
    c001 = volume[z0 + 1, y0, x0]
    c101 = volume[z0 + 1, y0, x0 + 1]
    c011 = volume[z0 + 1, y0 + 1, x0]
    c111 = volume[z0 + 1, y0 + 1, x0 + 1]
    c00 = c000 * (1 - dx) + c100 * dx
    c10 = c010 * (1 - dx) + c110 * dx
    c01 = c001 * (1 - dx) + c101 * dx
    c11 = c011 * (1 - dx) + c111 * dx
    c0 = c00 * (1 - dy) + c10 * dy
    c1 = c01 * (1 - dy) + c11 * dy
    result[valid] = c0 * (1 - dz) + c1 * dz
    return result


def _directions(count: int) -> list[np.ndarray]:
    # Fibonacci sampling on one hemisphere avoids counting antipodal views twice.
    golden = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for index in range(count * 2):
        z = (index + 0.5) / count - 1.0
        if z < 0:
            continue
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        point = np.array([radius * math.cos(index * golden), radius * math.sin(index * golden), z])
        points.append(point)
        if len(points) == count:
            break
    return points


def _projection_bank(volume: np.ndarray, size: int, view_count: int, inplane_steps: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="xy")
    center = (np.array(volume.shape[::-1], dtype=np.float32) - 1) / 2
    directions = _directions(view_count)
    templates = []
    template_views = []
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for view_id, normal in enumerate(directions):
        u = np.cross(ref, normal)
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        coords = (
            center[0] + xx * u[0] + yy * v[0] + zz * normal[0],
            center[1] + xx * u[1] + yy * v[1] + zz * normal[1],
            center[2] + xx * u[2] + yy * v[2] + zz * normal[2],
        )
        rotated = _sample_trilinear(volume, *coords)
        projection = rotated.sum(axis=0)
        image = Image.fromarray(projection.astype(np.float32), mode="F")
        for angle_index in range(inplane_steps):
            angle = 360.0 * angle_index / inplane_steps
            rotated_2d = np.asarray(image.rotate(angle, resample=Image.Resampling.BILINEAR), dtype=np.float32)
            templates.append(rotated_2d)
            template_views.append(view_id)
            templates.append(np.fliplr(rotated_2d))
            template_views.append(view_id)
    bank = np.stack(templates).astype(np.float32)
    bank -= bank.mean(axis=(1, 2), keepdims=True)
    norms = np.linalg.norm(bank.reshape(len(bank), -1), axis=1, keepdims=True)
    bank /= np.maximum(norms[:, :, None], 1e-8)
    return bank.reshape(len(bank), -1), np.asarray(template_views, dtype=np.int32)


def compute_reference_projection_metrics(
    reference_map_path: str | Path,
    class_average_stack: np.ndarray,
    class_statistics: list[dict[str, Any]],
    *,
    class_pixel_size_A: float | None = None,
    view_count: int = 42,
    inplane_steps: int = 12,
    min_consistency_ncc: float = 0.20,
) -> dict[str, Any]:
    """Match class averages against a coarse bank of reference-map projections."""
    volume, voxel_size = read_mrc_volume(reference_map_path)
    images = np.asarray(class_average_stack, dtype=np.float32)
    if images.ndim == 2:
        images = images[None, ...]
    if images.ndim != 3:
        raise ValueError(f"Expected 2D class-average stack, received {images.shape}")
    count, height, width = images.shape
    size = min(height, width, 64)
    size = max(16, size - size % 2)
    psize = float(class_pixel_size_A or max(voxel_size))
    map_psize = float(np.mean(voxel_size))
    physical_box_A = max(height, width) * psize
    wanted_side = max(size, int(round(physical_box_A / map_psize)))
    wanted_side = min(min(volume.shape), wanted_side)
    wanted_side -= wanted_side % size
    wanted_side = max(size, wanted_side)
    starts = [(dim - wanted_side) // 2 for dim in volume.shape]
    crop = volume[
        starts[0]:starts[0] + wanted_side,
        starts[1]:starts[1] + wanted_side,
        starts[2]:starts[2] + wanted_side,
    ]
    if crop.shape != (wanted_side, wanted_side, wanted_side):
        raise ValueError("Reference map could not be cropped to a cubic particle box")
    # Block-average the reference to the class-average sampling grid.
    block_size = wanted_side // size
    coarse = crop.reshape(
        size, block_size, size, block_size, size, block_size,
    ).mean(axis=(1, 3, 5), dtype=np.float32)
    coarse -= float(coarse.mean())
    std = float(coarse.std())
    if not np.isfinite(std) or std <= 1e-8:
        raise ValueError("Reference map has no usable density variation")
    coarse /= std
    bank, template_views = _projection_bank(coarse, size, view_count, inplane_steps)
    resized = []
    for image in images:
        pil = Image.fromarray(image.astype(np.float32), mode="F")
        image_small = np.asarray(
            pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32,
        ).copy()
        image_small -= float(image_small.mean())
        norm = float(np.linalg.norm(image_small))
        resized.append(image_small.reshape(-1) / max(norm, 1e-8))
    class_matrix = np.stack(resized, axis=1)
    correlations = np.abs(bank @ class_matrix)
    best_template = np.argmax(correlations, axis=0)
    best_ncc = correlations[best_template, np.arange(count)].clip(0, 1)
    best_views = template_views[best_template]
    stats_by_id = {int(item["class_id"]): item for item in class_statistics if "class_id" in item}
    weights = np.asarray([
        max(0, int(stats_by_id.get(class_id, {}).get("particle_count", 0)))
        for class_id in range(count)
    ], dtype=np.float64)
    total_weight = float(weights.sum())
    weighted_ncc = float(np.average(best_ncc, weights=weights)) if total_weight else None
    consistent = best_ncc >= min_consistency_ncc
    consistency_coverage = float(weights[consistent].sum() / total_weight) if total_weight else None
    valid_views = best_views[(weights > 0) & consistent]
    valid_weights = weights[(weights > 0) & consistent]
    if valid_views.size:
        bins = np.bincount(valid_views, weights=valid_weights, minlength=view_count)
        probabilities = bins[bins > 0] / bins.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        normalized_entropy = entropy / math.log(view_count) if len(probabilities) > 1 and view_count > 1 else 0.0
        effective_views = float(math.exp(entropy))
    else:
        entropy = normalized_entropy = effective_views = None
    return {
        "status": "completed",
        "metric_version": "reference-projection-v1",
        "reference_map_path": str(Path(reference_map_path).resolve()),
        "reference_map_voxel_size_A": list(voxel_size),
        "class_average_count": int(count),
        "projection_grid_size": int(size),
        "view_count": int(view_count),
        "inplane_rotations_per_view": int(inplane_steps),
        "min_consistency_ncc": float(min_consistency_ncc),
        "particle_weighted_projection_consistency_ncc": weighted_ncc,
        "particle_weighted_consistent_class_fraction": consistency_coverage,
        "view_diversity": {
            "particle_weighted_view_entropy": entropy,
            "normalized_view_entropy": normalized_entropy,
            "effective_view_count": effective_views,
            "matched_class_count": int(valid_views.size),
        },
        "per_class": [
            {
                "class_id": int(class_id),
                "particle_count": int(weights[class_id]),
                "best_projection_ncc": float(best_ncc[class_id]),
                "best_view_id": int(best_views[class_id]),
                "matched_reference_projection": bool(consistent[class_id]),
            }
            for class_id in range(count)
        ],
        "limitations": [
            "Offline reference-map metric; do not expose the reference map or scores to the policy prompt.",
            "Coarse projection search is a morphology-consistency proxy, not a calibrated truncation probability.",
            "View entropy is approximate and depends on the angular projection grid and NCC threshold.",
        ],
    }
