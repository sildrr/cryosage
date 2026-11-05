import struct

import numpy as np

from reference_projection_metrics import (
    _projection_bank,
    compute_reference_projection_metrics,
    resolve_reference_volume,
)


def _write_mrc(path, volume, voxel=1.0):
    nz, ny, nx = volume.shape
    header = bytearray(1024)
    struct.pack_into("<4i", header, 0, nx, ny, nz, 2)
    struct.pack_into("<3i", header, 28, nx, ny, nz)
    struct.pack_into("<3f", header, 40, nx * voxel, ny * voxel, nz * voxel)
    struct.pack_into("<3i", header, 64, 1, 2, 3)
    struct.pack_into("<i", header, 92, 0)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(np.asarray(volume, dtype="<f4").tobytes())


def test_reference_map_resolver_prefers_matching_full_dataset_particle_count(tmp_path):
    root = tmp_path / "mapDB"
    (root / "byEMDB" / "6287").mkdir(parents=True)
    (root / "recon.tsv").write_text(
        "empiar\temdb\tpdb\ttype\tsize\tres\tres2\tnumber\tbox\tpid\tjid\n"
        "10025\t6287\tNA\tNoMod\t2048\t2.8\t2.58\t1000\t400\tP1\tJ0\n"
    )
    (root / "byEMDB" / "6287" / "3.0_1000_a.mrc").touch()
    (root / "byEMDB" / "6287" / "2.9_1000_b.mrc").touch()
    (root / "byEMDB" / "6287" / "2.0_500_c.mrc").touch()
    found = resolve_reference_volume("EMPIAR-10025", "EMD-6287", mapdb_root=root)
    assert found["available"] is True
    assert found["used_exact_particle_count_match"] is True
    assert found["map_particle_count"] == 1000
    assert found["map_resolution_A"] == 2.9
    assert found["path"].endswith("2.9_1000_b.mrc")


def test_projection_consistency_matches_a_projection_of_reference_map(tmp_path):
    axis = np.linspace(-1, 1, 32, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    volume = (
        np.exp(-((x / 0.45) ** 2 + (y / 0.22) ** 2 + (z / 0.3) ** 2) * 3)
        + 0.5 * np.exp(-(((x - 0.25) / 0.12) ** 2 + ((y + 0.2) / 0.15) ** 2 + ((z - 0.15) / 0.2) ** 2) * 5)
    ).astype(np.float32)
    map_path = tmp_path / "reference.mrc"
    _write_mrc(map_path, volume)
    bank, _ = _projection_bank(volume, size=32, view_count=8, inplane_steps=4)
    class_average = bank[0].reshape(32, 32)[None, ...]
    result = compute_reference_projection_metrics(
        map_path,
        class_average,
        [{"class_id": 0, "particle_count": 500}],
        class_pixel_size_A=1.0,
        view_count=8,
        inplane_steps=4,
    )
    assert result["status"] == "completed"
    assert result["particle_weighted_projection_consistency_ncc"] > 0.99
    assert result["particle_weighted_consistent_class_fraction"] == 1.0
