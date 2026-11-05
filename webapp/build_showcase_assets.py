"""Export approved numeric FSC data and maps, then render homepage-only previews."""
import json
import io
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from webapp.core import REPO, SHOWCASE
from webapp.dashboard import client, event_images


def fsc_data(cs, project, job_uid):
    job = cs.find_job(project, job_uid)
    info = job.outputs["volume"].latest_summary_stats.get("fsc_info")
    curves = []
    if info:
        for key, label in (("fsc_autotight_noisesub", "Corrected mask"), ("fsc_resmask", "Resolution mask"), ("fsc_nomask", "Unmasked")):
            values = info.get(key)
            if values:
                shells = info.get("radwns") or (np.arange(len(values)) + .5).tolist()
                frequency = (np.array(shells[:len(values)]) / (info["N"] * info["psize"])).tolist()
                curves.append({"field": key, "label": label, "frequency_inv_A": frequency, "values": values})
    else:
        # Validation Jobs store their actual curves in XML, not volume summary arrays.
        for suffix, label in (("mask", "Input mask"), ("sphericalmask", "Spherical mask"), ("nomask", "Unmasked")):
            remote = f"{job_uid}/{job_uid}_fsc_{suffix}.xml"
            buffer = io.BytesIO()
            cs.find_project(project).download_file(remote, buffer)
            root = ET.fromstring(buffer.getvalue())
            coordinates = root.findall("coordinate")
            curves.append({"field": remote, "label": label,
                           "frequency_inv_A": [float(c.findtext("x")) for c in coordinates],
                           "values": [float(c.findtext("y")) for c in coordinates]})
    return curves


def main():
    destination = REPO / "webapp/static/showcase"
    cache = Path("/hdd1/linweifan/cryosparc/cryoagent_web/diagnostics/showcase_final_maps")
    destination.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    cs = client()
    manifest = []
    for case in SHOWCASE:
        project, uid = case["project"], case["job"]
        job = cs.find_job(project, uid)
        if job.status != "completed":
            raise ValueError(f"Showcase job is not completed: {project}/{uid}")
        # 11280's published headline is the independent J118 validation of J117.
        fsc_job = "J118" if case["id"] == "11280" else uid
        images = event_images(project, fsc_job, 0)
        fsc = next(image for image in images if image["caption"].startswith(("FSC Iteration", "Gold Standard FSC")))
        target = destination / f"{case['id']}-fsc.png"
        raw = b"".join(cs.api.assets.download(fsc["asset_id"]).stream())
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Expected original FSC PNG")
        target.write_bytes(raw)
        curves = fsc_data(cs, project, fsc_job)
        (destination / f"{case['id']}-curves.json").write_text(json.dumps(curves))
        volume = job.load_output("volume")
        field = "map_sharp/path" if "map_sharp/path" in volume.fields() else "map/path"
        source = str(volume[field][0])
        map_file = cache / f"{project}_{uid}.mrc"
        if not map_file.exists():
            cs.find_project(project).download_file(source, map_file)
        manifest.append({"dataset_id": case["id"], "project": project, "workspace": case["workspace"],
                         "job": uid, "fsc_job": fsc_job, "fsc_asset": fsc, "map_field": field, "map_source": source,
                         "map_cache": str(map_file), "preview": "Single oblique isosurface; see rendering.json for display settings. Not structural validation.",
                         "curve_source": f"{case['id']}-curves.json"})
        print(f"Exported {case['id']}: {project}/{uid}", flush=True)
    (destination / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    python = os.getenv("CRYOAGENT_SHOWCASE_RENDER_PYTHON", "/ssd1/linweifan/cryosparc/cryosparc_worker/.pixi/envs/worker/bin/python")
    subprocess.run([python, "-m", "webapp.render_showcase_assets"], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
