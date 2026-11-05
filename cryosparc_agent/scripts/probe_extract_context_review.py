"""Read-only matched extraction review with full-field context and ablations."""

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cryosparc_client import cryosparc_client
from model_direct_runner import resolve_api_key, run_openai_compatible_model
from vision_inputs import render_full_micrograph_preview, full_field_pick_position
from probe_extract_box_review import read_slice, render_sheet


def match_coordinate(dataset, micrograph_uid, center):
    indices = np.flatnonzero(np.asarray(dataset["location/micrograph_uid"]) == micrograph_uid)
    if not len(indices):
        raise ValueError("No matching micrograph in comparison extraction")
    dx = np.asarray(dataset["location/center_x_frac"])[indices] - center[0]
    dy = np.asarray(dataset["location/center_y_frac"])[indices] - center[1]
    distances = np.hypot(dx, dy)
    index = int(np.argmin(distances))
    if distances[index] > 1e-7 or np.sum(distances <= 1e-7) != 1:
        raise ValueError("Comparison extraction has no unique identical pick coordinate")
    return int(indices[index])


def neighbor_geometry(points_native, center_native, sizes):
    delta = np.asarray(points_native, dtype=float) - np.asarray(center_native, dtype=float)
    distance = np.linalg.norm(delta, axis=1)
    other = distance > 1.0
    return {
        "extra_picked_centers_by_box_px": {
            str(size): int(np.sum(other & (np.max(np.abs(delta), axis=1) < size / 2)))
            for size in sizes
        },
        "nearest_other_picked_center_distance_native_px": float(np.min(distance[other])) if other.any() else None,
    }


def verified_history(progress_path, extract_id, refine_id):
    nodes = {n["id"]: n for n in json.loads(progress_path.read_text())["nodes"]}
    visited, pending = set(), [refine_id]
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(nodes.get(node, {}).get("parents", []))
    if extract_id not in visited:
        raise ValueError("Historical refinement is not a descendant of the baseline extraction")
    extract, refine = nodes[extract_id], nodes[refine_id]
    if extract["status"] != "completed" or refine["status"] != "completed":
        raise ValueError("Historical jobs must be completed")
    return {
        "extract_job": extract_id, "box_size_pix": extract["parameters"]["box_size_pix"],
        "refine_job": refine_id, "corrected_FSC_resolution_A": refine["resolution_A"],
        "causal_limitations": [
            "This is a different picking/selection population, not a controlled box-only quality comparison.",
            "A historical resolution establishes a useful baseline, not optimality or proof of completeness.",
        ],
    }


def build(args):
    meta = json.loads((args.previous_evidence / "metadata.json").read_text())
    project = cryosparc_client().find_project(meta["project"])
    root = Path(str(project.dir()))
    comparison_job = project.find_job(args.comparison_job)
    if comparison_job.doc.get("status") != "completed":
        raise ValueError("Comparison extraction must be completed")
    comparison = comparison_job.load_output("particles")
    picks = project.find_job(meta["source_picks"]).load_output("particles")
    mic_uids = np.asarray(picks["location/micrograph_uid"])
    x = np.asarray(picks["location/center_x_frac"], dtype=float)
    y = np.asarray(picks["location/center_y_frac"], dtype=float)
    sizes = [meta["original_box_px"], meta["comparison_box_px"]]
    psize = meta["native_pixel_size_A"]
    original_images, comparison_images, samples = [], [], []
    for sample in meta["samples"]:
        index = match_coordinate(comparison, sample["micrograph_uid"], sample["center_fraction"])
        actual = read_slice(root / str(comparison["blob/path"][index]), int(comparison["blob/idx"][index]))
        if actual.shape != (sizes[1], sizes[1]) or not np.isclose(float(comparison["blob/psize_A"][index]), psize):
            raise ValueError("Comparison shape/sampling differs from requested native box")
        original = read_slice(Path(sample["stack_path"]), sample["stack_index"])
        if original.shape != (sizes[0], sizes[0]):
            raise ValueError("Original stack geometry differs from saved metadata")
        original_images.append(original[::-1].copy())
        comparison_images.append(actual[::-1].copy())
        row = dict(sample)
        matching = np.flatnonzero(mic_uids == sample["micrograph_uid"])
        shape = np.asarray(picks["location/micrograph_shape"][matching[0]], dtype=float)
        points = np.column_stack((x[matching] * shape[1], y[matching] * shape[0]))
        center = np.asarray(sample["center_fraction"]) * shape[::-1]
        row.update(neighbor_geometry(points, center, sizes))
        row.update(comparison_job=args.comparison_job, comparison_particle_row=index,
                   comparison_stack_path=str(root / str(comparison["blob/path"][index])),
                   comparison_stack_index=int(comparison["blob/idx"][index]))
        samples.append(row)
    ids = [s["sample_id"] for s in samples]
    names = []
    for name, images, lowpass in [
        ("actual_640_raw.png", original_images, None),
        ("actual_640_lowpass8A.png", original_images, 8),
        ("actual_440_raw.png", comparison_images, None),
        ("actual_440_lowpass8A.png", comparison_images, 8),
    ]:
        render_sheet(images, ids, psize, lowpass).save(args.output_dir / name)
        names.append(name)

    groups = sorted({s["micrograph_uid"] for s in samples})
    mic_metadata = []
    for group_index, uid in enumerate(groups):
        subset = [s for s in samples if s["micrograph_uid"] == uid]
        indices = np.flatnonzero(mic_uids == uid)
        if not np.isclose(float(picks["location/micrograph_psize_A"][indices[0]]), psize):
            raise ValueError("This native-sampling pilot cannot mix binned particles and native micrographs")
        mic_path = root / str(picks["location/micrograph_path"][indices[0]])
        mic = read_slice(mic_path, 0)
        overview, info = render_full_micrograph_preview(mic, psize, size=1024)
        draw = ImageDraw.Draw(overview)
        height, width = info["image_shape_pixels"]
        sx, sy = width / mic.shape[1], height / mic.shape[0]
        for sample in subset:
            cx, cy = full_field_pick_position(*sample["center_fraction"], info)
            dx = (x[indices] - sample["center_fraction"][0]) * mic.shape[1]
            dy = (y[indices] - sample["center_fraction"][1]) * mic.shape[0]
            near = (np.max(np.abs(np.column_stack((dx, dy))), axis=1) < max(sizes)/2) & (np.hypot(dx, dy) > 1)
            for i in indices[near]:
                px, py = full_field_pick_position(x[i], y[i], info)
                draw.line((px - 2, py, px + 2, py), fill="yellow", width=1)
                draw.line((px, py - 2, px, py + 2), fill="yellow", width=1)
            for size, color in zip(sizes, ("magenta", "cyan")):
                hx, hy = size * sx / 2, size * sy / 2
                draw.rectangle((cx-hx, cy-hy, cx+hx, cy+hy), outline=color, width=1)
            draw.ellipse((cx-2, cy-2, cx+2, cy+2), outline="white", width=1)
            draw.text((cx + max(sizes)*sx/2 + 3, cy - 6), sample["sample_id"], fill="white", stroke_width=1, stroke_fill="black")
        canvas = Image.new("RGB", (1024, 1060), "black")
        canvas.paste(overview, (0, 36))
        legend = ImageDraw.Draw(canvas)
        legend.text((8, 5), f"M{group_index+1}: {', '.join(s['sample_id'] for s in subset)} | full micrograph, Y upward", fill="white")
        legend.text((8, 20), f"magenta={sizes[0]}px; cyan={sizes[1]}px; white=target; yellow=other picked center (not verified particle)", fill="white")
        name = f"full_micrograph_{group_index+1}_marked.png"
        canvas.save(args.output_dir / name)
        names.append(name)
        mic_metadata.append({"image": name, "uid": uid, "path": str(mic_path), "display": info,
                             "sample_ids": [s["sample_id"] for s in subset]})
    history = verified_history(args.progress, args.baseline_extract, args.baseline_refine)
    bundle = {
        "schema_version": "extract-context-review-pilot-1", "project": meta["project"],
        "source_picks": meta["source_picks"], "original_job": meta["extract_job"],
        "comparison_job": args.comparison_job, "native_pixel_size_A": psize,
        "box_sizes_px": sizes, "samples": samples, "historical_baseline": history,
        "images": names, "micrograph_images": mic_metadata,
        "sampling": meta["sampling"], "submitted_jobs": False,
        "display": "Entire boxes and full micrographs, no micrograph crops; Y upward in all new panels. Filtered particle views use Gaussian FWHM 8 A; native arrays are unchanged.",
    }
    (args.output_dir / "evidence.json").write_text(json.dumps(bundle, indent=2))
    return bundle


def decision_facts(bundle, include_history, comparison_outputs=True):
    facts = {k: bundle[k] for k in ["native_pixel_size_A", "box_sizes_px", "sampling", "display"]}
    facts["matched_samples"] = [{k: s[k] for k in [
        "sample_id", "extra_picked_centers_by_box_px", "nearest_other_picked_center_distance_native_px",
    ]} for s in bundle["samples"]]
    facts["statistics_limitations"] = [
        "Picked centers are not verified molecular objects; false positives and missed particles are possible.",
        "Neighbor counts necessarily decrease with a smaller field and do not by themselves establish optimal box size.",
        ("Both jobs use identical pick centers for these samples; they are actual matched outputs, not artificial crops."
         if comparison_outputs else
         "Only current 640 px particle images are supplied. The 440 px overlay and neighbor counts describe a candidate geometric window, not observed 440 px extraction results."),
    ]
    if include_history:
        facts["historical_baseline"] = bundle["historical_baseline"]
    return facts


def probe(args, bundle):
    from webapp.manage import apply_runtime_profile
    from webapp.proxy import configure_compute_proxy
    apply_runtime_profile()
    configure_compute_proxy()
    key = resolve_api_key()
    instruction = (
        "Assess extraction-box adequacy using the supplied matched real particle outputs, "
        "full-field micrograph context, calibrated scales and picked-center statistics. "
        "Do not assume either size is correct. Separate visible molecular objects from automatic "
        "pick markers, noise and CTF textures. A box can legitimately contain neighboring density; "
        "check central-target completeness and necessary margin, not isolation alone. "
        "The full micrographs are uncropped and labelled with the corresponding sample IDs; magenta boxes "
        "are 640 px, cyan 440 px, white centers are targets and yellow crosses are additional "
        "automatic pick centers, NOT expert-verified particle identities. All displays are Y upward. "
        "Historical FSC on a different particle population, when present, is a baseline, not proof "
        "that a particular box is optimal. You may abstain. No CryoSPARC changes are authorized. "
        "Return JSON only, descriptive strings in Chinese: recommendation "
        "(maintain_640|prefer_440|other_size|insufficient_evidence), recommended_box_px or null, "
        "central_target_completeness (assessment, supporting_sample_ids, confidence), "
        "neighbor_signal_assessment, per_sample (sample_id, visible_observation, "
        "picked_center_statistics_interpretation, confidence), "
        "evidence_basis (visual, coordinate_statistics, historical_baseline), "
        "rationale, uncertainty, and further_validation_needed. Be explicit which observations "
        "are genuinely visual and which are inherited numeric facts."
    )
    for label, images, history in [
        ("visual_statistics", True, False),
        ("visual_statistics_history", True, True),
        ("statistics_history_only", False, True),
        ("early_output_context", True, False),
    ]:
        if args.condition != "all" and args.condition != label:
            continue
        destination = args.output_dir / (label + "_result.json")
        if destination.exists():
            print(json.dumps({"test": label, "skipped": "existing saved response"}), flush=True)
            continue
        early = label == "early_output_context"
        prompt = instruction + "\nEvidence facts:\n" + json.dumps(decision_facts(bundle, history, not early))
        image_names = bundle["images"]
        if early:
            image_names = image_names[:2] + image_names[4:]
            prompt += (
                "\nEARLY-OUTPUT CONDITION: The first two images are the current 640 px raw "
                "and filtered extraction sheets, followed by six full micrographs. NO 440 px "
                "particle outputs or historical FSC results are supplied. The cyan 440 px "
                "outlines are only a proposed alternative field. Do not claim to have inspected "
                "440 px extracted particles. Judge whether the current view is justified and "
                "whether a smaller window is worth testing from the available context alone. "
                "You may abstain; distinguish a proposal from demonstrated improvement."
            )
        elif images:
            prompt += (
                "\nThe first four images are 640 raw, 640 filtered, 440 raw, 440 filtered, "
                "followed by six full micrographs. These show 12 distinct pick centers, not "
                "48 independent particles. Both sizes are actual matched extraction outputs."
            )
        if not images:
            prompt += (
                "\nCONTROL CONDITION: NO images are attached. The image legend above describes "
                "the paired experiment only; it does not mean you have seen those images. "
                "Set visual evidence to unavailable, do not invent per-sample observations, and "
                "distinguish a numeric/history-based suggestion from a visual completeness check."
            )
        content = [{"type": "text", "text": prompt}]
        if images:
            for name in image_names:
                encoded = base64.b64encode((args.output_dir / name).read_bytes()).decode()
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + encoded, "detail": "high",
                }})
        (args.output_dir / (label + "_prompt.txt")).write_text(prompt)
        start = time.monotonic()
        result = run_openai_compatible_model(
            [{"role": "user", "content": content}], os.environ["OPENAI_BASE_URL"], key,
            os.environ["OPENAI_MODEL"], max_new_tokens=6144, timeout_seconds=240, max_retries=0,
        )
        result["elapsed_seconds"] = round(time.monotonic() - start, 2)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (args.output_dir / (label + "_output.txt")).write_text(result["raw_text"])
        print(json.dumps({"test": label, "elapsed_seconds": result["elapsed_seconds"],
                          "finish_reason": result["raw_response"]["choices"][0].get("finish_reason"),
                          "output": result["raw_text"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-evidence", type=Path, required=True)
    parser.add_argument("--comparison-job", default="J607")
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--baseline-extract", default="J592")
    parser.add_argument("--baseline-refine", default="J602")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submit-api", action="store_true")
    parser.add_argument("--condition", choices=(
        "all", "visual_statistics", "visual_statistics_history", "statistics_history_only", "early_output_context",
    ), default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "evidence.json"
    bundle = json.loads(path.read_text()) if path.exists() else build(args)
    if bundle["comparison_job"] != args.comparison_job or (
        bundle["historical_baseline"]["extract_job"], bundle["historical_baseline"]["refine_job"]
    ) != (args.baseline_extract, args.baseline_refine):
        raise ValueError("Saved evidence does not match the requested comparison or baseline jobs")
    if args.submit_api:
        probe(args, bundle)
    else:
        print(json.dumps({"output_dir": str(args.output_dir), "images": len(bundle["images"]),
                          "samples": len(bundle["samples"])}))


if __name__ == "__main__":
    main()
