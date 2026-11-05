"""One isolated visual API estimate; never create or queue CryoSPARC jobs."""
import argparse
import base64
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from model_direct_runner import resolve_api_key, run_openai_compatible_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-6.1-sol")
    args = parser.parse_args()
    payload = args.image.read_bytes()
    prompt = (
        "Estimate particle size and a sensible initial CryoSPARC extraction box using only "
        "this full-field micrograph and its calibrated scale bar. No dataset identity, "
        "known molecule, reference diameter, historical settings, or candidate box list "
        "is supplied. The whole native image is 7676 rows by 7420 columns at 0.6575 A/pixel; "
        "it was area-downsampled to 1024 rows by 990 columns, placed in a 1024x1024 canvas "
        "with 17-pixel horizontal padding on each side. Contrast is mean +/-3 SD after "
        "downsampling; Y is upward; there is no additional Gaussian filter. The scale bar "
        "is approximately 1000 A (100 nm), 203 displayed pixels. Do not confuse native "
        "pixels with displayed pixels or mistake aggregated groups/ice for single particles. "
        "Estimate a typical single-particle diameter range and largest apparent dimension "
        "from several examples. Give 2-3 approximate example centers in the displayed "
        "1024x1024 image with x measured from left and y from top, and apparent width/height "
        "in displayed pixels, so the visual estimate is auditable. Convert sizes to A, then "
        "recommend an initial native extraction box (integer multiple of 8) with margin. "
        "Defocus, CTF fit resolution and target resolution are NOT available, so do not "
        "claim a physically optimal CTF-delocalization box or invent those values. "
        "If the evidence is insufficient, say so rather than forcing a confident answer. "
        "Return only JSON with fields: particle_diameter_A, particle_diameter_range_A, "
        "largest_particle_dimension_A, examples (x_display_px, y_display_px, width_display_px, "
        "height_display_px), blob_picker_min_diameter_A, blob_picker_max_diameter_A, "
        "extract_box_size_native_px, extract_box_size_A, confidence, reason, limitations. "
        "Use null for measurements you cannot support. Explain the measurement basis and "
        "margin in reason, without private internal reasoning."
    )
    messages = [
        {"role": "system", "content": "You are a cautious cryo-EM visual measurement assistant. Return auditable estimates, not dataset-specific recalled defaults."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(payload).decode("ascii"), "detail": "high"}},
        ]},
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt.txt").write_text(prompt)
    started = time.monotonic()
    result = run_openai_compatible_model(
        messages, os.getenv("OPENAI_BASE_URL", "https://api.ofox.ai/v1"), resolve_api_key(),
        args.model, max_new_tokens=4096, temperature=0, timeout_seconds=300, max_retries=1,
    )
    summary = {"model": args.model, "image_sha256": hashlib.sha256(payload).hexdigest(),
               "timestamp": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.monotonic() - started, 2),
               "attempts": result["attempts"], "usage": result["raw_response"].get("usage"),
               "finish_reason": result["raw_response"]["choices"][0].get("finish_reason"),
               "raw_text": result["raw_text"], "cryosparc_job_submission": False}
    (args.output_dir / "raw_response.json").write_text(json.dumps(result["raw_response"], indent=2, ensure_ascii=False))
    (args.output_dir / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.output_dir / "model_output.txt").write_text(result["raw_text"])
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
