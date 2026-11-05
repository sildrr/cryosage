"""One authorized, read-only visual API probe; no CryoSPARC execution tools."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

from model_direct_runner import resolve_api_key, run_openai_compatible_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-context", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    visual = json.loads(args.visual_context.read_text())
    url = visual["contact_sheet"]["data_url"]
    data = base64.b64decode(url.split(",", 1)[1])
    selected = [0, 6, 7, 10, 11, 13, 18, 25, 27, 28, 30, 31, 35, 37, 41, 43, 46, 47]
    prompt = (
        "Independently assess whether the current extraction field of view is adequate using this "
        "actual CryoSPARC Class 2D contact sheet. The sheet is the original visual evidence used "
        "for Select 2D job J524, from source Class 2D job J523. Focus on the retained class IDs "
        f"{selected}; other shown classes are contextual, not retained. "
        "Verified acquisition/extraction facts: native micrograph pixel size 0.6575 A/pixel; "
        "extraction box side 320 native pixels = 210.4 A. Each Class 2D average is stored at "
        "128x128 pixels with 1.6437501 A/pixel, covering the same approximately 210.4 A field. "
        "Each average was resized to a 192x192 display tile and labeled class_id=<ID>; label bars "
        "are outside the particle image. Display pixels, class-average pixels and native "
        "extraction pixels are different units. Do not infer real-space cropping merely from "
        "the Class 2D sampling size. No 3D map, expected molecule, target diameter, reference "
        "structure, or later reconstruction outcome is supplied. "
        "Determine whether there is supported evidence of particle truncation, inadequate "
        "margin, off-centering, or instead intact particles within adequate margins. Keep "
        "those possibilities separate. A low-signal border does not prove completeness: "
        "particle images may be circularly masked/windowed. Conversely, a small or noisy "
        "class alone does not prove truncation. Report concrete class IDs and visible "
        "observations, approximate particle extent if measurable, and uncertainties. "
        "Choose maintain, increase, decrease, or insufficient_evidence for box size. If a "
        "change is supported, suggest a native-pixel box multiple of 8 and justify its "
        "physical margin; otherwise do not force a change. This is an assessment only, "
        "not permission to execute jobs. Return JSON only with: assessment "
        "(truncation_likely|truncation_possible|no_clear_truncation|insufficient_evidence), "
        "confidence (0 to 1), per_class_evidence (class_id, observation, suspected_issue), "
        "estimated_particle_extent_A (or null), box_recommendation "
        "(decision, current_native_px, recommended_native_px or null, recommended_side_A or null, reason), "
        "alternative_explanations, and additional_evidence_needed. Provide auditable visual "
        "reasons, not private internal reasoning. Answer descriptive strings in Chinese."
    )
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
    ]}]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt.txt").write_text(prompt)
    (args.output_dir / "class_evidence.png").write_bytes(data)
    summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "model": "openai/gpt-6.1-sol", "source_visual_context": str(args.visual_context),
               "image_sha256": hashlib.sha256(data).hexdigest(), "selected_class_ids": selected,
               "cryosparc_jobs_submitted": False, "max_retries": 0}
    started = time.monotonic()
    try:
        response = run_openai_compatible_model(
            messages, os.getenv("OPENAI_BASE_URL", "https://api.ofox.ai/v1"), resolve_api_key(),
            summary["model"], max_new_tokens=4096, temperature=0, timeout_seconds=300, max_retries=0)
        raw = response["raw_response"]
        summary.update(success=True, elapsed_seconds=round(time.monotonic()-started, 2),
                       attempts=response.get("attempts"), usage=raw.get("usage"),
                       finish_reason=raw["choices"][0].get("finish_reason"), raw_text=response["raw_text"])
        (args.output_dir / "raw_response.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2))
        (args.output_dir / "model_output.txt").write_text(response["raw_text"])
    except Exception as exc:
        summary.update(success=False, elapsed_seconds=round(time.monotonic()-started, 2),
                       error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (args.output_dir / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
