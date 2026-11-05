"""Small, observed movie facts retained in model-facing result summaries."""
from numbers import Integral
from typing import Any


MOVIE_SUMMARY_FIELDS = (
    "movie_blob/shape", "movie_blob/format", "movie_blob/psize_A",
    "movie_blob/is_gain_corrected", "movie_blob/has_defect_file",
)

MOTION_CORRECTION_GUIDANCE = (
    "For native Patch/Full-frame Motion Correction, frame_end is optional; "
    "omit it to retain CryoSPARC's default end-of-movie behavior. Do not require "
    "an explicit frame_end or guess a frame index merely to launch the job. "
    "movie_blob/shape is [frames, height, width]; use the observed shape when available. "
    "The external exec_path requirement belongs to MotionCor2, not native Patch Motion Correction. "
    "Use the supplied pixel size; do not infer super-resolution sampling from TIFF format "
    "or image dimensions alone. Request missing metadata only when required for the chosen action."
)


def movie_summary_values(output: dict[str, Any]) -> dict[str, Any]:
    """Whitelist small movie fields, not arbitrary potentially large summaries."""
    result = {}
    for key in MOVIE_SUMMARY_FIELDS:
        value = (output.get("summary_values") or {}).get(key)
        if key.endswith("/shape"):
            if (isinstance(value, (list, tuple)) and len(value) == 3
                    and all(isinstance(x, Integral) and not isinstance(x, bool) and x > 0
                            for x in value)):
                result[key] = [int(x) for x in value]
        elif isinstance(value, (int, float, bool)) or (isinstance(value, str) and len(value) <= 64):
            result[key] = value
    return result


def movie_metadata(output: dict[str, Any]) -> dict[str, Any]:
    values = movie_summary_values(output)
    metadata = {key.split("/", 1)[1]: value for key, value in values.items()}
    if "shape" in metadata:
        metadata["frame_count"] = metadata["shape"][0]
        metadata["shape_axes"] = ["frames", "height", "width"]
    return metadata
