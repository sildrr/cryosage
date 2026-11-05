"""Bound inline-image request bytes without cropping or dropping evidence."""

import base64
import copy
import hashlib
import io
import json
import os


DEFAULT_MAX_REQUEST_BYTES = 28 * 1024 * 1024


def encode_payload(payload):
    return json.dumps(payload).encode("utf-8")


def prepare_api_payload(payload, max_bytes=None):
    if max_bytes is None:
        max_bytes = int(os.environ.get(
            "CRYOAGENT_API_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES
        ))
    if max_bytes <= 0:
        raise ValueError("CRYOAGENT_API_MAX_REQUEST_BYTES must be positive.")
    body = encode_payload(payload)
    report = {
        "original_bytes": len(body), "final_bytes": len(body),
        "max_bytes": max_bytes, "compressed_images": [],
    }
    if len(body) <= max_bytes:
        return payload, body, report

    from PIL import Image

    prepared = copy.deepcopy(payload)
    images = []
    for message_index, message in enumerate(prepared.get("messages", prepared.get("input", []))):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") not in {"image_url", "input_image"}:
                continue
            if part["type"] == "input_image":
                image_url, url_key = part, "image_url"
            else:
                image_url, url_key = part.get("image_url"), "url"
            if not isinstance(image_url, dict):
                continue
            url = image_url.get(url_key, "")
            if not isinstance(url, str) or not url.startswith("data:image/"):
                continue
            header, separator, encoded = url.partition(",")
            if not separator or not header.endswith(";base64"):
                continue
            # Keep small plots and labels lossless; large micrographs dominate bytes.
            if len(encoded) < 256 * 1024:
                continue
            original = base64.b64decode(encoded, validate=True)
            images.append((image_url, url_key, original, {
                "message_index": message_index, "part_index": part_index,
                "original_image_bytes": len(original),
                "original_sha256": hashlib.sha256(original).hexdigest(),
            }))

    changes = {}
    # Only fall back to chroma subsampling after the original high-fidelity
    # passes fail. Keep all image pixels/evidence; small plots remain lossless.
    for quality, subsampling in ((92, 0), (85, 0), (75, 0), (75, 2), (65, 2), (60, 2)):
        for index, (image_url, url_key, original, metadata) in enumerate(images):
            # Each pass starts from original pixels, avoiding cumulative JPEG loss.
            with Image.open(io.BytesIO(original)) as source:
                if getattr(source, "is_animated", False):
                    continue
                rgba = source.convert("RGBA")
                rgb = Image.new("RGB", source.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                output = io.BytesIO()
                rgb.save(output, format="JPEG", quality=quality, subsampling=subsampling)
                compressed = output.getvalue()
                dimensions = list(source.size)
            replacement = "data:image/jpeg;base64," + base64.b64encode(compressed).decode("ascii")
            if len(replacement) >= len(image_url[url_key]):
                continue
            image_url[url_key] = replacement
            changes[index] = dict(
                metadata, wire_image_bytes=len(compressed), dimensions=dimensions,
                wire_sha256=hashlib.sha256(compressed).hexdigest(),
                codec="jpeg", quality=quality, subsampling=subsampling,
            )
        body = encode_payload(prepared)
        if len(body) <= max_bytes:
            report.update(final_bytes=len(body), compressed_images=list(changes.values()))
            return prepared, body, report

    raise ValueError(
        f"API request exceeds byte budget after image encoding: {len(body)} > {max_bytes} "
        f"bytes (original {report['original_bytes']}). No images or context were removed. "
        "Reduce evidence count/text upstream or configure CRYOAGENT_API_MAX_REQUEST_BYTES "
        "to match the provider's documented limit."
    )
