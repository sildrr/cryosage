#!/usr/bin/env python3
"""Local multimodal Qwen+LoRA worker used by the autonomous MCP runner."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-image-pixels", type=int, default=200704)
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def image_from_url(url: str):
    from PIL import Image

    if url.startswith("data:"):
        header, payload = url.split(",", 1)
        data = unquote_to_bytes(payload) if ";base64" not in header else base64.b64decode(payload)
        return Image.open(BytesIO(data)).convert("RGB")
    raise ValueError("The local visual worker only accepts embedded data URLs.")


def merge_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Qwen's chat template permits one leading system message only."""
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "system":
            converted.append(message)
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", ""))
                for part in content
                if part.get("type") == "text"
            )
        system_parts.append(str(content))
    if system_parts:
        converted.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return converted


def materialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in merge_system_messages(messages):
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue
        parts = []
        for part in content:
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if not url:
                    raise ValueError("Image attachment is missing image_url.url.")
                parts.append({"type": "image", "image": image_from_url(url)})
            elif part.get("type") == "image" and part.get("path"):
                from PIL import Image

                with Image.open(part["path"]) as image:
                    parts.append({"type": "image", "image": image.convert("RGB")})
            else:
                parts.append(part)
        converted.append({**message, "content": parts})
    return converted


def run_worker(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, args.torch_dtype) if args.torch_dtype != "auto" else "auto"
    required_gpus = int(os.getenv("CRYOAGENT_MODEL_REQUIRED_GPUS", "0"))
    if torch.cuda.device_count() < required_gpus:
        raise RuntimeError(f"Expected {required_gpus} visible GPUs, found {torch.cuda.device_count()}")
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    device_map = getattr(model, "hf_device_map", {})
    if required_gpus > 1:
        gpu_devices = {str(value) for value in device_map.values()
                       if isinstance(value, int) or str(value).startswith("cuda")}
        if len(gpu_devices) < required_gpus:
            raise RuntimeError(f"Model was not sharded across {required_gpus} GPUs: {device_map}")
    write_event({
        "event": "model_loaded",
        "base_model": args.base_model,
        "adapter": args.adapter,
        "device": str(next(model.parameters()).device),
        "vision": True,
        "device_map": device_map,
        "visible_gpu_count": torch.cuda.device_count(),
    })

    image_kwargs = {
        "size": {
            "shortest_edge": min(128 * 128, args.max_image_pixels),
            "longest_edge": args.max_image_pixels,
        }
    }
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("command") == "shutdown":
            write_event({"event": "shutdown"})
            return
        messages = materialize_messages(request["messages"])
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"images_kwargs": image_kwargs},
        )
        device = next(model.parameters()).device
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=int(request.get("max_new_tokens", args.max_new_tokens)),
                do_sample=float(request.get("temperature", args.temperature)) > 0,
                temperature=max(float(request.get("temperature", args.temperature)), 1e-5),
                use_cache=True,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        raw_text = processor.tokenizer.decode(
            output[0, prompt_length:], skip_special_tokens=True
        )
        write_event({
            "event": "generation",
            "request_id": request.get("request_id"),
            "raw_text": raw_text,
            "prompt_tokens": int(prompt_length),
        })
        del inputs, output


def write_event(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def main() -> None:
    args = parse_args()
    if not args.worker:
        raise SystemExit("This script currently supports only --worker mode.")
    run_worker(args)


if __name__ == "__main__":
    main()
