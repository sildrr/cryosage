"""Paid, opt-in synthetic vision/tool probe; never contacts CryoSPARC."""

import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
import secrets
import sys
import time

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_direct_runner import run_openai_compatible_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-mode", choices=("explicit", "disabled"), default="disabled")
    parser.add_argument("--agent-kb-loop", action="store_true",
                        help="Also exercise the real KB loop with a synthetic read-only executor")
    args = parser.parse_args()
    key = None
    for line in Path(args.env_file).expanduser().read_text().splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name.strip() == "OPENAI_API_KEY":
            key = value.strip().strip("\"'")
    if not key:
        raise ValueError("OPENAI_API_KEY is missing in the selected env file")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    results = []

    def call(label, messages, **kwargs):
        start = time.monotonic()
        if args.cache_mode == "explicit":
            messages = [{"role": "system", "content": [{
                "type": "text", "text": "This is a synthetic, read-only integration test.",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }]}] + messages
            kwargs.update(prompt_cache_key="cryoagent:packy-integration-probe",
                          prompt_cache_options={"mode": "explicit", "ttl": "30m"})
        try:
            result = run_openai_compatible_model(
                messages, args.api_base, key, args.model, max_new_tokens=2048,
                timeout_seconds=120, max_retries=0, **kwargs,
            )
        except Exception as exc:
            (output / (label + "_error.txt")).write_text(str(exc).replace(key, "[REDACTED]"))
            raise RuntimeError(f"{label} failed: {str(exc).replace(key, '[REDACTED]')}") from None
        elapsed = round(time.monotonic() - start, 2)
        # The saved request contains only synthetic images/text, never headers/keys.
        (output / (label + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        results.append({
            "test": label, "elapsed_seconds": elapsed,
            "finish_reason": result["raw_response"]["choices"][0].get("finish_reason"),
            "usage": result["raw_response"].get("usage"),
            "raw_text": result["raw_text"], "tool_call_count": len(result["tool_calls"]),
            "cache_fallback_used": result["cache_fallback_used"],
            "request_profile": result["request_profile"],
        })
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        return result

    image = Image.new("RGB", (768, 768), "white")
    draw = ImageDraw.Draw(image)
    for bounds in ((80, 100, 200, 220), (380, 85, 500, 205), (270, 440, 390, 560)):
        draw.ellipse(bounds, fill="blue")
    draw.rectangle((530, 360, 680, 490), fill="red")
    image.save(output / "synthetic_vision.png")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    vision = call("vision", [{"role": "user", "content": [
        {"type": "text", "text": "Inspect the attached image. Return only JSON with integer keys blue_circle_count and red_rectangle_count. Count the shapes you can see; do not describe them."},
        {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
    ]}])
    vision_ok = json.loads(vision["raw_text"]) == {"blue_circle_count": 3, "red_rectangle_count": 1}

    tools = [{"type": "function", "function": {
        "name": "lookup_probe_calibration",
        "description": "Read a synthetic calibration record. Read-only; no external side effects.",
        "parameters": {"type": "object", "properties": {
            "record_id": {"type": "string"},
        }, "required": ["record_id"], "additionalProperties": False},
    }}]
    messages = [{"role": "user", "content": (
        "Call lookup_probe_calibration with record_id SYNTHETIC-001. You cannot know its "
        "values before calling the tool. Then return only JSON with record_id, marker, "
        "and pixel_size_A copied exactly from the tool result."
    )}]
    first = call("tool_request", messages, tools=tools, tool_choice="auto")
    calls = first["tool_calls"]
    if len(calls) != 1:
        raise ValueError("Expected exactly one tool call")
    tool_call = calls[0]
    if tool_call["function"]["name"] != "lookup_probe_calibration":
        raise ValueError("Unexpected tool name")
    if json.loads(tool_call["function"]["arguments"]) != {"record_id": "SYNTHETIC-001"}:
        raise ValueError("Unexpected tool arguments")
    record = {"record_id": "SYNTHETIC-001", "marker": secrets.token_hex(8), "pixel_size_A": 0.6575}
    messages.extend([first["assistant_message"], {
        "role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(record),
    }])
    final = call("tool_final", messages, tools=tools, tool_choice="auto")
    tools_ok = not final["tool_calls"] and json.loads(final["raw_text"]) == record
    agent_ok = None
    if args.agent_kb_loop:
        from kb_tool_loop import run_kb_tool_call_loop

        agent_record = {"dataset_id": "SYNTHETIC-001", "marker": secrets.token_hex(8)}
        agent_calls = []

        async def model_call(working, definitions):
            return call(f"agent_kb_{len(agent_calls)}_{len(working)}", working,
                        tools=definitions, tool_choice="auto" if definitions else None)

        async def executor(name, arguments):
            if name != "kb_get_dataset_summary" or arguments != {"dataset_id": "SYNTHETIC-001"}:
                raise ValueError("Only the synthetic summary lookup is allowed")
            agent_calls.append(name)
            return agent_record

        agent_result = asyncio.run(run_kb_tool_call_loop([{
            "role": "user", "content": (
                "Read-only integration test. Call kb_get_dataset_summary for dataset_id "
                "SYNTHETIC-001 exactly once, then return only JSON containing the dataset_id "
                "and marker from its result. Do not guess the marker or call other tools."
            ),
        }], model_call, executor, max_tool_calls=2))
        agent_ok = len(agent_calls) == 1 and json.loads(agent_result["raw_text"]) == agent_record
    summary = {"api_base": args.api_base, "model": args.model,
               "vision_passed": vision_ok, "tool_roundtrip_passed": tools_ok,
               "agent_kb_loop_passed": agent_ok, "calls": results}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "calls"}), flush=True)
    if not (vision_ok and tools_ok) or agent_ok is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
