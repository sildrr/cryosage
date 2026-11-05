# Runs local or OpenAI-compatible models and converts text output into V2 decisions.
import copy
import fcntl
import json
import os
import re
import ssl
import time
import uuid
import warnings
from http.client import RemoteDisconnected
from pathlib import Path
from urllib import error, request
from urllib.parse import urlsplit
from typing import Any, Dict, List, Optional

try:
    from .transport_errors import certificate_failure, transient_tls_failure
except ImportError:
    from transport_errors import certificate_failure, transient_tls_failure

try:
    from .api_payload_budget import prepare_api_payload
    from .api_responses_adapter import responses_payload, normalize_responses
except ImportError:
    from api_payload_budget import prepare_api_payload
    from api_responses_adapter import responses_payload, normalize_responses


DEFAULT_SYSTEM_PROMPT = (
    "You are a CryoSPARC workflow decision model. Return exactly one valid JSON "
    "object. Do not include markdown, comments, explanations, or thinking text."
)


def _record_api_attempt(body: str, status: Optional[int], failed: bool) -> None:
    """Append metadata only; the parent sets a workflow-specific ledger path."""
    path = os.environ.get("CRYOAGENT_USAGE_LEDGER")
    if not path:
        return
    try:
        try:
            response = json.loads(body)
        except (ValueError, TypeError):
            response = {}
        if not isinstance(response, dict):
            response = {}
        usage = response.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}

        def tokens(value: Any) -> int:
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

        inputs = tokens(usage.get("input_tokens", usage.get("prompt_tokens")))
        outputs = tokens(usage.get("output_tokens", usage.get("completion_tokens")))
        details = usage.get("input_tokens_details", usage.get("prompt_tokens_details")) or {}
        entry = {
            "version": 1, "request_id": uuid.uuid4().hex,
            "response_id": response.get("id") if isinstance(response.get("id"), str) else None,
            "timestamp": time.time(), "http_status": status,
            "failed": failed or bool(response.get("error")) or response.get("status") in {"failed", "cancelled"},
            "input_tokens": inputs, "output_tokens": outputs,
            "total_tokens": tokens(usage.get("total_tokens")) or inputs + outputs,
            "cached_tokens": tokens(details.get("cached_tokens")) if isinstance(details, dict) else 0,
            "usage_recorded": bool(usage),
        }
        # No prompts, URLs, headers, keys, or error bodies enter the ledger.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as ledger:
            fcntl.flock(ledger, fcntl.LOCK_EX)
            os.fchmod(ledger.fileno(), 0o600)
            ledger.write(json.dumps(entry, separators=(",", ":")) + "\n")
            ledger.flush()
    except (OSError, ValueError, TypeError) as exc:
        # Accounting must not cause a paid request to be retried.
        warnings.warn(f"API usage ledger write failed ({type(exc).__name__})", RuntimeWarning)


def build_workflow_decision_prompt(
    model_input: Dict[str, Any],
    visual_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build chat messages for the direct model closed-loop test."""
    from selection_safety import SELECTION_SAFETY_GUIDANCE
    try:
        from .cryosift_adapter import CRYOSIFT_SCORE_GUIDANCE
    except ImportError:
        from cryosift_adapter import CRYOSIFT_SCORE_GUIDANCE
    output_contract = {
        "schema_version": "2.0",
        "decision_type": "forward | branch | rollback | stop | request_input",
        "action": "CryoSPARC job type for forward/branch decisions, or omit for stop.",
        "parameters": "Only parameters that should override defaults.",
        "reason": "Short reason for the decision.",
        "confidence": "Number from 0.0 to 1.0.",
        "risk_flags": [],
        "evidence": [],
        "requested_inputs": "Required only for request_input, e.g. [\"particle_diameter_A\"].",
    }
    user_content = {
        "instruction": (
        "Choose the next workflow action from the current CryoSPARC state. "
        "Only select a job type present in model_input.candidate_actions; "
        "do not invent connections or job types. If no safe candidate is "
            "available, return request_input or stop. When particle diameter is "
            "missing, prefer a forward Blob Picker with a deliberately broad "
            "exploratory diameter range; use request_input only when no "
            "scientifically reasonable range can be proposed. " + CRYOSIFT_SCORE_GUIDANCE + SELECTION_SAFETY_GUIDANCE
        ),
        "model_input": model_input,
        "output_contract": output_contract,
        "valid_examples": [
            {
                "schema_version": "2.0",
                "decision_type": "forward",
                "action": "class_2D_new",
                "parameters": {"compute_num_gpus": 4, "class2D_K": 50},
                "reason": "Particles are extracted and ready for 2D classification.",
                "confidence": 0.85,
                "risk_flags": [],
                "evidence": ["The last job completed with a particles output."],
            },
            {
                "schema_version": "2.0",
                "decision_type": "stop",
                "reason": "No safe next action is clear from the current state.",
                "confidence": 0.5,
                "risk_flags": ["needs_human_review"],
                "evidence": [],
            },
            {
                "schema_version": "2.0",
                "decision_type": "request_input",
                "requested_inputs": ["particle_diameter_A"],
                "reason": "Blob picking requires a particle diameter that is not available.",
                "confidence": 0.95,
                "risk_flags": ["needs_human_input"],
                "evidence": ["No reliable particle diameter is present in the dataset context."],
            },
            {
                "schema_version": "2.0",
                "decision_type": "forward",
                "action": "blob_picker_gpu",
                "parameters": {"diameter": 150, "diameter_max": 250},
                "reason": "Use a deliberately broad exploratory diameter range.",
                "confidence": 0.65,
                "risk_flags": ["exploratory_parameter_range"],
                "evidence": ["Particle size is uncertain; the range is explicitly exploratory."],
            },
        ],
    }
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]
    if visual_context:
        sheet = visual_context.get("contact_sheet") or {}
        image_url = sheet.get("data_url") if isinstance(sheet, dict) else None
        image_urls = [image_url] if isinstance(image_url, str) and image_url else []
        full_field = visual_context.get("visual_evidence_version") == "full_field_v4"
        if full_field:
            scatter = visual_context.get("score_scatter") or {}
            if scatter.get("data_url"):
                image_urls.append(scatter["data_url"])
            image_urls.extend(item["data_url"] for item in visual_context.get("evidence_images", []) if item.get("data_url"))
        image_available = bool(image_urls)
        visual_text = {
            "visual_instruction": (
                "Review the uncropped full micrographs, calibrated scale bars, labelled "
                "pick examples, and any NCC/Power plot or candidate extraction boxes. "
                "Use per-image display/native pixel conversions. Candidate boxes use "
                "native pixels; high pick scores do not prove genuine particle quality."
            ) if full_field else (
                "The attached image is a class-average contact sheet. Each tile is labelled "
                "class_id=<integer>. Inspect particle quality, structural consistency, "
                "noise, and view diversity. Use the labels exactly in selected_templates."
            ),
            "visual_context": {
                key: value for key, value in visual_context.items()
                if key not in {"contact_sheet", "score_scatter", "evidence_images"}
            },
            "visual_attachment_status": (
                "available" if image_available else
                "unavailable; make the decision from the structured state without requesting human input solely for the missing image"
            ),
        }
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": json.dumps(visual_text, ensure_ascii=False)},
        ]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url, **({"detail": "high"} if full_field else {})}})
        messages.append({"role": "user", "content": content})
    return messages


def render_chat_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Render Qwen chat messages while disabling thinking when supported."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def run_qwen_lora_model(
    messages: List[Dict[str, str]],
    base_model_path: str,
    adapter_path: Optional[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
) -> Dict[str, Any]:
    """Load the base model plus optional LoRA adapter and generate raw text."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = normalize_optional_path(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    prompt = render_chat_prompt(tokenizer, messages)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "raw_text": raw_text,
    }


def run_openai_compatible_model(
    messages, api_base, api_key, model_name, max_new_tokens=512, temperature=0.0,
    timeout_seconds=300, max_retries=3, retry_backoff_seconds=2.0,
    prompt_cache_key=None, prompt_cache_options=None, tools=None, tool_choice=None,
):
    """Try the primary provider, then an explicitly configured same-model backup."""
    try:
        from .api_failover import route_for, recoverable, audit
    except ImportError:
        from api_failover import route_for, recoverable, audit
    route = route_for(api_base, model_name)
    kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature,
        timeout_seconds=timeout_seconds, max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds, prompt_cache_key=prompt_cache_key,
        prompt_cache_options=prompt_cache_options, tools=tools, tool_choice=tool_choice)
    try:
        return _run_openai_compatible_model_once(messages,api_base,api_key,model_name,**kwargs)
    except RuntimeError as exc:
        if not route or not recoverable(exc):
            raise
        event = dict(primary_base=api_base, fallback_base=route['fallback_base'],
                     requested_model=model_name, fallback_model=route['fallback_model'])
        audit(route, {**event, 'status':'attempting_fallback'})
        fallback_key = read_api_key_file(Path(route['fallback_env_file']).expanduser(), 'OPENAI_API_KEY')
        if not fallback_key:
            raise ValueError('Configured fallback API credential file contains no key') from exc
        try:
            result = _run_openai_compatible_model_once(messages, route['fallback_base'],
                fallback_key, route['fallback_model'], direct=route.get('direct') is True, **kwargs)
        except Exception:
            audit(route, {**event, 'status':'fallback_failed'})
            raise
        result['provider_failover'] = event
        audit(route, {**event, 'status':'fallback_succeeded'})
        return result


def _run_openai_compatible_model_once(
    messages: List[Dict[str, str]],
    api_base: str,
    api_key: str,
    model_name: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    timeout_seconds: int = 300,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    direct: bool = False,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible endpoint with bounded transient-error retries."""
    if not api_base:
        raise ValueError("api_base is required for the OpenAI-compatible backend.")
    if not api_key:
        raise ValueError("api_key is required for the OpenAI-compatible backend.")
    if not model_name:
        raise ValueError("model_name is required for the OpenAI-compatible backend.")

    # API4 empirically ignores images in earlier consecutive user messages.
    # Preserve every block in order, without crossing assistant/tool boundaries.
    if urlsplit(api_base).hostname == '15.204.107.114' and urlsplit(api_base).port == 3009:
        from prompt_context import coalesce_adjacent_user_messages
        messages = coalesce_adjacent_user_messages(messages)

    base_payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    # Packy's Chat Completions route has incompatible reasoning/tool restrictions.
    # Use Responses there, without changing other providers' wire protocol.
    request_profile = "openai_compatible"
    if urlsplit(api_base).hostname in {"packyapi.ai", "www.packyapi.ai"}:
        request_profile = "packy_responses"
    if tools:
        base_payload["tools"] = tools
    if tool_choice:
        base_payload["tool_choice"] = tool_choice

    cache_enabled = bool(prompt_cache_key or prompt_cache_options)

    def build_payload(include_cache: bool) -> Dict[str, Any]:
        payload = dict(base_payload)
        if include_cache:
            if prompt_cache_key:
                payload["prompt_cache_key"] = prompt_cache_key
            if prompt_cache_options:
                payload["prompt_cache_options"] = prompt_cache_options
        return responses_payload(payload) if request_profile == "packy_responses" else payload

    def remove_explicit_cache_breakpoints(value: Any) -> Any:
        if isinstance(value, list):
            return [remove_explicit_cache_breakpoints(item) for item in value]
        if isinstance(value, dict):
            return {
                key: remove_explicit_cache_breakpoints(item)
                for key, item in value.items()
                if key != "prompt_cache_breakpoint"
            }
        return value

    payload_sizing = None
    endpoint = api_base.rstrip("/") + (
        "/responses" if request_profile == "packy_responses" else "/chat/completions"
    )
    retryable_http_codes = {408, 429, 500, 502, 503, 504}
    # Some compatible gateways occasionally return a body-less 403 while the
    # upstream route is being refreshed. Retry that transient form once, but
    # never retry explicit authentication or quota failures.
    bodyless_forbidden_retries = 0
    cache_fallback_used = False
    attempts = 0
    provider_type = None
    if urlsplit(api_base).hostname == "api.ofox.ai":
        configured = os.getenv("CRYOAGENT_OFOX_PROVIDER_TYPE")
        # The default Azure route currently returns a traffic-group 404.
        # Pin the documented healthy route without changing the requested model.
        if configured is None and model_name in {
            "openai/gpt-6.1-sol", "gpt-6.1-sol", "openai/gpt-6-luna", "gpt-6-luna",
        }:
            configured = "openai"
        if configured and configured != "auto":
            if configured not in {"openai", "azure_foundry", "foundry"}:
                raise ValueError("Unsupported CRYOAGENT_OFOX_PROVIDER_TYPE")
            provider_type = configured
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if provider_type:
        request_headers["X-OfoxAI-Provider-Type"] = provider_type
    while True:
        attempts += 1
        payload = build_payload(cache_enabled)
        payload, request_body, attempt_sizing = prepare_api_payload(payload)
        if payload_sizing is None:
            payload_sizing = attempt_sizing
        payload_sizing["final_bytes"] = attempt_sizing["final_bytes"]
        req = request.Request(
            endpoint,
            data=request_body,
            headers=request_headers,
            method="POST",
        )
        attempt_body = ""
        attempt_status = None
        attempt_failed = True
        try:
            open_url = request.build_opener(request.ProxyHandler({})).open if direct else request.urlopen
            with open_url(req, timeout=timeout_seconds) as response:
                attempt_status = getattr(response, "status", 200)
                raw_response = response.read().decode("utf-8")
                attempt_body = raw_response
            attempt_failed = False
            break
        except error.HTTPError as exc:
            attempt_status = exc.code
            error_body = exc.read().decode("utf-8", errors="replace")
            attempt_body = error_body
            cache_error = (
                cache_enabled
                and not cache_fallback_used
                and (
                    "prompt_cache_breakpoint" in error_body
                    or "prompt_cache_options" in error_body
                    or "prompt cache" in error_body.lower()
                )
            )
            if cache_error or (
                cache_enabled
                and not cache_fallback_used
                and exc.code == 403
                and not error_body.strip()
            ):
                # Providers are not consistent about exposing cache support in
                # their OpenAI-compatible surface. Retry once without optional
                # cache extensions; implicit prefix caching can still apply.
                cache_enabled = False
                cache_fallback_used = True
                base_payload["messages"] = remove_explicit_cache_breakpoints(
                    copy.deepcopy(base_payload["messages"])
                )
                continue
            if exc.code == 403 and not error_body.strip():
                if bodyless_forbidden_retries < 1:
                    bodyless_forbidden_retries += 1
                    time.sleep(retry_backoff_seconds * 2)
                    continue
            if exc.code not in retryable_http_codes or attempts > max_retries:
                raise RuntimeError(
                    f"OpenAI-compatible request failed with HTTP {exc.code}: {error_body}"
                ) from exc
        except (error.URLError, RemoteDisconnected, ConnectionResetError, TimeoutError, ssl.SSLError) as exc:
            error_body = str(exc)
            if (certificate_failure(exc) or
                    isinstance(exc, ssl.SSLError) and not transient_tls_failure(exc) or
                    attempts > max_retries):
                raise RuntimeError(
                    f"OpenAI-compatible request failed to reach {endpoint}: {error_body}"
                ) from exc
        finally:
            _record_api_attempt(attempt_body, attempt_status, attempt_failed)
        time.sleep(retry_backoff_seconds * (2 ** (attempts - 1)))

    parsed = json.loads(raw_response)
    provider_raw_response = parsed
    if request_profile == "packy_responses":
        parsed = normalize_responses(parsed)
    try:
        message = parsed["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "OpenAI-compatible response did not contain choices[0].message."
        ) from exc

    content = message.get("content")
    raw_text = normalize_message_content(content) if content is not None else ""
    return {
        "endpoint": endpoint,
        "request_profile": request_profile,
        "provider_type": provider_type,
        "request_payload": payload,
        "request_payload_sizing": payload_sizing,
        "raw_response": parsed,
        "provider_raw_response": provider_raw_response,
        "raw_text": raw_text,
        "assistant_message": message,
        "tool_calls": message.get("tool_calls") or [],
        "attempts": attempts,
        "cache_fallback_used": cache_fallback_used,
    }


def resolve_api_key(
    explicit_api_key: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> str:
    """Resolve an API key without requiring secrets in project files or argv."""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()
    env_value = os.environ.get(api_key_env, "").strip()
    if env_value:
        return env_value
    env_file = Path(
        os.getenv("CRYOAGENT_API_ENV_FILE", "~/.config/cryoagent/api.env")
    ).expanduser()
    value = read_api_key_file(env_file, api_key_env)
    if value:
        return value
    raise ValueError(
        f"API key not provided. Set --api-key, export {api_key_env}, "
        f"or configure {env_file}."
    )


def read_api_key_file(env_file: Path, api_key_env: str = 'OPENAI_API_KEY') -> Optional[str]:
    """Read an assignment-style env file or a single bare sk- key without logging it."""
    if env_file.is_file():
        lines = env_file.read_text(encoding='utf-8').splitlines()
        meaningful = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith('#')]
        if len(meaningful) == 1 and meaningful[0].startswith('sk-') and '=' not in meaningful[0]:
            return meaningful[0]
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            name, separator, value = stripped.partition("=")
            if separator and name.strip() == api_key_env:
                value = value.strip().strip("\"'")
                if value:
                    return value
    return None


def normalize_message_content(content: Any) -> str:
    """Normalize OpenAI-compatible message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    raise ValueError(
        f"Unsupported message content type from API response: {type(content).__name__}"
    )


def normalize_optional_path(path: Optional[str]) -> Optional[str]:
    """Treat empty strings and common sentinel values as no adapter."""
    if path is None:
        return None
    stripped = path.strip()
    if stripped.lower() in {"", "none", "null"}:
        return None
    return stripped


def parse_model_decision_text(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object from model text."""
    json_text = extract_first_json_object(clean_model_text(text))
    return json.loads(json_text)


def clean_model_text(text: str) -> str:
    """Remove common non-JSON wrappers from model output."""
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^/think\b.*?(?=\{)", "", cleaned, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    return cleaned


def extract_first_json_object(text: str) -> str:
    """Return the first balanced JSON object from a larger string."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in model output.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("No balanced JSON object found in model output.")
