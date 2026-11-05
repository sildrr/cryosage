import base64
import copy
import io
import json
import os
import unittest
from unittest import mock

from PIL import Image

from api_payload_budget import encode_payload, prepare_api_payload
from model_direct_runner import run_openai_compatible_model


def image_url(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class ApiPayloadBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.micrograph = image_url(Image.frombytes("RGB", (1024, 1024), os.urandom(1024 * 1024 * 3)))
        cls.scatter = image_url(Image.new("RGB", (720, 560), "white"))

    def payload(self):
        return {
            "model": "test", "prompt_cache_key": "stable",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "contract",
                    "prompt_cache_breakpoint": {"mode": "explicit"}}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url, "detail": "high"}}
                    for url in [self.micrograph] * 12 + [self.scatter]
                ]},
            ],
        }

    def test_small_request_is_byte_identical(self):
        payload = {"messages": [{"role": "user", "content": "test"}]}
        prepared, body, report = prepare_api_payload(payload)
        self.assertEqual(body, encode_payload(payload))
        self.assertEqual(prepared, payload)
        self.assertEqual(report["compressed_images"], [])

    def test_thirteen_images_fit_without_changing_dimensions_or_count(self):
        payload = self.payload()
        original = copy.deepcopy(payload)
        prepared, body, report = prepare_api_payload(payload)
        self.assertGreater(report["original_bytes"], 32_000_000)
        self.assertLessEqual(len(body), 28 * 1024 * 1024)
        self.assertEqual(len(prepared["messages"][1]["content"]), 13)
        self.assertEqual(payload, original)
        self.assertEqual(prepared["messages"][0], original["messages"][0])
        self.assertEqual(prepared["messages"][1]["content"][-1], original["messages"][1]["content"][-1])
        for part in prepared["messages"][1]["content"][:-1]:
            self.assertEqual(part["image_url"]["detail"], "high")
            url = part["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as img:
                self.assertEqual(img.size, (1024, 1024))
        self.assertEqual(len(report["compressed_images"]), 12)

    def test_text_only_overflow_fails_without_truncating(self):
        with self.assertRaisesRegex(ValueError, "No images or context were removed"):
            prepare_api_payload({"messages": [{"content": "x" * 1000}]}, max_bytes=500)

    def test_tight_budget_uses_chroma_fallback_without_resizing_or_dropping(self):
        payload = self.payload()
        payload['messages'][1]['content'] = payload['messages'][1]['content'][:2] + [
            payload['messages'][1]['content'][-1]]
        original = copy.deepcopy(payload)
        prepared, body, report = prepare_api_payload(payload, max_bytes=3 * 1024**2)
        self.assertLessEqual(len(body), 3 * 1024**2)
        self.assertEqual(len(prepared['messages'][1]['content']), 3)
        self.assertEqual(prepared['messages'][1]['content'][-1], original['messages'][1]['content'][-1])
        self.assertEqual(payload, original)
        self.assertTrue(any(r['subsampling'] == 2 for r in report['compressed_images']))
        self.assertTrue(all(r['dimensions'] == [1024, 1024] for r in report['compressed_images']))

    def test_responses_inline_images_obey_the_same_byte_budget(self):
        payload = {"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": self.micrograph, "detail": "high"}
            for _ in range(12)
        ]}], "model": "test", "max_output_tokens": 256}
        prepared, body, report = prepare_api_payload(payload)
        self.assertLessEqual(len(body), 28 * 1024 * 1024)
        self.assertEqual(len(report["compressed_images"]), 12)
        self.assertTrue(prepared["input"][0]["content"][0]["image_url"].startswith("data:image/jpeg;"))
        self.assertTrue(payload["input"][0]["content"][0]["image_url"].startswith("data:image/png;"))

    def test_nonpositive_budget_rejected(self):
        with self.assertRaises(ValueError):
            prepare_api_payload({}, max_bytes=0)

    def test_configured_budget(self):
        with mock.patch.dict(os.environ, {"CRYOAGENT_API_MAX_REQUEST_BYTES": "500"}):
            with self.assertRaises(ValueError):
                prepare_api_payload({"messages": [{"content": "x" * 1000}]})

    def test_actual_transport_and_audit_match(self):
        payload = self.payload()
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        with mock.patch("model_direct_runner.request.urlopen", return_value=response) as urlopen:
            result = run_openai_compatible_model(
                payload["messages"], "https://example.test/v1", "secret", "test",
            )
        wire = urlopen.call_args.args[0].data
        self.assertLessEqual(len(wire), 28 * 1024 * 1024)
        self.assertEqual(json.loads(wire), result["request_payload"])
        self.assertEqual(len(wire), result["request_payload_sizing"]["final_bytes"])
        self.assertEqual(len(result["request_payload_sizing"]["compressed_images"]), 12)
        self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
