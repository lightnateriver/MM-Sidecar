from __future__ import annotations

import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from mm_sidecar.contracts import MediaTransport
from mm_sidecar.integrations.vllm_patch.context import RequestCapture
from mm_sidecar.integrations.vllm_patch.normalization import (
    build_normalized_image_from_url,
)


def _make_image() -> Image.Image:
    return Image.new("RGB", (288, 512), color=(12, 34, 56))


def _make_base64_data_url() -> str:
    image = _make_image()
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


class VllmPatchNormalizationTests(unittest.TestCase):
    def test_normalize_http_image(self) -> None:
        image = _make_image()
        image.format = "JPEG"

        normalized = build_normalized_image_from_url(
            image_url="https://example.com/demo.jpg?x=1",
            image=image,
            media_uuid="uuid-http",
            request_scope_key="req-http",
            item_index=0,
        )

        self.assertEqual(normalized.source_ref.transport, MediaTransport.HTTP)
        self.assertEqual(normalized.orig_size_hw, (512, 288))
        self.assertEqual(normalized.mime_type, "image/jpeg")
        self.assertIsNone(normalized.byte_size)

    def test_normalize_base64_image(self) -> None:
        image = _make_image()
        image.format = "JPEG"
        data_url = _make_base64_data_url()

        normalized = build_normalized_image_from_url(
            image_url=data_url,
            image=image,
            media_uuid="uuid-b64",
            request_scope_key="req-b64",
            item_index=3,
        )

        self.assertEqual(normalized.source_ref.transport, MediaTransport.BASE64)
        self.assertEqual(normalized.source_ref.request_scope_key, "req-b64")
        self.assertTrue(normalized.source_ref.source_key.endswith("image:3"))
        self.assertEqual(normalized.mime_type, "image/jpeg")
        self.assertIsNotNone(normalized.byte_size)
        self.assertGreater(normalized.byte_size or 0, 0)

    def test_normalize_local_file_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "demo.jpg"
            _make_image().save(image_path, format="JPEG")

            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-local",
                    request_scope_key="req-local",
                    item_index=1,
                )

            self.assertEqual(normalized.source_ref.transport, MediaTransport.LOCAL_PATH)
            self.assertEqual(normalized.source_ref.local_path, str(image_path.resolve()))
            self.assertEqual(normalized.local_materialized_path, str(image_path.resolve()))
            self.assertEqual(normalized.orig_size_hw, (512, 288))
            self.assertIsNotNone(normalized.byte_size)

    def test_request_capture_serialization(self) -> None:
        capture = RequestCapture(request_id="req-1", method="POST", path="/v1/chat/completions")
        capture.add_render_metadata(
            {
                "prompt": "hello world",
                "multi_modal_uuids": {"image": ["uuid-1", "uuid-2"]},
                "multi_modal_data": {"image": ["placeholder"]},
            }
        )
        capture.finalize(status_code=200)
        payload = capture.to_dict()

        self.assertTrue(payload["prompt_has_multimodal"])
        self.assertEqual(payload["prompt_text_length"], 11)
        self.assertEqual(payload["prompt_mm_uuid_counts"]["image"], 2)
        self.assertEqual(payload["status_code"], 200)


if __name__ == "__main__":
    unittest.main()
