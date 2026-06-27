from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from unittest import mock

from PIL import Image

from mm_sidecar.integrations.vllm_patch.carrier import (
    REQUEST_PAYLOAD_KEY,
    attach_sidecar_payload_to_params,
    build_request_sidecar_payload,
    decode_sidecar_request_plan,
)
from mm_sidecar.integrations.vllm_patch.context import RequestCapture
from mm_sidecar.integrations.vllm_patch.normalization import (
    build_normalized_image_from_url,
)
from mm_sidecar.integrations.vllm_patch.sidecar_bridge import prepare_capture_for_sidecar
from mm_sidecar.sidecar import InlineProcessorWorkerPool, SidecarManager


def _make_image() -> Image.Image:
    return Image.new("RGB", (288, 512), color=(90, 45, 12))


class _FakeVisionConfig:
    patch_size = 14
    spatial_merge_size = 2
    temporal_patch_size = 1


class _FakeHFConfig:
    vision_config = _FakeVisionConfig()
    _name_or_path = "fake-qwen3.5-vl"
    _commit_hash = "fake-rev"


class _FakeModelConfig:
    hf_config = _FakeHFConfig()
    model = "fake-qwen3.5-vl"
    revision = "fake-rev"


class _FakeRenderer:
    model_config = _FakeModelConfig()


class _FakeParams:
    mm_processor_kwargs = {
        "do_resize": True,
        "min_pixels": 28 * 28,
        "max_pixels": 1280 * 28 * 28,
    }
    media_io_kwargs = {}
    extra_args = None


@dataclass(frozen=True)
class _FrozenParams:
    mm_processor_kwargs: dict
    media_io_kwargs: dict
    extra_args: dict | None = None


class VllmPatchCarrierTests(unittest.TestCase):
    def test_attach_payload_to_params_and_decode_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "carrier.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-carrier",
                    request_scope_key="req-carrier",
                    item_index=0,
                )

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
            capture = RequestCapture(
                request_id="req-carrier",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=manager,
            )
            capture.add_normalized_image(0, "uuid-carrier", normalized)
            with mock.patch.dict(
                os.environ,
                {"MM_SIDECAR_MIN_IMAGE_COUNT": "1"},
                clear=False,
            ):
                prepare_capture_for_sidecar(capture, _FakeRenderer(), _FakeParams())

            params = _FakeParams()
            payload = attach_sidecar_payload_to_params(params, capture)
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertIsInstance(params.extra_args, dict)
            self.assertIn(REQUEST_PAYLOAD_KEY, params.extra_args)

            plan = decode_sidecar_request_plan(params.extra_args[REQUEST_PAYLOAD_KEY])
            self.assertTrue(plan.enabled)
            self.assertEqual(plan.request_id, "req-carrier")
            self.assertEqual(plan.prepared_image_count, 1)
            self.assertEqual(len(plan.fallback_descriptors), 1)
            self.assertEqual(len(plan.handles), 1)
            self.assertEqual(plan.fallback_descriptors[0].request_media_index, 0)
            self.assertGreater(plan.total_placeholder_token_count, 0)
            manager.close()

    def test_attach_payload_to_frozen_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "carrier-frozen.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-frozen",
                    request_scope_key="req-frozen",
                    item_index=0,
                )

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
            capture = RequestCapture(
                request_id="req-frozen",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=manager,
            )
            capture.add_normalized_image(0, "uuid-frozen", normalized)
            params = _FrozenParams(
                mm_processor_kwargs=_FakeParams.mm_processor_kwargs,
                media_io_kwargs={},
            )
            with mock.patch.dict(
                os.environ,
                {"MM_SIDECAR_MIN_IMAGE_COUNT": "1"},
                clear=False,
            ):
                prepare_capture_for_sidecar(capture, _FakeRenderer(), params)

            payload = attach_sidecar_payload_to_params(params, capture)

            self.assertIsNotNone(payload)
            self.assertIsInstance(params.extra_args, dict)
            assert params.extra_args is not None
            self.assertIn(REQUEST_PAYLOAD_KEY, params.extra_args)
            manager.close()

    def test_build_request_payload_when_manager_unavailable_keeps_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "carrier-no-manager.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-no-manager",
                    request_scope_key="req-no-manager",
                    item_index=0,
                )

            capture = RequestCapture(
                request_id="req-no-manager",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=None,
            )
            capture.add_normalized_image(0, "uuid-no-manager", normalized)
            payload = prepare_capture_for_sidecar(capture, _FakeRenderer(), _FakeParams())
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertFalse(payload["enabled"])
            request_payload = build_request_sidecar_payload(capture)
            self.assertIsNotNone(request_payload)
            assert request_payload is not None
            self.assertEqual(request_payload["prepared_image_count"], 1)
            self.assertEqual(len(request_payload["fallback_descriptors"]), 1)
            self.assertEqual(len(request_payload["handles"]), 0)
            plan = decode_sidecar_request_plan(request_payload)
            self.assertFalse(plan.enabled)
            self.assertEqual(len(plan.fallback_descriptors), 1)
            self.assertEqual(plan.fallback_descriptors[0].request_media_index, 0)


if __name__ == "__main__":
    unittest.main()
