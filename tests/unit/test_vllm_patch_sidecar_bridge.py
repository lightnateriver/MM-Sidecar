from __future__ import annotations

import base64
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

from PIL import Image

from mm_sidecar.contracts import (
    CapturedImageRef,
    ImageScheduleItem,
    MediaTransport,
    ProcessorSignature,
)
from mm_sidecar.integrations.vllm_patch.context import RequestCapture
from mm_sidecar.integrations.vllm_patch.normalization import (
    build_normalized_image_from_url,
)
from mm_sidecar.integrations.vllm_patch.sidecar_bridge import (
    build_fallback_descriptors,
    prepare_capture_for_sidecar,
    prepare_single_capture_item_for_sidecar,
)
from mm_sidecar.sidecar import InlineProcessorWorkerPool, SidecarManager
from mm_sidecar.sidecar.protocol import SidecarState, SidecarStatusSnapshot


def _make_image() -> Image.Image:
    return Image.new("RGB", (288, 512), color=(90, 45, 12))


def _make_base64_data_url() -> str:
    image = _make_image()
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


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
    media_io_kwargs = {
        "image": {
            "headers": {
                "Authorization": "Bearer test-token",
                "X-Trace-Id": "trace-123",
            }
        }
    }


class VllmPatchSidecarBridgeTests(unittest.TestCase):
    def test_build_fallback_descriptors_plans_all_transports(self) -> None:
        capture = RequestCapture(
            request_id="req-bridge",
            method="POST",
            path="/v1/chat/completions",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "local.jpg"
            _make_image().save(image_path, format="JPEG")

            with Image.open(image_path) as local_image:
                local_normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=local_image,
                    media_uuid="uuid-local",
                    request_scope_key="req-bridge",
                    item_index=0,
                )

            http_image = _make_image()
            http_image.format = "JPEG"
            http_normalized = build_normalized_image_from_url(
                image_url="https://example.com/demo.jpg",
                image=http_image,
                media_uuid="uuid-http",
                request_scope_key="req-bridge",
                item_index=1,
            )

            base64_image = _make_image()
            base64_image.format = "JPEG"
            base64_normalized = build_normalized_image_from_url(
                image_url=_make_base64_data_url(),
                image=base64_image,
                media_uuid="uuid-base64",
                request_scope_key="req-bridge",
                item_index=2,
            )

            capture.add_normalized_image(0, "uuid-local", local_normalized)
            capture.add_normalized_image(1, "uuid-http", http_normalized)
            capture.add_normalized_image(2, "uuid-base64", base64_normalized)

            descriptors = build_fallback_descriptors(
                capture=capture,
                renderer=_FakeRenderer(),
                params=_FakeParams(),
            )

        self.assertEqual(len(descriptors), 3)
        self.assertEqual(
            [descriptor.captured_image.source_ref.transport for descriptor in descriptors],
            [
                MediaTransport.LOCAL_PATH,
                MediaTransport.HTTP,
                MediaTransport.BASE64,
            ],
        )
        self.assertEqual(
            [descriptor.request_media_index for descriptor in descriptors],
            [0, 1, 2],
        )
        self.assertTrue(all(descriptor.item_identity for descriptor in descriptors))
        self.assertTrue(all(descriptor.processor_signature_value for descriptor in descriptors))
        http_descriptor = next(
            descriptor
            for descriptor in descriptors
            if descriptor.captured_image.source_ref.transport is MediaTransport.HTTP
        )
        self.assertEqual(
            http_descriptor.http_headers,
            (
                ("Authorization", "Bearer test-token"),
                ("X-Trace-Id", "trace-123"),
            ),
        )
        for descriptor in descriptors:
            self.assertIsNotNone(descriptor.orig_size_hw)
            self.assertEqual(descriptor.orig_size_hw, (512, 288))

    def test_build_fallback_descriptors_accepts_descriptor_only_capture(self) -> None:
        capture = RequestCapture(
            request_id="req-descriptor-only",
            method="POST",
            path="/v1/chat/completions",
        )

        image_ref = build_normalized_image_from_url(
            image_url="https://example.com/descriptor-only.jpg",
            image=_make_image(),
            media_uuid="uuid-descriptor-only",
            request_scope_key="req-descriptor-only",
            item_index=0,
        ).source_ref
        capture.add_captured_image_ref(
            item_index=0,
            media_uuid="uuid-descriptor-only",
            image_ref=CapturedImageRef(
                source_ref=image_ref,
                mime_type="image/jpeg",
                byte_size=1234,
                local_materialized_path=None,
            ),
        )

        descriptors = build_fallback_descriptors(
            capture=capture,
            renderer=_FakeRenderer(),
            params=_FakeParams(),
        )

        self.assertEqual(len(descriptors), 1)
        self.assertEqual(descriptors[0].request_media_index, 0)
        self.assertEqual(descriptors[0].item_identity, image_ref.source_key)
        self.assertIsNone(descriptors[0].orig_size_hw)

    def test_prepare_single_capture_item_for_sidecar_stores_handle(self) -> None:
        capture = RequestCapture(
            request_id="req-single-prepare",
            method="POST",
            path="/v1/chat/completions",
            sidecar_manager=SidecarManager(worker_pool=InlineProcessorWorkerPool()),
        )
        image_ref = build_normalized_image_from_url(
            image_url="https://example.com/single-prepare.jpg",
            image=_make_image(),
            media_uuid="uuid-single-prepare",
            request_scope_key="req-single-prepare",
            item_index=0,
        ).source_ref

        result = prepare_single_capture_item_for_sidecar(
            capture=capture,
            renderer=_FakeRenderer(),
            params=_FakeParams(),
            item_index=0,
            captured_ref=CapturedImageRef(
                source_ref=image_ref,
                mime_type="image/jpeg",
                byte_size=1234,
                local_materialized_path=None,
            ),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result["prepared"])
        self.assertEqual(capture.get_prepared_handle(0), result["handle"])
        self.assertIsNotNone(capture.get_prepared_descriptor(0))

    def test_prepare_capture_for_sidecar_stores_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "prepare.jpg"
            _make_image().save(image_path, format="JPEG")

            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-local-prepare",
                    request_scope_key="req-prepare",
                    item_index=0,
                )

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
            capture = RequestCapture(
                request_id="req-prepare",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=manager,
            )
            capture.add_normalized_image(0, "uuid-local-prepare", normalized)

            payload = prepare_capture_for_sidecar(
                capture=capture,
                renderer=_FakeRenderer(),
                params=_FakeParams(),
            )

            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertTrue(payload["enabled"])
            self.assertEqual(payload["prepared_image_count"], 1)
            self.assertEqual(len(payload["planned_items"]), 1)
            self.assertEqual(payload["planned_items"][0]["transport"], "local_path")
            self.assertIn("source_plan_preview", payload)
            self.assertEqual(payload["source_plan_preview"]["request_id"], "req-prepare")
            self.assertEqual(len(payload["source_plan_preview"]["entries"]), 1)
            self.assertIn("timings_ms", payload)
            self.assertGreaterEqual(payload["timings_ms"]["total"], 0.0)
            self.assertGreater(payload["total_placeholder_token_count"], 0)
            self.assertEqual(len(payload["handles"]), 1)
            self.assertEqual(len(payload["initial_statuses"]), 1)
            self.assertIn("timings_ms", payload["initial_statuses"][0])
            self.assertIs(capture.sidecar_prepare, payload)
            self.assertEqual(payload["planned_items"][0]["transport"], "local_path")
            self.assertEqual(payload["planned_items"][0]["request_media_index"], 0)

            cached_payload = prepare_capture_for_sidecar(
                capture=capture,
                renderer=_FakeRenderer(),
                params=_FakeParams(),
            )
            self.assertIs(cached_payload, payload)
            manager.close()

    def test_prepare_capture_for_sidecar_uses_sidecar_metadata_without_normalized_image(self) -> None:
        manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
        capture = RequestCapture(
            request_id="req-descriptor-metadata",
            method="POST",
            path="/v1/chat/completions",
            sidecar_manager=manager,
        )

        image_ref = build_normalized_image_from_url(
            image_url=_make_base64_data_url(),
            image=_make_image(),
            media_uuid="uuid-descriptor-metadata",
            request_scope_key="req-descriptor-metadata",
            item_index=0,
        ).source_ref

        capture.add_captured_image_ref(
            item_index=0,
            media_uuid="uuid-descriptor-metadata",
            image_ref=CapturedImageRef(
                source_ref=image_ref,
                mime_type="image/jpeg",
                byte_size=1234,
                local_materialized_path=None,
            ),
        )

        payload = prepare_capture_for_sidecar(
            capture=capture,
            renderer=_FakeRenderer(),
            params=_FakeParams(),
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["prepared_image_count"], 1)
        self.assertEqual(len(payload["planned_items"]), 1)
        self.assertEqual(payload["planned_items"][0]["request_media_index"], 0)
        self.assertGreater(payload["planned_items"][0]["placeholder_token_count"], 0)
        self.assertEqual(payload["planned_items"][0]["transport"], "base64")
        self.assertGreater(payload["total_placeholder_token_count"], 0)
        manager.close()

    def test_prepare_capture_for_sidecar_backfills_partial_metadata(self) -> None:
        manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
        capture = RequestCapture(
            request_id="req-partial-metadata",
            method="POST",
            path="/v1/chat/completions",
            sidecar_manager=manager,
        )

        images = []
        for item_index in range(3):
            image = _make_image()
            image.format = "JPEG"
            normalized = build_normalized_image_from_url(
                image_url=_make_base64_data_url(),
                image=image,
                media_uuid=f"uuid-partial-{item_index}",
                request_scope_key="req-partial-metadata",
                item_index=item_index,
            )
            capture.add_normalized_image(
                item_index,
                f"uuid-partial-{item_index}",
                normalized,
            )
            images.append(normalized)

        descriptors = build_fallback_descriptors(
            capture=capture,
            renderer=_FakeRenderer(),
            params=_FakeParams(),
        )
        handles = manager.prepare(descriptors)

        first_schedule_item = ImageScheduleItem(
            item_index=0,
            item_identity=descriptors[0].item_identity,
            processor_signature=ProcessorSignature(
                value=descriptors[0].processor_signature_value
            ),
            orig_size_hw=images[0].orig_size_hw,
            preprocessed_size_hw=images[0].orig_size_hw,
            image_grid_thw=(1, 32, 18),
            placeholder_token_count=144,
        )

        partial_snapshots = (
            SidecarStatusSnapshot(
                handle=handles[0],
                state=SidecarState.SIDECAR_RUNNING,
                epoch=handles[0].epoch,
                updated_at_ms=0.0,
                schedule_item=first_schedule_item,
            ),
            SidecarStatusSnapshot(
                handle=handles[1],
                state=SidecarState.SIDECAR_RUNNING,
                epoch=handles[1].epoch,
                updated_at_ms=0.0,
                schedule_item=None,
            ),
            SidecarStatusSnapshot(
                handle=handles[2],
                state=SidecarState.SIDECAR_RUNNING,
                epoch=handles[2].epoch,
                updated_at_ms=0.0,
                schedule_item=None,
            ),
        )

        with mock.patch.object(
            manager,
            "wait_for_metadata",
            return_value=partial_snapshots,
        ), mock.patch.object(
            manager,
            "batch_get_status",
            return_value=partial_snapshots,
        ):
            payload = prepare_capture_for_sidecar(
                capture=capture,
                renderer=_FakeRenderer(),
                params=_FakeParams(),
            )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["prepared_image_count"], 3)
        self.assertEqual(len(payload["planned_items"]), 3)
        self.assertEqual(
            [item["request_media_index"] for item in payload["planned_items"]],
            [0, 1, 2],
        )
        self.assertTrue(
            all(item["placeholder_token_count"] > 0 for item in payload["planned_items"])
        )
        manager.close()

    def test_prepare_capture_for_sidecar_descriptor_only_retries_partial_metadata(self) -> None:
        manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
        capture = RequestCapture(
            request_id="req-descriptor-retry",
            method="POST",
            path="/v1/chat/completions",
            sidecar_manager=manager,
        )

        source_refs = []
        for item_index in range(3):
            normalized = build_normalized_image_from_url(
                image_url=_make_base64_data_url(),
                image=_make_image(),
                media_uuid=f"uuid-descriptor-retry-{item_index}",
                request_scope_key="req-descriptor-retry",
                item_index=item_index,
            )
            source_refs.append(normalized.source_ref)
            capture.add_captured_image_ref(
                item_index=item_index,
                media_uuid=f"uuid-descriptor-retry-{item_index}",
                image_ref=CapturedImageRef(
                    source_ref=normalized.source_ref,
                    mime_type="image/jpeg",
                    byte_size=1234,
                    local_materialized_path=None,
                ),
            )

        descriptors = build_fallback_descriptors(
            capture=capture,
            renderer=_FakeRenderer(),
            params=_FakeParams(),
        )
        handles = manager.prepare(descriptors)

        def _schedule_item(index: int) -> ImageScheduleItem:
            return ImageScheduleItem(
                item_index=index,
                item_identity=descriptors[index].item_identity,
                processor_signature=ProcessorSignature(
                    value=descriptors[index].processor_signature_value
                ),
                orig_size_hw=(512, 288),
                preprocessed_size_hw=(532, 308),
                image_grid_thw=(1, 38, 22),
                placeholder_token_count=209,
            )

        partial_snapshots = (
            SidecarStatusSnapshot(
                handle=handles[0],
                state=SidecarState.SIDECAR_RUNNING,
                epoch=handles[0].epoch,
                updated_at_ms=0.0,
                schedule_item=_schedule_item(0),
            ),
            SidecarStatusSnapshot(
                handle=handles[1],
                state=SidecarState.SIDECAR_RUNNING,
                epoch=handles[1].epoch,
                updated_at_ms=0.0,
                schedule_item=None,
            ),
            SidecarStatusSnapshot(
                handle=handles[2],
                state=SidecarState.SIDECAR_RUNNING,
                epoch=handles[2].epoch,
                updated_at_ms=0.0,
                schedule_item=None,
            ),
        )
        full_snapshots = tuple(
            SidecarStatusSnapshot(
                handle=handles[item_index],
                state=SidecarState.SIDECAR_RUNNING,
                epoch=handles[item_index].epoch,
                updated_at_ms=0.0,
                schedule_item=_schedule_item(item_index),
            )
            for item_index in range(3)
        )

        with mock.patch.object(
            manager,
            "wait_for_metadata",
            side_effect=[partial_snapshots, full_snapshots],
        ) as wait_for_metadata_mock, mock.patch.object(
            manager,
            "batch_get_status",
            return_value=full_snapshots,
        ), mock.patch.dict(
            os.environ,
            {
                "MM_SIDECAR_DESCRIPTOR_ONLY_CAPTURE": "1",
                "MM_SIDECAR_DESCRIPTOR_ONLY_METADATA_WAIT_MS": "10.0",
            },
            clear=False,
        ):
            payload = prepare_capture_for_sidecar(
                capture=capture,
                renderer=_FakeRenderer(),
                params=_FakeParams(),
            )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["prepared_image_count"], 3)
        self.assertEqual(len(payload["planned_items"]), 3)
        self.assertEqual(wait_for_metadata_mock.call_count, 2)
        self.assertGreaterEqual(
            payload["timings_ms"]["descriptor_only_metadata_retry"],
            0.0,
        )
        self.assertEqual(capture.errors, [])
        manager.close()

    def test_prepare_capture_for_sidecar_without_manager_keeps_fallback_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "no-manager.jpg"
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

            payload = prepare_capture_for_sidecar(
                capture=capture,
                renderer=_FakeRenderer(),
                params=_FakeParams(),
            )

            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertFalse(payload["enabled"])
            self.assertEqual(payload["reason"], "sidecar_manager_unavailable")
            self.assertEqual(payload["prepared_image_count"], 1)
            self.assertEqual(len(capture.iter_prepared_sidecar_items()), 1)
            descriptor = capture.get_prepared_descriptor(0)
            self.assertIsNotNone(descriptor)
            self.assertEqual(descriptor.request_media_index, 0)
            self.assertIsNone(capture.get_prepared_handle(0))


if __name__ == "__main__":
    unittest.main()
