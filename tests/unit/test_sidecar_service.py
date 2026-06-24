from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from mm_sidecar.contracts import (
    CapturedImageRef,
    IngressLimits,
    MediaTransport,
    NormalizedImage,
    ProcessorConfig,
    ProcessorSignature,
)
from mm_sidecar.contracts.identity import build_local_source_key
from mm_sidecar.contracts.media_source import MediaSourceRef
from mm_sidecar.sidecar import (
    SidecarServiceConfig,
    SidecarServiceProcess,
    SidecarState,
)
from mm_sidecar.sidecar.config import MemoryCacheConfig, SidecarManagerConfig, WorkerPoolConfig
from mm_sidecar.sidecar.protocol import FallbackDescriptor


def _make_processor_signature() -> ProcessorSignature:
    return ProcessorSignature.from_config(
        ProcessorConfig(
            model_name="qwen3.5-vl",
            revision="sidecar-service-test",
            processor_name="qwen-basic",
            patch_size=14,
            merge_size=2,
            temporal_patch_size=1,
            min_pixels=4,
            max_pixels=288 * 512,
        )
    )


def _make_limits() -> IngressLimits:
    return IngressLimits(
        max_image_count=40,
        max_encoded_bytes=8 * 1024 * 1024,
        max_decoded_bytes=16 * 1024 * 1024,
        max_pixels_per_image=4 * 1024 * 1024,
    )


def _make_jpeg_bytes(size: tuple[int, int] = (288, 512)) -> bytes:
    image = Image.new("RGB", size, color=(12, 34, 56))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _build_captured_image(normalized: NormalizedImage) -> CapturedImageRef:
    return CapturedImageRef(
        source_ref=normalized.source_ref,
        mime_type=normalized.mime_type,
        byte_size=normalized.byte_size,
        local_materialized_path=normalized.local_materialized_path,
    )


def _build_local_descriptor(path: Path, item_index: int = 0) -> FallbackDescriptor:
    stat_result = path.stat()
    with Image.open(path) as image:
        normalized = NormalizedImage(
            source_ref=MediaSourceRef(
                transport=MediaTransport.LOCAL_PATH,
                source_key=build_local_source_key(
                    str(path),
                    mtime_ns=stat_result.st_mtime_ns,
                    size_bytes=stat_result.st_size,
                ),
                media_uuid=f"uuid-local-{item_index}",
                request_scope_key=None,
                local_path=str(path.resolve()),
            ),
            orig_size_hw=(image.height, image.width),
            mime_type="image/jpeg",
            byte_size=stat_result.st_size,
            decoded_size_hw=(image.height, image.width),
            local_materialized_path=str(path.resolve()),
        )
    return FallbackDescriptor(
        request_id="req-local-service",
        request_media_index=item_index,
        captured_image=_build_captured_image(normalized),
        ingress_limits=_make_limits(),
        processor_signature_value=_make_processor_signature().value,
        item_identity=f"local:{path.name}:{item_index}",
        orig_size_hw=normalized.orig_size_hw,
    )


class SidecarServiceTests(unittest.TestCase):
    def test_service_prepare_wait_metadata_and_fetch_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "service.jpg"
            image_path.write_bytes(_make_jpeg_bytes())
            service = SidecarServiceProcess(
                SidecarServiceConfig(
                    worker_pool_mode="inline",
                    manager=SidecarManagerConfig(
                        cache=MemoryCacheConfig(max_reusable_bytes=8 * 1024 * 1024),
                        workers=WorkerPoolConfig(worker_count=1),
                    ),
                )
            )
            client = service.start()
            try:
                descriptor = _build_local_descriptor(image_path)
                handles = client.prepare([descriptor])
                metadata = client.wait_for_metadata(handles, timeout_ms=500.0)
                self.assertEqual(len(metadata), 1)
                self.assertIsNotNone(metadata[0].schedule_item)
                self.assertIn(
                    metadata[0].state,
                    {SidecarState.SIDECAR_RUNNING, SidecarState.READY},
                )

                ready = client.wait_for_states(handles, {SidecarState.READY}, 500.0)
                self.assertEqual(ready[0].state, SidecarState.READY)
                artifact = client.fetch_ready(handles[0])
                self.assertIsNotNone(artifact)
                assert artifact is not None
                self.assertEqual(artifact.payload.image_grid_thw, (1, 36, 20))
            finally:
                try:
                    client.shutdown()
                finally:
                    service.join(timeout=2.0)
                    service.terminate()

    def test_service_process_worker_pool_can_spawn_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "service-process.jpg"
            image_path.write_bytes(_make_jpeg_bytes())
            service = SidecarServiceProcess(
                SidecarServiceConfig(
                    worker_pool_mode="process",
                    start_method="fork",
                    manager=SidecarManagerConfig(
                        cache=MemoryCacheConfig(max_reusable_bytes=8 * 1024 * 1024),
                        workers=WorkerPoolConfig(
                            worker_count=2,
                            cpu_affinity_map=None,
                            start_method="fork",
                        ),
                    ),
                )
            )
            client = service.start()
            try:
                descriptor = _build_local_descriptor(image_path)
                handles = client.prepare([descriptor])
                ready = client.wait_for_states(handles, {SidecarState.READY}, 1000.0)
                self.assertEqual(ready[0].state, SidecarState.READY)
                artifact = client.fetch_ready(handles[0])
                self.assertIsNotNone(artifact)
                assert artifact is not None
                self.assertEqual(artifact.payload.image_grid_thw, (1, 36, 20))
            finally:
                try:
                    client.shutdown()
                finally:
                    service.join(timeout=2.0)
                    service.terminate()

    def test_service_terminate_uses_graceful_shutdown_for_process_pool(self) -> None:
        service = SidecarServiceProcess(
            SidecarServiceConfig(
                worker_pool_mode="process",
                start_method="fork",
                manager=SidecarManagerConfig(
                    cache=MemoryCacheConfig(max_reusable_bytes=8 * 1024 * 1024),
                    workers=WorkerPoolConfig(worker_count=2, start_method="fork"),
                ),
            )
        )
        client = service.start()
        self.assertEqual(client.stats().queued_items, 0)

        service.terminate()
        service.join(timeout=2.0)

        self.assertFalse(service._process.is_alive())


if __name__ == "__main__":
    unittest.main()
