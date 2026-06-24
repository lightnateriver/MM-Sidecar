from __future__ import annotations

import base64
import os
import tempfile
import unittest
from unittest import mock
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
from mm_sidecar.contracts.identity import (
    build_base64_source_key,
    build_http_source_key,
    build_local_source_key,
)
from mm_sidecar.contracts.media_source import MediaSourceRef
from mm_sidecar.sidecar import (
    InlineProcessorWorkerPool,
    MemoryCacheConfig,
    MultiProcessProcessorWorkerPool,
    SidecarManager,
    SidecarManagerConfig,
    SidecarState,
    WorkerPoolConfig,
)
from mm_sidecar.sidecar.processor import WorkerResult, WorkerTask
from mm_sidecar.sidecar.protocol import FallbackDescriptor


def _make_processor_signature() -> ProcessorSignature:
    return ProcessorSignature.from_config(
        ProcessorConfig(
            model_name="qwen3.5-vl",
            revision="stage-c-test",
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
        request_id="req-local",
        request_media_index=item_index,
        captured_image=_build_captured_image(normalized),
        ingress_limits=_make_limits(),
        processor_signature_value=_make_processor_signature().value,
        item_identity=f"local:{path.name}:{item_index}",
        orig_size_hw=normalized.orig_size_hw,
    )


def _build_base64_descriptor(item_index: int = 0) -> FallbackDescriptor:
    payload = base64.b64encode(_make_jpeg_bytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{payload}"
    decoded = base64.b64decode(payload)
    with Image.open(BytesIO(decoded)) as image:
        normalized = NormalizedImage(
            source_ref=MediaSourceRef(
                transport=MediaTransport.BASE64,
                source_key=build_base64_source_key("req-base64", item_index),
                media_uuid=f"uuid-base64-{item_index}",
                request_scope_key="req-base64",
                image_url=data_url,
                mime_type="image/jpeg",
            ),
            orig_size_hw=(image.height, image.width),
            mime_type="image/jpeg",
            byte_size=len(decoded),
            decoded_size_hw=(image.height, image.width),
        )
    return FallbackDescriptor(
        request_id="req-base64",
        request_media_index=item_index,
        captured_image=_build_captured_image(normalized),
        ingress_limits=_make_limits(),
        processor_signature_value=_make_processor_signature().value,
        item_identity=f"base64:item:{item_index}",
        orig_size_hw=normalized.orig_size_hw,
    )


def _build_http_descriptor(url: str, item_index: int = 0) -> FallbackDescriptor:
    with Image.open(BytesIO(_make_jpeg_bytes())) as image:
        normalized = NormalizedImage(
            source_ref=MediaSourceRef(
                transport=MediaTransport.HTTP,
                source_key=build_http_source_key(url),
                media_uuid=f"uuid-http-{item_index}",
                request_scope_key=None,
                image_url=url,
            ),
            orig_size_hw=(image.height, image.width),
            mime_type="image/jpeg",
            byte_size=None,
            decoded_size_hw=(image.height, image.width),
        )
    return FallbackDescriptor(
        request_id="req-http",
        request_media_index=item_index,
        captured_image=_build_captured_image(normalized),
        ingress_limits=_make_limits(),
        processor_signature_value=_make_processor_signature().value,
        item_identity=f"http:{item_index}",
        orig_size_hw=normalized.orig_size_hw,
        http_timeout_ms=5_000,
    )


class _ManualWorkerPool:
    def __init__(self) -> None:
        self.worker_count = 1
        self._results: list[WorkerResult] = []
        self._tasks: list[WorkerTask] = []

    def submit(self, task: WorkerTask) -> None:
        self._tasks.append(task)
        self._results.append(
            WorkerResult(
                cache_key=task.cache_key,
                epoch=task.epoch,
                worker_id=task.assigned_worker_id,
                event_type="started",
                at_ms=1.0,
            )
        )

    def finish_all_as_ready(self) -> None:
        for task in list(self._tasks):
            self._results.append(
                WorkerResult(
                    cache_key=task.cache_key,
                    epoch=task.epoch,
                    worker_id=task.assigned_worker_id,
                    event_type="ready",
                    at_ms=2.0,
                    descriptor=None,
                    payload=None,
                )
            )

    def poll(self, max_items: int | None = None) -> list[WorkerResult]:
        if max_items is None or max_items >= len(self._results):
            results = list(self._results)
            self._results.clear()
            return results
        results = self._results[:max_items]
        del self._results[:max_items]
        return results

    def close(self) -> None:
        self._results.clear()
        self._tasks.clear()


class _FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class SidecarRuntimeTests(unittest.TestCase):
    def test_local_path_ready_and_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "demo.jpg"
            image_path.write_bytes(_make_jpeg_bytes())
            worker_pool = InlineProcessorWorkerPool(worker_count=2)
            manager = SidecarManager(
                config=SidecarManagerConfig(
                    cache=MemoryCacheConfig(max_reusable_bytes=8 * 1024 * 1024),
                    workers=WorkerPoolConfig(worker_count=2),
                ),
                worker_pool=worker_pool,
            )
            descriptor = _build_local_descriptor(image_path)

            handles = manager.prepare([descriptor])
            snapshots = manager.wait_for_states(handles, {SidecarState.READY}, 500.0)
            self.assertEqual(snapshots[0].state, SidecarState.READY)
            artifact = manager.fetch_ready(handles[0])
            self.assertIsNotNone(artifact)
            assert artifact is not None
            self.assertEqual(artifact.descriptor.payload_dtype, "float32")
            self.assertEqual(artifact.descriptor.payload_shape, (720, 588))
            self.assertIsNotNone(artifact.timings_ms)
            assert artifact.timings_ms is not None
            self.assertIn("source", artifact.timings_ms)
            self.assertIn("decode", artifact.timings_ms)
            self.assertIn("preprocess", artifact.timings_ms)
            self.assertIn("total", artifact.timings_ms)
            self.assertEqual(worker_pool.submission_count, 1)

            second_handles = manager.prepare([descriptor])
            self.assertEqual(worker_pool.submission_count, 1)
            self.assertEqual(second_handles[0].cache_key, handles[0].cache_key)
            manager.close()

    def test_base64_transport_reaches_ready(self) -> None:
        manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
        descriptor = _build_base64_descriptor()
        handles = manager.prepare([descriptor])
        snapshots = manager.wait_for_states(handles, {SidecarState.READY}, 500.0)
        self.assertEqual(snapshots[0].state, SidecarState.READY)
        artifact = manager.fetch_ready(handles[0])
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.descriptor.payload_dtype, "float32")
        manager.close()

    def test_http_transport_reaches_ready(self) -> None:
        payload = _make_jpeg_bytes()
        manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
        descriptor = _build_http_descriptor("http://example.com/demo.jpg")
        with mock.patch(
            "mm_sidecar.sidecar.processor.urllib.request.urlopen",
            return_value=_FakeHttpResponse(payload),
        ):
            handles = manager.prepare([descriptor])
            snapshots = manager.wait_for_states(handles, {SidecarState.READY}, 500.0)
            self.assertEqual(snapshots[0].state, SidecarState.READY)
            artifact = manager.fetch_ready(handles[0])
            self.assertIsNotNone(artifact)
        manager.close()

    def test_fallback_claim_discards_stale_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "claim.jpg"
            image_path.write_bytes(_make_jpeg_bytes())
            worker_pool = _ManualWorkerPool()
            manager = SidecarManager(worker_pool=worker_pool)
            descriptor = _build_local_descriptor(image_path)

            handles = manager.prepare([descriptor])
            snapshots = manager.batch_get_status(handles)
            self.assertEqual(snapshots[0].state, SidecarState.SIDECAR_RUNNING)

            claim_results = manager.try_fallback_claim(handles, "rank-2")
            self.assertTrue(claim_results[0].granted)
            self.assertEqual(claim_results[0].state, SidecarState.FALLBACK_CLAIMED)

            worker_pool.finish_all_as_ready()
            snapshots = manager.batch_get_status([claim_results[0].handle])
            self.assertEqual(snapshots[0].state, SidecarState.FALLBACK_CLAIMED)
            self.assertIsNone(manager.fetch_ready(claim_results[0].handle))
            manager.close()

    def test_multiprocess_worker_pool_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "proc.jpg"
            image_path.write_bytes(_make_jpeg_bytes())
            allowed_cpus = tuple(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else (0,)
            cpu_map = ((allowed_cpus[0],),)
            worker_pool = MultiProcessProcessorWorkerPool(
                WorkerPoolConfig(
                    worker_count=1,
                    cpu_affinity_map=cpu_map,
                    start_method="fork",
                )
            )
            manager = SidecarManager(
                config=SidecarManagerConfig(
                    cache=MemoryCacheConfig(max_reusable_bytes=8 * 1024 * 1024),
                    workers=WorkerPoolConfig(worker_count=1),
                ),
                worker_pool=worker_pool,
            )
            descriptor = _build_local_descriptor(image_path)
            handles = manager.prepare([descriptor])
            snapshots = manager.wait_for_states(handles, {SidecarState.READY}, 2_000.0)
            self.assertEqual(snapshots[0].state, SidecarState.READY)
            artifact = manager.fetch_ready(handles[0])
            self.assertIsNotNone(artifact)
            manager.close()


if __name__ == "__main__":
    unittest.main()
