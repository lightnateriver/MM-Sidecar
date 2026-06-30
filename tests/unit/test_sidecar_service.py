from __future__ import annotations

import os
import tempfile
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

from PIL import Image

from mm_sidecar.contracts import (
    CapturedImageRef,
    IngressLimits,
    LocalFileTensorPayloadRef,
    MediaTransport,
    NormalizedImage,
    ProcessorConfig,
    ProcessorSignature,
    StorageKind,
)
from mm_sidecar.contracts.identity import build_local_source_key
from mm_sidecar.contracts.media_source import MediaSourceRef
from mm_sidecar.sidecar import (
    SidecarServiceConfig,
    SidecarServiceProcess,
    SidecarState,
)
from mm_sidecar.sidecar.config import MemoryCacheConfig, SidecarManagerConfig, WorkerPoolConfig
from mm_sidecar.sidecar.artifact_store import load_local_file_tensor_ref
from mm_sidecar.sidecar.protocol import FallbackDescriptor
from mm_sidecar.sidecar.coordinator import build_ranked_claimer_id
from mm_sidecar.sidecar.processor import run_descriptor_locally


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


def _wait_for_ready_artifact(
    client,
    handle,
    *,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.02,
):
    deadline = time.monotonic() + timeout_s
    artifact = client.fetch_ready(handle)
    while artifact is None and time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        artifact = client.fetch_ready(handle)
    return artifact


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
                self.assertIsNotNone(artifact.fetch_diagnostics_ms)
                assert artifact.fetch_diagnostics_ms is not None
                self.assertIn("client_rpc_total", artifact.fetch_diagnostics_ms)
                self.assertIn("manager_fetch_total", artifact.fetch_diagnostics_ms)
                self.assertNotIn(
                    "manager_fetch_batch_count",
                    artifact.fetch_diagnostics_ms,
                )
            finally:
                try:
                    client.shutdown()
                finally:
                    service.join(timeout=2.0)
                    service.terminate()

    def test_service_fetch_ready_batch_returns_ordered_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = [
                Path(tmpdir) / "service-batch-0.jpg",
                Path(tmpdir) / "service-batch-1.jpg",
            ]
            for image_path in image_paths:
                image_path.write_bytes(_make_jpeg_bytes())
            service = SidecarServiceProcess(
                SidecarServiceConfig(
                    worker_pool_mode="inline",
                    manager=SidecarManagerConfig(
                        cache=MemoryCacheConfig(max_reusable_bytes=16 * 1024 * 1024),
                        workers=WorkerPoolConfig(worker_count=1),
                    ),
                )
            )
            client = service.start()
            try:
                descriptors = [
                    _build_local_descriptor(image_path, index)
                    for index, image_path in enumerate(image_paths)
                ]
                handles = client.prepare(descriptors)
                ready = client.wait_for_states(handles, {SidecarState.READY}, 500.0)
                self.assertEqual([snapshot.state for snapshot in ready], [SidecarState.READY] * 2)

                artifacts = client.fetch_ready_batch(handles)
                self.assertEqual(len(artifacts), 2)
                self.assertTrue(all(artifact is not None for artifact in artifacts))
                self.assertEqual(
                    [artifact.handle.request_media_index for artifact in artifacts if artifact is not None],
                    [0, 1],
                )
                batch_count = 0.0
                for artifact in artifacts:
                    assert artifact is not None
                    self.assertEqual(artifact.payload.image_grid_thw, (1, 36, 20))
                    self.assertIsNotNone(artifact.fetch_diagnostics_ms)
                    assert artifact.fetch_diagnostics_ms is not None
                    self.assertIn("client_rpc_total", artifact.fetch_diagnostics_ms)
                    self.assertIn("client_rpc_batch_count", artifact.fetch_diagnostics_ms)
                    self.assertIn("manager_fetch_batch_count", artifact.fetch_diagnostics_ms)
                    batch_count += artifact.fetch_diagnostics_ms["client_rpc_batch_count"]
                self.assertAlmostEqual(batch_count, 1.0)
            finally:
                try:
                    client.shutdown()
                finally:
                    service.join(timeout=2.0)
                    service.terminate()

    def test_service_records_worker_fetch_profiles(self) -> None:
        service = SidecarServiceProcess(
            SidecarServiceConfig(
                worker_pool_mode="inline",
                manager=SidecarManagerConfig(
                    workers=WorkerPoolConfig(worker_count=1),
                ),
            )
        )
        client = service.start()
        try:
            client.record_worker_fetch_profile(
                "req-profile",
                {
                    "profile_type": "vit_dp_shard_fetch",
                    "client_rpc_total": 12.5,
                    "fetch_ms": 3.25,
                    "worker_fetch_total_ms": 9.75,
                    "local_request_media_indexes": [0, 2],
                },
            )
            profiles = client.list_worker_fetch_profiles("req-profile")
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0]["request_id"], "req-profile")
            self.assertEqual(profiles[0]["profile_type"], "vit_dp_shard_fetch")
            self.assertEqual(profiles[0]["client_rpc_total"], 12.5)
            self.assertEqual(profiles[0]["local_request_media_indexes"], [0.0, 2.0])
            self.assertIn("observed_at_ms", profiles[0])
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
                self.assertIsNotNone(artifact.fetch_diagnostics_ms)
                assert artifact.fetch_diagnostics_ms is not None
                self.assertIn("client_rpc_total", artifact.fetch_diagnostics_ms)
                self.assertIn("manager_fetch_total", artifact.fetch_diagnostics_ms)
                self.assertIn(
                    "worker_to_manager_receive_ms",
                    artifact.fetch_diagnostics_ms,
                )
                self.assertIn(
                    "worker_to_manager_cache_done_ms",
                    artifact.fetch_diagnostics_ms,
                )
            finally:
                try:
                    client.shutdown()
                finally:
                    service.join(timeout=2.0)
                    service.terminate()

    def test_service_process_worker_pool_can_return_local_file_payload_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "service-local-file.jpg"
            image_path.write_bytes(_make_jpeg_bytes())
            payload_dir = Path(tmpdir) / "payloads"
            service = SidecarServiceProcess(
                SidecarServiceConfig(
                    worker_pool_mode="process",
                    start_method="spawn",
                    manager=SidecarManagerConfig(
                        cache=MemoryCacheConfig(max_reusable_bytes=8 * 1024 * 1024),
                        workers=WorkerPoolConfig(worker_count=1, start_method="spawn"),
                    ),
                )
            )
            with mock.patch.dict(
                os.environ,
                {
                    "MM_SIDECAR_PAYLOAD_STORAGE": "local_file",
                    "MM_SIDECAR_PAYLOAD_DIR": str(payload_dir),
                },
            ):
                client = service.start()
                ref_path: Path | None = None
                try:
                    descriptor = _build_local_descriptor(image_path)
                    handles = client.prepare([descriptor])
                    artifact = _wait_for_ready_artifact(client, handles[0])
                    self.assertIsNotNone(artifact)
                    assert artifact is not None
                    self.assertEqual(artifact.descriptor.storage_kind, StorageKind.LOCAL_FILE)
                    self.assertEqual(artifact.payload.storage_kind, StorageKind.LOCAL_FILE)
                    self.assertIsInstance(
                        artifact.payload.pixel_values,
                        LocalFileTensorPayloadRef,
                    )
                    ref = artifact.payload.pixel_values
                    assert isinstance(ref, LocalFileTensorPayloadRef)
                    self.assertEqual(ref.format, "npy")
                    ref_path = Path(ref.path)
                    self.assertTrue(ref_path.exists())
                    array = load_local_file_tensor_ref(ref)
                    self.assertEqual(tuple(array.shape), tuple(artifact.payload.payload_shape))
                    self.assertEqual(str(array.dtype), artifact.payload.payload_dtype)
                    self.assertIsNotNone(artifact.fetch_diagnostics_ms)
                    assert artifact.fetch_diagnostics_ms is not None
                    self.assertIn(
                        "payload_local_file_write_ms",
                        artifact.fetch_diagnostics_ms,
                    )
                finally:
                    try:
                        client.shutdown()
                    finally:
                        service.join(timeout=2.0)
                        service.terminate()
                self.assertIsNotNone(ref_path)
                assert ref_path is not None
                self.assertFalse(ref_path.exists())

    def test_service_process_worker_pool_can_return_raw_local_file_payload_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "service-raw-local-file.jpg"
            image_path.write_bytes(_make_jpeg_bytes())
            payload_dir = Path(tmpdir) / "payloads"
            service = SidecarServiceProcess(
                SidecarServiceConfig(
                    worker_pool_mode="process",
                    start_method="spawn",
                    manager=SidecarManagerConfig(
                        cache=MemoryCacheConfig(max_reusable_bytes=8 * 1024 * 1024),
                        workers=WorkerPoolConfig(worker_count=1, start_method="spawn"),
                    ),
                )
            )
            with mock.patch.dict(
                os.environ,
                {
                    "MM_SIDECAR_PAYLOAD_STORAGE": "local_file",
                    "MM_SIDECAR_PAYLOAD_FILE_FORMAT": "raw",
                    "MM_SIDECAR_PAYLOAD_DIR": str(payload_dir),
                },
            ):
                client = service.start()
                ref_path: Path | None = None
                try:
                    descriptor = _build_local_descriptor(image_path)
                    handles = client.prepare([descriptor])
                    artifact = _wait_for_ready_artifact(client, handles[0])
                    self.assertIsNotNone(artifact)
                    assert artifact is not None
                    self.assertEqual(artifact.descriptor.storage_kind, StorageKind.LOCAL_FILE)
                    self.assertEqual(artifact.payload.storage_kind, StorageKind.LOCAL_FILE)
                    ref = artifact.payload.pixel_values
                    self.assertIsInstance(ref, LocalFileTensorPayloadRef)
                    assert isinstance(ref, LocalFileTensorPayloadRef)
                    self.assertEqual(ref.format, "raw")
                    ref_path = Path(ref.path)
                    self.assertEqual(ref_path.suffix, ".raw")
                    self.assertTrue(ref_path.exists())
                    self.assertEqual(ref_path.stat().st_size, ref.nbytes)
                    array = load_local_file_tensor_ref(ref)
                    self.assertEqual(tuple(array.shape), tuple(artifact.payload.payload_shape))
                    self.assertEqual(str(array.dtype), artifact.payload.payload_dtype)
                finally:
                    try:
                        client.shutdown()
                    finally:
                        service.join(timeout=2.0)
                        service.terminate()
                self.assertIsNotNone(ref_path)
                assert ref_path is not None
                self.assertFalse(ref_path.exists())

    def test_service_process_worker_pool_can_return_torch_bf16_local_file_payload_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "service-torch-bf16-local-file.jpg"
            image_path.write_bytes(_make_jpeg_bytes())
            payload_dir = Path(tmpdir) / "payloads"
            service = SidecarServiceProcess(
                SidecarServiceConfig(
                    worker_pool_mode="process",
                    start_method="spawn",
                    manager=SidecarManagerConfig(
                        cache=MemoryCacheConfig(max_reusable_bytes=8 * 1024 * 1024),
                        workers=WorkerPoolConfig(worker_count=1, start_method="spawn"),
                    ),
                )
            )
            with mock.patch.dict(
                os.environ,
                {
                    "MM_SIDECAR_PAYLOAD_STORAGE": "local_file",
                    "MM_SIDECAR_PAYLOAD_FILE_FORMAT": "torch",
                    "MM_SIDECAR_PAYLOAD_DTYPE": "bf16",
                    "MM_SIDECAR_PAYLOAD_DIR": str(payload_dir),
                },
            ):
                client = service.start()
                ref_path: Path | None = None
                try:
                    descriptor = _build_local_descriptor(image_path)
                    handles = client.prepare([descriptor])
                    artifact = _wait_for_ready_artifact(client, handles[0])
                    self.assertIsNotNone(artifact)
                    assert artifact is not None
                    ref = artifact.payload.pixel_values
                    self.assertIsInstance(ref, LocalFileTensorPayloadRef)
                    assert isinstance(ref, LocalFileTensorPayloadRef)
                    self.assertEqual(ref.format, "torch")
                    self.assertEqual(ref.dtype, "bfloat16")
                    ref_path = Path(ref.path)
                    self.assertTrue(ref_path.exists())
                    tensor = load_local_file_tensor_ref(ref)
                    self.assertEqual(str(tensor.dtype), "torch.bfloat16")
                    self.assertEqual(tuple(tensor.shape), tuple(artifact.payload.payload_shape))
                    self.assertIsNotNone(artifact.fetch_diagnostics_ms)
                    assert artifact.fetch_diagnostics_ms is not None
                    self.assertIn("payload_torch_save_ms", artifact.fetch_diagnostics_ms)
                finally:
                    try:
                        client.shutdown()
                    finally:
                        service.join(timeout=2.0)
                        service.terminate()
                self.assertIsNotNone(ref_path)
                assert ref_path is not None
                self.assertFalse(ref_path.exists())

    def test_service_process_worker_pool_can_return_numpy_bf16_local_file_payload_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "service-numpy-bf16-local-file.jpg"
            image_path.write_bytes(_make_jpeg_bytes())
            payload_dir = Path(tmpdir) / "payloads"
            service = SidecarServiceProcess(
                SidecarServiceConfig(
                    worker_pool_mode="process",
                    start_method="spawn",
                    manager=SidecarManagerConfig(
                        cache=MemoryCacheConfig(max_reusable_bytes=8 * 1024 * 1024),
                        workers=WorkerPoolConfig(worker_count=1, start_method="spawn"),
                    ),
                )
            )
            with mock.patch.dict(
                os.environ,
                {
                    "MM_SIDECAR_PAYLOAD_STORAGE": "local_file",
                    "MM_SIDECAR_PAYLOAD_FILE_FORMAT": "numpy_bf16",
                    "MM_SIDECAR_PAYLOAD_DTYPE": "bf16",
                    "MM_SIDECAR_PAYLOAD_DIR": str(payload_dir),
                },
            ):
                client = service.start()
                ref_path: Path | None = None
                try:
                    descriptor = _build_local_descriptor(image_path)
                    handles = client.prepare([descriptor])
                    artifact = _wait_for_ready_artifact(client, handles[0])
                    self.assertIsNotNone(artifact)
                    assert artifact is not None
                    ref = artifact.payload.pixel_values
                    self.assertIsInstance(ref, LocalFileTensorPayloadRef)
                    assert isinstance(ref, LocalFileTensorPayloadRef)
                    self.assertEqual(ref.format, "numpy_bf16")
                    self.assertEqual(ref.dtype, "bfloat16")
                    ref_path = Path(ref.path)
                    self.assertTrue(ref_path.exists())
                    array = load_local_file_tensor_ref(ref)
                    self.assertEqual(str(array.dtype), "uint16")
                    self.assertEqual(tuple(array.shape), tuple(artifact.payload.payload_shape))
                    self.assertIsNotNone(artifact.fetch_diagnostics_ms)
                    assert artifact.fetch_diagnostics_ms is not None
                    self.assertIn(
                        "payload_numpy_bf16_save_ms",
                        artifact.fetch_diagnostics_ms,
                    )
                finally:
                    try:
                        client.shutdown()
                    finally:
                        service.join(timeout=2.0)
                        service.terminate()
                self.assertIsNotNone(ref_path)
                assert ref_path is not None
                self.assertFalse(ref_path.exists())

    def test_service_can_publish_and_fetch_request_local_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "service-fallback-local.jpg"
            image_path.write_bytes(_make_jpeg_bytes())
            service = SidecarServiceProcess(
                SidecarServiceConfig(
                    worker_pool_mode="process",
                    start_method="fork",
                    manager=SidecarManagerConfig(
                        cache=MemoryCacheConfig(max_reusable_bytes=8 * 1024 * 1024),
                        workers=WorkerPoolConfig(worker_count=1, start_method="fork"),
                    ),
                )
            )
            client = service.start()
            try:
                descriptor = _build_local_descriptor(image_path)
                handles = client.prepare([descriptor])
                claim_id = build_ranked_claimer_id(
                    request_id=descriptor.request_id,
                    producer_rank=0,
                )
                claims = client.try_fallback_claim(handles, claim_id)
                self.assertTrue(claims[0].granted)
                local_artifact = run_descriptor_locally(
                    descriptor,
                    epoch=claims[0].handle.epoch,
                )
                snapshot = client.publish_fallback_local_result(
                    claims[0].handle,
                    claim_id,
                    local_artifact.descriptor,
                    local_artifact.payload,
                    local_artifact.timings_ms,
                )
                self.assertEqual(snapshot.state, SidecarState.FALLBACK_LOCAL_DONE)
                artifact = client.fetch_ready(claims[0].handle)
                self.assertIsNotNone(artifact)
                assert artifact is not None
                self.assertEqual(
                    artifact.payload.image_grid_thw,
                    local_artifact.payload.image_grid_thw,
                )
                self.assertIsNotNone(artifact.fetch_diagnostics_ms)
                assert artifact.fetch_diagnostics_ms is not None
                self.assertIn("client_rpc_total", artifact.fetch_diagnostics_ms)
                self.assertIn("manager_local_payload", artifact.fetch_diagnostics_ms)
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
