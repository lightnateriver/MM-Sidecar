from __future__ import annotations

from dataclasses import replace
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
    InlineProcessorWorkerPool,
    SidecarFallbackCoordinator,
    SidecarManager,
    SidecarState,
    SourcePlanDecision,
)
from mm_sidecar.sidecar.coordinator import build_ranked_claimer_id
from mm_sidecar.sidecar.processor import WorkerResult, WorkerTask
from mm_sidecar.sidecar.processor import run_descriptor_locally
from mm_sidecar.sidecar.protocol import FallbackDescriptor


def _make_jpeg_bytes(size: tuple[int, int] = (288, 512)) -> bytes:
    image = Image.new("RGB", size, color=(11, 22, 33))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _make_processor_signature() -> ProcessorSignature:
    return ProcessorSignature.from_config(
        ProcessorConfig(
            model_name="qwen3.5-vl",
            revision="stage-d-test",
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


def _make_descriptor(path: Path, request_id: str, item_index: int) -> FallbackDescriptor:
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
                media_uuid=f"uuid-{request_id}-{item_index}",
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
        request_id=request_id,
        request_media_index=item_index,
        captured_image=CapturedImageRef(
            source_ref=normalized.source_ref,
            mime_type=normalized.mime_type,
            byte_size=normalized.byte_size,
            local_materialized_path=normalized.local_materialized_path,
        ),
        ingress_limits=_make_limits(),
        processor_signature_value=_make_processor_signature().value,
        item_identity=f"{normalized.source_ref.source_key}|{item_index}",
        orig_size_hw=normalized.orig_size_hw,
    )


class _ManualWorkerPool:
    def __init__(self) -> None:
        self.worker_count = 1
        self._results: list[WorkerResult] = []

    def submit(self, task: WorkerTask) -> None:
        self._results.append(
            WorkerResult(
                cache_key=task.cache_key,
                epoch=task.epoch,
                worker_id=task.assigned_worker_id,
                event_type="started",
                at_ms=1.0,
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


class _BatchCountingSidecarManager(SidecarManager):
    def __init__(self) -> None:
        super().__init__(worker_pool=InlineProcessorWorkerPool())
        self.fetch_ready_batch_calls = 0
        self.fetch_ready_calls = 0

    def fetch_ready_batch(self, handles):
        self.fetch_ready_batch_calls += 1
        return super().fetch_ready_batch(handles)

    def fetch_ready(self, handle):
        self.fetch_ready_calls += 1
        return super().fetch_ready(handle)


class SidecarCoordinatorTests(unittest.TestCase):
    def test_fetch_according_to_plan_uses_sidecar_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "ready.jpg"
            image_path.write_bytes(_make_jpeg_bytes())

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
            descriptor = _make_descriptor(image_path, "req-ready", 0)
            handles = manager.prepare([descriptor])
            snapshots = manager.wait_for_states(handles, {SidecarState.READY}, 500.0)
            self.assertEqual(snapshots[0].state, SidecarState.READY)

            coordinator = SidecarFallbackCoordinator(
                manager=manager,
                claimer_id="rank-0",
                producer_rank=0,
                near_ready_wait_ms=0.0,
            )
            batch = coordinator.fetch_according_to_plan(
                descriptors=[descriptor],
                handles=list(handles),
            )

            self.assertEqual(len(batch.sidecar_artifacts), 1)
            self.assertEqual(len(batch.fallback_descriptors), 0)
            self.assertEqual(batch.source_plan.entries[0].decision, SourcePlanDecision.USE_SIDECAR)
            manager.close()

    def test_fetch_according_to_plan_batches_ready_sidecar_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = [
                Path(tmpdir) / "ready-batch-0.jpg",
                Path(tmpdir) / "ready-batch-1.jpg",
            ]
            for image_path in image_paths:
                image_path.write_bytes(_make_jpeg_bytes())

            manager = _BatchCountingSidecarManager()
            descriptors = [
                _make_descriptor(image_path, "req-ready-batch", index)
                for index, image_path in enumerate(image_paths)
            ]
            handles = manager.prepare(descriptors)
            snapshots = manager.wait_for_states(handles, {SidecarState.READY}, 500.0)
            self.assertEqual([snapshot.state for snapshot in snapshots], [SidecarState.READY] * 2)

            coordinator = SidecarFallbackCoordinator(
                manager=manager,
                claimer_id="rank-0",
                producer_rank=0,
                near_ready_wait_ms=0.0,
            )
            batch = coordinator.fetch_according_to_plan(
                descriptors=descriptors,
                handles=list(handles),
            )

            self.assertEqual(manager.fetch_ready_batch_calls, 1)
            self.assertEqual(manager.fetch_ready_calls, 0)
            self.assertEqual(
                [artifact.handle.request_media_index for artifact in batch.sidecar_artifacts],
                [0, 1],
            )
            self.assertEqual(len(batch.fallback_descriptors), 0)
            manager.close()

    def test_build_source_plan_claims_fallback_when_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "claim.jpg"
            image_path.write_bytes(_make_jpeg_bytes())

            worker_pool = _ManualWorkerPool()
            manager = SidecarManager(worker_pool=worker_pool)
            descriptor = _make_descriptor(image_path, "req-claim", 0)
            handles = manager.prepare([descriptor])

            coordinator = SidecarFallbackCoordinator(
                manager=manager,
                claimer_id="rank-2",
                producer_rank=2,
                near_ready_wait_ms=0.0,
            )
            plan = coordinator.build_source_plan(
                descriptors=[descriptor],
                handles=list(handles),
            )

            self.assertEqual(plan.entries[0].decision, SourcePlanDecision.FALLBACK)
            self.assertEqual(plan.entries[0].producer_rank, 2)
            self.assertEqual(plan.entries[0].reason, "fallback_claim_granted")
            manager.close()

    def test_manager_unavailable_uses_fail_open_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "failopen.jpg"
            image_path.write_bytes(_make_jpeg_bytes())
            descriptor = _make_descriptor(image_path, "req-fail-open", 0)

            coordinator = SidecarFallbackCoordinator(
                manager=None,
                claimer_id="rank-1",
                producer_rank=1,
            )
            plan = coordinator.build_source_plan(
                descriptors=[descriptor],
                handles=None,
            )

            self.assertTrue(plan.used_fail_open)
            self.assertEqual(plan.entries[0].decision, SourcePlanDecision.FALLBACK)
            self.assertEqual(plan.entries[0].reason, "manager_unavailable_fail_open")

    def test_preview_source_plan_does_not_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "preview.jpg"
            image_path.write_bytes(_make_jpeg_bytes())

            worker_pool = _ManualWorkerPool()
            manager = SidecarManager(worker_pool=worker_pool)
            descriptor = _make_descriptor(image_path, "req-preview", 0)
            handles = manager.prepare([descriptor])

            coordinator = SidecarFallbackCoordinator(
                manager=manager,
                claimer_id="rank-preview",
                producer_rank=3,
                near_ready_wait_ms=0.0,
            )
            plan = coordinator.preview_source_plan(
                descriptors=[descriptor],
                handles=list(handles),
            )

            self.assertEqual(plan.entries[0].decision, SourcePlanDecision.FALLBACK)
            self.assertEqual(plan.entries[0].reason, "preview_requires_fallback")
            snapshots = manager.batch_get_status(handles)
            self.assertIn(
                snapshots[0].state,
                {SidecarState.QUEUED, SidecarState.SIDECAR_RUNNING},
            )
            manager.close()

    def test_no_claim_source_plan_can_wait_for_ready_without_claiming(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "preview-ready.jpg"
            image_path.write_bytes(_make_jpeg_bytes())

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
            descriptor = _make_descriptor(image_path, "req-preview-ready", 0)
            handles = manager.prepare([descriptor])

            coordinator = SidecarFallbackCoordinator(
                manager=manager,
                claimer_id="rank-preview-ready",
                producer_rank=3,
                near_ready_wait_ms=500.0,
                poll_interval_ms=1.0,
            )
            plan = coordinator.build_source_plan(
                descriptors=[descriptor],
                handles=list(handles),
                claim=False,
                wait_for_ready=True,
            )

            self.assertEqual(plan.entries[0].decision, SourcePlanDecision.USE_SIDECAR)
            self.assertIsNone(plan.entries[0].producer_rank)
            self.assertGreaterEqual(plan.near_ready_wait_ms, 0.0)
            snapshots = manager.batch_get_status(handles)
            self.assertIsNone(snapshots[0].claimed_by)
            manager.close()

    def test_claim_denied_raises_instead_of_repeating_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "claim_denied.jpg"
            image_path.write_bytes(_make_jpeg_bytes())

            worker_pool = _ManualWorkerPool()
            manager = SidecarManager(worker_pool=worker_pool)
            descriptor = _make_descriptor(image_path, "req-claim-denied", 0)
            handles = manager.prepare([descriptor])
            foreign_claim = manager.try_fallback_claim(handles, "foreign-rank")
            self.assertTrue(foreign_claim[0].granted)

            coordinator = SidecarFallbackCoordinator(
                manager=manager,
                claimer_id="rank-2",
                producer_rank=2,
                near_ready_wait_ms=0.0,
            )
            with self.assertRaisesRegex(RuntimeError, "fallback claim denied"):
                coordinator.build_source_plan(
                    descriptors=[descriptor],
                    handles=[foreign_claim[0].handle],
                )

            manager.close()

    def test_fetch_according_to_plan_uses_provided_plan_without_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "fixed_plan.jpg"
            image_path.write_bytes(_make_jpeg_bytes())

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
            descriptor = _make_descriptor(image_path, "req-fixed-plan", 0)
            handles = manager.prepare([descriptor])
            snapshots = manager.wait_for_states(handles, {SidecarState.READY}, 500.0)
            self.assertEqual(snapshots[0].state, SidecarState.READY)

            coordinator = SidecarFallbackCoordinator(
                manager=manager,
                claimer_id="rank-fixed",
                producer_rank=0,
                near_ready_wait_ms=0.0,
            )
            plan = coordinator.build_source_plan(
                descriptors=[descriptor],
                handles=list(handles),
            )
            batch = coordinator.fetch_according_to_plan(
                descriptors=[descriptor],
                handles=None,
                source_plan=plan,
            )

            self.assertEqual(batch.source_plan, plan)
            self.assertEqual(len(batch.sidecar_artifacts), 1)
            self.assertEqual(len(batch.fallback_descriptors), 0)
            manager.close()

    def test_consumer_observes_and_fetches_request_local_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "consumer-fallback.jpg"
            image_path.write_bytes(_make_jpeg_bytes())

            worker_pool = _ManualWorkerPool()
            manager = SidecarManager(worker_pool=worker_pool)
            descriptor = _make_descriptor(image_path, "req-consumer-fallback", 0)
            handles = manager.prepare([descriptor])

            coordinator = SidecarFallbackCoordinator(
                manager=manager,
                claimer_id=build_ranked_claimer_id(
                    request_id="req-consumer-fallback",
                    producer_rank=0,
                ),
                producer_rank=0,
                near_ready_wait_ms=0.0,
            )
            claimed_plan = coordinator.build_source_plan(
                descriptors=[descriptor],
                handles=list(handles),
            )
            self.assertEqual(
                claimed_plan.entries[0].decision,
                SourcePlanDecision.FALLBACK,
            )
            claimed_handle = claimed_plan.entries[0].handle
            self.assertIsNotNone(claimed_handle)
            assert claimed_handle is not None

            local_artifact = run_descriptor_locally(
                descriptor,
                epoch=claimed_handle.epoch,
            )
            snapshot = manager.publish_fallback_local_result(
                claimed_handle,
                build_ranked_claimer_id(
                    request_id="req-consumer-fallback",
                    producer_rank=0,
                ),
                local_artifact.descriptor,
                local_artifact.payload,
                local_artifact.timings_ms,
            )
            self.assertEqual(snapshot.state, SidecarState.FALLBACK_LOCAL_DONE)

            consumer = SidecarFallbackCoordinator(
                manager=manager,
                claimer_id=build_ranked_claimer_id(
                    request_id="req-consumer-fallback",
                    producer_rank=1,
                ),
                producer_rank=1,
                near_ready_wait_ms=0.0,
                fallback_wait_ms=20.0,
                observe_plan_wait_ms=20.0,
            )
            observed_plan = consumer.observe_source_plan(descriptors=[descriptor])
            self.assertEqual(
                observed_plan.entries[0].decision,
                SourcePlanDecision.FALLBACK,
            )
            self.assertEqual(observed_plan.entries[0].producer_rank, 0)

            batch = consumer.fetch_according_to_plan(
                descriptors=[descriptor],
                source_plan=observed_plan,
            )
            self.assertEqual(len(batch.sidecar_artifacts), 1)
            self.assertEqual(len(batch.fallback_descriptors), 0)
            self.assertEqual(
                batch.sidecar_artifacts[0].payload.image_grid_thw,
                local_artifact.payload.image_grid_thw,
            )
            manager.close()

    def test_stale_handle_from_previous_request_cannot_claim_or_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "stale_handle.jpg"
            image_path.write_bytes(_make_jpeg_bytes())

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
            descriptor = _make_descriptor(image_path, "req-old", 0)
            old_handles = manager.prepare([descriptor])
            snapshots = manager.wait_for_states(old_handles, {SidecarState.READY}, 500.0)
            self.assertEqual(snapshots[0].state, SidecarState.READY)

            new_descriptor = replace(descriptor, request_id="req-new")
            new_handles = manager.prepare([new_descriptor])
            new_snapshots = manager.batch_get_status(new_handles)
            self.assertEqual(new_snapshots[0].state, SidecarState.READY)

            stale_status = manager.batch_get_status(old_handles)
            self.assertEqual(stale_status[0].state, SidecarState.ABSENT)
            self.assertEqual(stale_status[0].error_message, "stale_handle")

            stale_claim = manager.try_fallback_claim(old_handles, "rank-stale")
            self.assertFalse(stale_claim[0].granted)
            self.assertEqual(stale_claim[0].error_message, "stale_handle")
            self.assertIsNone(manager.fetch_ready(old_handles[0]))
            self.assertIsNotNone(manager.fetch_ready(new_handles[0]))
            manager.close()


if __name__ == "__main__":
    unittest.main()
