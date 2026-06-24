from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from mm_sidecar.contracts import (
    ImageScheduleItem,
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
from mm_sidecar.sidecar.processor import WorkerResult, WorkerTask
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
        normalized_image=normalized,
        schedule_item=ImageScheduleItem(
            item_index=item_index,
            item_identity=f"{normalized.source_ref.source_key}|{item_index}",
            processor_signature=_make_processor_signature(),
            orig_size_hw=normalized.orig_size_hw,
            preprocessed_size_hw=normalized.orig_size_hw,
            image_grid_thw=(1, 36, 20),
            placeholder_token_count=180,
        ),
        ingress_limits=_make_limits(),
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

    def test_claim_denied_falls_back_fail_open_for_request(self) -> None:
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
            plan = coordinator.build_source_plan(
                descriptors=[descriptor],
                handles=[foreign_claim[0].handle],
            )

            self.assertEqual(plan.entries[0].decision, SourcePlanDecision.FALLBACK)
            self.assertEqual(plan.entries[0].producer_rank, 2)
            self.assertEqual(plan.entries[0].reason, "claim_denied_fail_open")
            manager.close()


if __name__ == "__main__":
    unittest.main()
