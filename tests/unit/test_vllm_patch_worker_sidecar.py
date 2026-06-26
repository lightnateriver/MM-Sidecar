from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from mm_sidecar.integrations.vllm_patch.carrier import (
    attach_sidecar_payload_to_params,
    build_request_sidecar_payload,
)
from mm_sidecar.integrations.vllm_patch.context import RequestCapture
from mm_sidecar.integrations.vllm_patch.normalization import (
    build_normalized_image_from_url,
)
from mm_sidecar.integrations.vllm_patch.sidecar_bridge import prepare_capture_for_sidecar
from mm_sidecar.integrations.vllm_patch.worker_sidecar import (
    TpWorkerRole,
    VitDpDirectEncodeResult,
    _all_tp_ranks_ready_for_direct_encode,
    _build_vit_dp_execution_plan_for_request,
    _manual_encode_and_gather_local_items,
    _resolve_vit_dp_local_indices,
    bind_request_mm_sidecar,
    build_worker_source_plan,
    prepare_scheduled_mm_inputs_before_encoder,
    install_gpu_model_runner_patch,
    reset_worker_sidecar_client_cache,
    try_replace_scheduled_mm_inputs_from_sidecar,
)
from mm_sidecar.sidecar import (
    InlineProcessorWorkerPool,
    SidecarManager,
    SidecarState,
    SourcePlanDecision,
)
from unittest import mock


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


class _FakeReqState:
    def __init__(self, params: _FakeParams) -> None:
        self.sampling_params = params


class _FakeNewReqData:
    def __init__(self, req_id: str) -> None:
        self.req_id = req_id


class _FakeSchedulerOutput:
    def __init__(self, req_id: str, encoder_input_ids: list[int] | None = None) -> None:
        self.scheduled_new_reqs = [_FakeNewReqData(req_id)]
        self.scheduled_encoder_inputs = {req_id: encoder_input_ids or [0]}


class _FakeFeature:
    modality = "image"
    data = "native-data"


class _ManualWorkerPool:
    def __init__(self) -> None:
        self.worker_count = 1
        self._results = []

    def submit(self, task) -> None:
        from mm_sidecar.sidecar.processor import WorkerResult

        self._results.append(
            WorkerResult(
                cache_key=task.cache_key,
                epoch=task.epoch,
                worker_id=task.assigned_worker_id,
                event_type="started",
                at_ms=1.0,
            )
        )

    def poll(self, max_items=None):
        if max_items is None or max_items >= len(self._results):
            results = list(self._results)
            self._results.clear()
            return results
        results = self._results[:max_items]
        del self._results[:max_items]
        return results

    def close(self) -> None:
        self._results.clear()


class VllmPatchWorkerSidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_worker_sidecar_client_cache()

    def test_bind_request_mm_sidecar_from_sampling_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-bind.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-bind",
                    request_scope_key="req-worker-bind",
                    item_index=0,
                )

            manager = SidecarManager(worker_pool=_ManualWorkerPool())
            capture = RequestCapture(
                request_id="req-worker-bind",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=manager,
            )
            capture.add_normalized_image(0, "uuid-worker-bind", normalized)
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            req_state = _FakeReqState(params)
            binding = bind_request_mm_sidecar(req_state)
            self.assertIsNotNone(binding)
            assert binding is not None
            self.assertEqual(binding.request_id, "req-worker-bind")
            self.assertEqual(binding.prepared_image_count, 1)
            self.assertGreater(binding.total_placeholder_token_count, 0)
            self.assertEqual(len(binding.decoded_plan.fallback_descriptors), 1)
            manager.close()

    def test_bind_request_mm_sidecar_from_synthetic_feature_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-bind-feature.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-bind-feature",
                    request_scope_key="req-worker-bind-feature",
                    item_index=0,
                )

            manager = SidecarManager(worker_pool=_ManualWorkerPool())
            capture = RequestCapture(
                request_id="req-worker-bind-feature",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=manager,
            )
            capture.add_normalized_image(0, "uuid-worker-bind-feature", normalized)
            params = _FakeParams()
            payload = prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            self.assertIsNotNone(payload)
            assert payload is not None

            feature_data = type(
                "SyntheticFeatureData",
                (),
                {
                    "_mm_sidecar_synthetic_placeholder": True,
                    "_mm_sidecar_request_payload": build_request_sidecar_payload(capture),
                },
            )()
            req_state = _FakeReqState(params)
            req_state.sampling_params = type("ParamsWithoutExtraArgs", (), {})()
            req_state.mm_features = [type("Feature", (), {"data": feature_data})()]

            binding = bind_request_mm_sidecar(req_state)
            self.assertIsNotNone(binding)
            assert binding is not None
            self.assertEqual(binding.request_id, "req-worker-bind-feature")
            self.assertEqual(binding.prepared_image_count, 1)
            manager.close()

    def test_bind_request_mm_sidecar_from_nested_feature_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-bind-nested-feature.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-bind-nested-feature",
                    request_scope_key="req-worker-bind-nested-feature",
                    item_index=0,
                )

            manager = SidecarManager(worker_pool=_ManualWorkerPool())
            capture = RequestCapture(
                request_id="req-worker-bind-nested-feature",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=manager,
            )
            capture.add_normalized_image(0, "uuid-worker-bind-nested-feature", normalized)
            params = _FakeParams()
            payload = prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            self.assertIsNotNone(payload)
            assert payload is not None

            nested_elem = type(
                "NestedFeatureElem",
                (),
                {
                    "_mm_sidecar_request_payload": build_request_sidecar_payload(capture),
                },
            )()
            req_state = _FakeReqState(params)
            req_state.sampling_params = type("ParamsWithoutExtraArgs", (), {})()
            req_state.mm_features = [
                type("Feature", (), {"data": {"pixel_values": nested_elem}})()
            ]

            binding = bind_request_mm_sidecar(req_state)
            self.assertIsNotNone(binding)
            assert binding is not None
            self.assertEqual(binding.request_id, "req-worker-bind-nested-feature")
            self.assertEqual(binding.prepared_image_count, 1)
            manager.close()

    def test_bind_request_mm_sidecar_reconstructs_from_manager_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-bind-lookup.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-bind-lookup",
                    request_scope_key="req-worker-bind-lookup",
                    item_index=0,
                )

            manager = SidecarManager(worker_pool=_ManualWorkerPool())
            capture = RequestCapture(
                request_id="req-worker-bind-lookup",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=manager,
            )
            capture.add_normalized_image(0, "uuid-worker-bind-lookup", normalized)
            params = _FakeParams()
            payload = prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            assert payload is not None

            cache_key = str(payload["handles"][0]["cache_key"])
            req_state = _FakeReqState(params)
            req_state.sampling_params = type("ParamsWithoutExtraArgs", (), {})()
            req_state.mm_features = [
                type(
                    "Feature",
                    (),
                    {"identifier": cache_key, "modality": "image", "data": None},
                )()
            ]

            with mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.get_worker_sidecar_client",
                return_value=manager,
            ):
                binding = bind_request_mm_sidecar(req_state)

            self.assertIsNotNone(binding)
            assert binding is not None
            self.assertEqual(binding.request_id, "req-worker-bind-lookup")
            self.assertEqual(binding.prepared_image_count, 1)
            self.assertEqual(len(binding.decoded_plan.fallback_descriptors), 1)
            self.assertEqual(len(binding.decoded_plan.handles), 1)
            manager.close()

    def test_build_worker_source_plan_fail_open_without_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-plan.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-plan",
                    request_scope_key="req-worker-plan",
                    item_index=0,
                )

            capture = RequestCapture(
                request_id="req-worker-plan",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=None,
            )
            capture.add_normalized_image(0, "uuid-worker-plan", normalized)
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            req_state = _FakeReqState(params)
            plan = build_worker_source_plan(req_state)
            self.assertIsNotNone(plan)
            assert plan is not None
            self.assertTrue(plan.used_fail_open)
            self.assertEqual(len(plan.entries), 1)
            self.assertEqual(plan.entries[0].reason, "manager_unavailable_fail_open")

    def test_install_gpu_model_runner_patch_binds_new_requests_without_changing_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-patch.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-patch",
                    request_scope_key="req-worker-patch",
                    item_index=0,
                )

            capture = RequestCapture(
                request_id="req-worker-patch",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=None,
            )
            capture.add_normalized_image(0, "uuid-worker-patch", normalized)
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            class FakeGPUModelRunner:
                def __init__(self) -> None:
                    self.requests = {"req-worker-patch": _FakeReqState(params)}

                def _update_states(self, scheduler_output):
                    return "update-result"

                def _batch_mm_inputs_from_scheduler(self, scheduler_output):
                    return "batch-result"

            self.assertTrue(install_gpu_model_runner_patch(FakeGPUModelRunner))
            self.assertFalse(install_gpu_model_runner_patch(FakeGPUModelRunner))

            runner = FakeGPUModelRunner()
            scheduler_output = _FakeSchedulerOutput("req-worker-patch")
            self.assertEqual(runner._update_states(scheduler_output), "update-result")
            req_state = runner.requests["req-worker-patch"]
            self.assertIsNotNone(getattr(req_state, "mm_sidecar_binding", None))
            self.assertEqual(runner.mm_sidecar_last_bound_count, 1)

            self.assertEqual(
                runner._batch_mm_inputs_from_scheduler(scheduler_output),
                "batch-result",
            )
            self.assertIsNotNone(
                getattr(req_state, "mm_sidecar_source_plan_preview", None)
            )
            self.assertEqual(runner.mm_sidecar_last_source_plan_count, 1)

    def test_try_replace_keeps_native_data_when_manager_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-native.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-native",
                    request_scope_key="req-worker-native",
                    item_index=0,
                )

            capture = RequestCapture(
                request_id="req-worker-native",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=None,
            )
            capture.add_normalized_image(0, "uuid-worker-native", normalized)
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            req_state = _FakeReqState(params)
            req_state.mm_features = [_FakeFeature()]
            runner = type("Runner", (), {})()
            runner.requests = {"req-worker-native": req_state}
            scheduler_output = _FakeSchedulerOutput("req-worker-native")

            replaced = try_replace_scheduled_mm_inputs_from_sidecar(
                runner,
                scheduler_output,
            )

            self.assertEqual(replaced, 0)
            self.assertEqual(req_state.mm_features[0].data, "native-data")

    def test_try_replace_runs_local_fallback_when_feature_data_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-fallback.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-fallback",
                    request_scope_key="req-worker-fallback",
                    item_index=0,
                )

            capture = RequestCapture(
                request_id="req-worker-fallback",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=None,
            )
            capture.add_normalized_image(0, "uuid-worker-fallback", normalized)
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            req_state = _FakeReqState(params)
            missing_feature = _FakeFeature()
            missing_feature.data = None
            req_state.mm_features = [missing_feature]
            runner = type("Runner", (), {})()
            runner.requests = {"req-worker-fallback": req_state}
            scheduler_output = _FakeSchedulerOutput("req-worker-fallback")

            with mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.replace_feature_data_from_sidecar_artifacts",
                side_effect=lambda state, artifacts: (
                    setattr(state.mm_features[0], "data", {"pixel_values": "fallback"})
                    or len(artifacts)
                ),
            ):
                replaced = try_replace_scheduled_mm_inputs_from_sidecar(
                    runner,
                    scheduler_output,
                )

            self.assertEqual(replaced, 1)
            self.assertEqual(req_state.mm_features[0].data, {"pixel_values": "fallback"})
            self.assertTrue(req_state.mm_sidecar_source_plan.used_fail_open)
            self.assertEqual(len(req_state.mm_sidecar_fallback_descriptors), 1)

    def test_try_replace_runs_local_fallback_when_feature_data_is_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-synthetic.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-synthetic",
                    request_scope_key="req-worker-synthetic",
                    item_index=0,
                )

            capture = RequestCapture(
                request_id="req-worker-synthetic",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=None,
            )
            capture.add_normalized_image(0, "uuid-worker-synthetic", normalized)
            params = _FakeParams()
            payload = prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)
            assert payload is not None

            req_state = _FakeReqState(params)
            synthetic_feature = _FakeFeature()
            synthetic_feature.data = type(
                "SyntheticFeatureData",
                (),
                {"_mm_sidecar_synthetic_placeholder": True},
            )()
            req_state.mm_features = [synthetic_feature]
            runner = type("Runner", (), {})()
            runner.requests = {"req-worker-synthetic": req_state}
            scheduler_output = _FakeSchedulerOutput("req-worker-synthetic")

            with mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.replace_feature_data_from_sidecar_artifacts",
                side_effect=lambda state, artifacts: (
                    setattr(state.mm_features[0], "data", {"pixel_values": "fallback"})
                    or len(artifacts)
                ),
            ):
                replaced = try_replace_scheduled_mm_inputs_from_sidecar(
                    runner,
                    scheduler_output,
                )

            self.assertEqual(replaced, 1)
            self.assertEqual(req_state.mm_features[0].data, {"pixel_values": "fallback"})

    def test_try_replace_treats_empty_tensor_feature_data_as_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-empty-tensor.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-empty-tensor",
                    request_scope_key="req-worker-empty-tensor",
                    item_index=0,
                )

            capture = RequestCapture(
                request_id="req-worker-empty-tensor",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=None,
            )
            capture.add_normalized_image(0, "uuid-worker-empty-tensor", normalized)
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            import torch

            req_state = _FakeReqState(params)
            req_state.mm_features = [
                type(
                    "Feature",
                    (),
                    {"data": {"pixel_values": torch.empty((0, 1536))}},
                )()
            ]
            runner = type("Runner", (), {})()
            runner.requests = {"req-worker-empty-tensor": req_state}
            scheduler_output = _FakeSchedulerOutput("req-worker-empty-tensor")

            with mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.replace_feature_data_from_sidecar_artifacts",
                side_effect=lambda state, artifacts: (
                    setattr(state.mm_features[0], "data", {"pixel_values": "replaced"})
                    or len(artifacts)
                ),
            ):
                replaced = try_replace_scheduled_mm_inputs_from_sidecar(
                    runner,
                    scheduler_output,
                )

            self.assertEqual(replaced, 1)
            self.assertEqual(req_state.mm_features[0].data, {"pixel_values": "replaced"})

    def test_try_replace_uses_ready_sidecar_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-ready.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-ready",
                    request_scope_key="req-worker-ready",
                    item_index=0,
                )

            manager = SidecarManager(worker_pool=_ManualWorkerPool())
            capture = RequestCapture(
                request_id="req-worker-ready",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=manager,
            )
            capture.add_normalized_image(0, "uuid-worker-ready", normalized)
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            req_state = _FakeReqState(params)
            synthetic_feature = _FakeFeature()
            synthetic_feature.data = type(
                "SyntheticFeatureData",
                (),
                {"_mm_sidecar_synthetic_placeholder": True},
            )()
            req_state.mm_features = [synthetic_feature]
            runner = type("Runner", (), {})()
            runner.requests = {"req-worker-ready": req_state}
            scheduler_output = _FakeSchedulerOutput("req-worker-ready")

            with mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.get_worker_sidecar_client",
                return_value=manager,
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.replace_feature_data_from_sidecar_artifacts",
                side_effect=lambda state, artifacts: (
                    setattr(state.mm_features[0], "data", {"pixel_values": "sidecar"})
                    or len(artifacts)
                ),
            ):
                replaced = try_replace_scheduled_mm_inputs_from_sidecar(
                    runner,
                    scheduler_output,
                )

            self.assertEqual(replaced, 1)
            self.assertNotEqual(req_state.mm_features[0].data, "native-data")
            self.assertIn("pixel_values", req_state.mm_features[0].data)
            self.assertEqual(runner.mm_sidecar_last_replaced_feature_count, 1)
            self.assertIsNotNone(getattr(runner, "mm_sidecar_last_fetch_profile_ms", None))
            profile = runner.mm_sidecar_last_fetch_profile_ms
            self.assertIn("source_plan_ms", profile)
            self.assertIn("fetch_ms", profile)
            self.assertIn("replace_ms", profile)
            manager.close()

    def test_try_replace_producer_rank_publishes_request_local_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-producer-fallback.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-producer-fallback",
                    request_scope_key="req-worker-producer-fallback",
                    item_index=0,
                )

            manager = SidecarManager(worker_pool=_ManualWorkerPool())
            capture = RequestCapture(
                request_id="req-worker-producer-fallback",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=manager,
            )
            capture.add_normalized_image(
                0,
                "uuid-worker-producer-fallback",
                normalized,
            )
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            req_state = _FakeReqState(params)
            synthetic_feature = _FakeFeature()
            synthetic_feature.data = type(
                "SyntheticFeatureData",
                (),
                {"_mm_sidecar_synthetic_placeholder": True},
            )()
            req_state.mm_features = [synthetic_feature]
            runner = type("Runner", (), {})()
            runner.requests = {"req-worker-producer-fallback": req_state}
            scheduler_output = _FakeSchedulerOutput("req-worker-producer-fallback")

            with mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.get_worker_sidecar_client",
                return_value=manager,
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar._resolve_tp_worker_role",
                return_value=TpWorkerRole(
                    local_rank=0,
                    world_size=2,
                    coordinator_rank=0,
                    is_coordinator=True,
                ),
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.replace_feature_data_from_sidecar_artifacts",
                side_effect=lambda state, artifacts: (
                    setattr(state.mm_features[0], "data", {"pixel_values": "fallback"})
                    or len(artifacts)
                ),
            ):
                replaced = try_replace_scheduled_mm_inputs_from_sidecar(
                    runner,
                    scheduler_output,
                )

            self.assertEqual(replaced, 1)
            binding = getattr(req_state, "mm_sidecar_binding")
            handle = binding.decoded_plan.handles[0]
            snapshots = manager.lookup_by_cache_keys([handle.cache_key])
            self.assertEqual(snapshots[0].state, SidecarState.FALLBACK_LOCAL_DONE)
            self.assertIsNotNone(snapshots[0].claimed_by)
            manager.close()

    def test_try_replace_consumer_rank_fetches_request_local_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-consumer-fallback.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-consumer-fallback",
                    request_scope_key="req-worker-consumer-fallback",
                    item_index=0,
                )

            manager = SidecarManager(worker_pool=_ManualWorkerPool())
            capture = RequestCapture(
                request_id="req-worker-consumer-fallback",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=manager,
            )
            capture.add_normalized_image(
                0,
                "uuid-worker-consumer-fallback",
                normalized,
            )
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            req_state = _FakeReqState(params)
            synthetic_feature = _FakeFeature()
            synthetic_feature.data = type(
                "SyntheticFeatureData",
                (),
                {"_mm_sidecar_synthetic_placeholder": True},
            )()
            req_state.mm_features = [synthetic_feature]
            runner = type("Runner", (), {})()
            runner.requests = {"req-worker-consumer-fallback": req_state}
            scheduler_output = _FakeSchedulerOutput("req-worker-consumer-fallback")

            producer_req_state = _FakeReqState(params)
            producer_feature = _FakeFeature()
            producer_feature.data = type(
                "SyntheticFeatureData",
                (),
                {"_mm_sidecar_synthetic_placeholder": True},
            )()
            producer_req_state.mm_features = [producer_feature]
            producer_runner = type("Runner", (), {})()
            producer_runner.requests = {"req-worker-consumer-fallback": producer_req_state}

            with mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.get_worker_sidecar_client",
                return_value=manager,
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.replace_feature_data_from_sidecar_artifacts",
                side_effect=lambda state, artifacts: (
                    setattr(state.mm_features[0], "data", {"pixel_values": "sidecar"})
                    or len(artifacts)
                ),
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar._resolve_tp_worker_role",
                return_value=TpWorkerRole(
                    local_rank=0,
                    world_size=2,
                    coordinator_rank=0,
                    is_coordinator=True,
                ),
            ):
                self.assertEqual(
                    try_replace_scheduled_mm_inputs_from_sidecar(
                        producer_runner,
                        scheduler_output,
                    ),
                    1,
                )

            with mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.get_worker_sidecar_client",
                return_value=manager,
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.replace_feature_data_from_sidecar_artifacts",
                side_effect=lambda state, artifacts: (
                    setattr(state.mm_features[0], "data", {"pixel_values": "peer"})
                    or len(artifacts)
                ),
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar._resolve_tp_worker_role",
                return_value=TpWorkerRole(
                    local_rank=1,
                    world_size=2,
                    coordinator_rank=0,
                    is_coordinator=False,
                ),
            ):
                replaced = try_replace_scheduled_mm_inputs_from_sidecar(
                    runner,
                    scheduler_output,
                )

            self.assertEqual(replaced, 1)
            self.assertEqual(req_state.mm_features[0].data, {"pixel_values": "peer"})
            self.assertIsNotNone(getattr(runner, "mm_sidecar_last_tp_role", None))
            manager.close()

    def test_try_replace_consumer_rank_degrades_to_local_fallback_when_peer_artifact_wait_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "worker-consumer-timeout.jpg"
            _make_image().save(image_path, format="JPEG")
            with Image.open(image_path) as image:
                normalized = build_normalized_image_from_url(
                    image_url=f"file://{image_path}",
                    image=image,
                    media_uuid="uuid-worker-consumer-timeout",
                    request_scope_key="req-worker-consumer-timeout",
                    item_index=0,
                )

            manager = SidecarManager(worker_pool=_ManualWorkerPool())
            capture = RequestCapture(
                request_id="req-worker-consumer-timeout",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=manager,
            )
            capture.add_normalized_image(
                0,
                "uuid-worker-consumer-timeout",
                normalized,
            )
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            req_state = _FakeReqState(params)
            synthetic_feature = _FakeFeature()
            synthetic_feature.data = type(
                "SyntheticFeatureData",
                (),
                {"_mm_sidecar_synthetic_placeholder": True},
            )()
            req_state.mm_features = [synthetic_feature]
            binding = bind_request_mm_sidecar(req_state)
            assert binding is not None
            handle = binding.decoded_plan.handles[0]
            runner = type("Runner", (), {})()
            runner.requests = {"req-worker-consumer-timeout": req_state}
            scheduler_output = _FakeSchedulerOutput("req-worker-consumer-timeout")

            with mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.get_worker_sidecar_client",
                return_value=manager,
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar._resolve_tp_worker_role",
                return_value=TpWorkerRole(
                    local_rank=1,
                    world_size=2,
                    coordinator_rank=0,
                    is_coordinator=False,
                ),
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.SidecarFallbackCoordinator.observe_source_plan",
                return_value=SimpleNamespace(
                    request_id="req-worker-consumer-timeout",
                    entries=(
                        SimpleNamespace(
                            request_media_index=0,
                            decision=SourcePlanDecision.FALLBACK,
                            producer_rank=0,
                            handle=handle,
                            state=SidecarState.FALLBACK_CLAIMED,
                            reason="fallback_observed_from_manager",
                        ),
                    ),
                    near_ready_wait_ms=0.0,
                    used_fail_open=False,
                ),
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.SidecarFallbackCoordinator.fetch_according_to_plan",
                side_effect=RuntimeError(
                    "remote fallback artifact unavailable for media index 0: "
                    "state=FALLBACK_CLAIMED"
                ),
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.replace_feature_data_from_sidecar_artifacts",
                side_effect=lambda state, artifacts: (
                    setattr(state.mm_features[0], "data", {"pixel_values": "local-degrade"})
                    or len(artifacts)
                ),
            ):
                replaced = try_replace_scheduled_mm_inputs_from_sidecar(
                    runner,
                    scheduler_output,
                )

            self.assertEqual(replaced, 1)
            self.assertEqual(
                req_state.mm_features[0].data,
                {"pixel_values": "local-degrade"},
            )
            manager.close()

    def test_try_replace_vit_dp_default_replaces_all_images_for_stock_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path0 = Path(tmpdir) / "worker-vit-dp-0.jpg"
            image_path1 = Path(tmpdir) / "worker-vit-dp-1.jpg"
            _make_image().save(image_path0, format="JPEG")
            _make_image().save(image_path1, format="JPEG")
            with Image.open(image_path0) as image0, Image.open(image_path1) as image1:
                normalized0 = build_normalized_image_from_url(
                    image_url=f"file://{image_path0}",
                    image=image0,
                    media_uuid="uuid-worker-vit-dp-0",
                    request_scope_key="req-worker-vit-dp",
                    item_index=0,
                )
                normalized1 = build_normalized_image_from_url(
                    image_url=f"file://{image_path1}",
                    image=image1,
                    media_uuid="uuid-worker-vit-dp-1",
                    request_scope_key="req-worker-vit-dp",
                    item_index=1,
                )

            capture = RequestCapture(
                request_id="req-worker-vit-dp",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=None,
            )
            capture.add_normalized_image(0, "uuid-worker-vit-dp-0", normalized0)
            capture.add_normalized_image(1, "uuid-worker-vit-dp-1", normalized1)
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            req_state = _FakeReqState(params)
            synthetic_feature0 = _FakeFeature()
            synthetic_feature0.data = type(
                "SyntheticFeatureData",
                (),
                {"_mm_sidecar_synthetic_placeholder": True},
            )()
            synthetic_feature1 = _FakeFeature()
            synthetic_feature1.data = type(
                "SyntheticFeatureData",
                (),
                {"_mm_sidecar_synthetic_placeholder": True},
            )()
            req_state.mm_features = [synthetic_feature0, synthetic_feature1]
            runner = type("Runner", (), {})()
            runner.requests = {"req-worker-vit-dp": req_state}
            runner.model = SimpleNamespace(use_data_parallel=True)
            scheduler_output = _FakeSchedulerOutput(
                "req-worker-vit-dp",
                encoder_input_ids=[0, 1],
            )

            def fake_replace_feature_data(state, artifacts):
                for artifact in artifacts:
                    state.mm_features[int(artifact.handle.request_media_index)].data = {
                        "kind": "local-real",
                        "pixel_values": SimpleNamespace(
                            data=SimpleNamespace(shape=(8, 8))
                        ),
                    }
                return len(artifacts)

            with mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar._resolve_tp_worker_role",
                return_value=TpWorkerRole(
                    local_rank=0,
                    world_size=2,
                    coordinator_rank=0,
                    is_coordinator=True,
                ),
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.replace_feature_data_from_sidecar_artifacts",
                side_effect=fake_replace_feature_data,
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.planned_item_to_vit_dp_placeholder_qwen_mm_kwargs_item",
                side_effect=lambda planned_item, processor_signature: {
                    "kind": "remote-placeholder",
                    "pixel_values": SimpleNamespace(
                        data=SimpleNamespace(shape=(4, 4))
                    ),
                },
            ):
                replaced = try_replace_scheduled_mm_inputs_from_sidecar(
                    runner,
                    scheduler_output,
                )

            self.assertEqual(replaced, 2)
            self.assertEqual(req_state.mm_features[0].data["kind"], "local-real")
            self.assertEqual(req_state.mm_features[1].data["kind"], "local-real")
            self.assertFalse(getattr(req_state, "mm_sidecar_vit_dp_prepared", False))

    def test_try_replace_vit_dp_direct_mode_prepares_without_stock_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path0 = Path(tmpdir) / "worker-vit-dp-direct-0.jpg"
            image_path1 = Path(tmpdir) / "worker-vit-dp-direct-1.jpg"
            _make_image().save(image_path0, format="JPEG")
            _make_image().save(image_path1, format="JPEG")
            with Image.open(image_path0) as image0, Image.open(image_path1) as image1:
                normalized0 = build_normalized_image_from_url(
                    image_url=f"file://{image_path0}",
                    image=image0,
                    media_uuid="uuid-worker-vit-dp-direct-0",
                    request_scope_key="req-worker-vit-dp-direct",
                    item_index=0,
                )
                normalized1 = build_normalized_image_from_url(
                    image_url=f"file://{image_path1}",
                    image=image1,
                    media_uuid="uuid-worker-vit-dp-direct-1",
                    request_scope_key="req-worker-vit-dp-direct",
                    item_index=1,
                )

            capture = RequestCapture(
                request_id="req-worker-vit-dp-direct",
                method="POST",
                path="/v1/chat/completions",
                sidecar_manager=None,
            )
            capture.add_normalized_image(0, "uuid-worker-vit-dp-direct-0", normalized0)
            capture.add_normalized_image(1, "uuid-worker-vit-dp-direct-1", normalized1)
            params = _FakeParams()
            prepare_capture_for_sidecar(capture, _FakeRenderer(), params)
            attach_sidecar_payload_to_params(params, capture)

            req_state = _FakeReqState(params)
            synthetic_feature0 = _FakeFeature()
            synthetic_feature0.data = type(
                "SyntheticFeatureData",
                (),
                {"_mm_sidecar_synthetic_placeholder": True},
            )()
            synthetic_feature1 = _FakeFeature()
            synthetic_feature1.data = type(
                "SyntheticFeatureData",
                (),
                {"_mm_sidecar_synthetic_placeholder": True},
            )()
            req_state.mm_features = [synthetic_feature0, synthetic_feature1]
            runner = type("Runner", (), {})()
            runner.requests = {"req-worker-vit-dp-direct": req_state}
            runner.model = SimpleNamespace(use_data_parallel=True)
            scheduler_output = _FakeSchedulerOutput(
                "req-worker-vit-dp-direct",
                encoder_input_ids=[0, 1],
            )

            with mock.patch.dict(
                os.environ,
                {"MM_SIDECAR_ENABLE_VIT_DP_DIRECT_ENCODE": "1"},
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar._resolve_tp_worker_role",
                return_value=TpWorkerRole(
                    local_rank=0,
                    world_size=2,
                    coordinator_rank=0,
                    is_coordinator=True,
                ),
            ), mock.patch(
                "mm_sidecar.integrations.vllm_patch.worker_sidecar.replace_feature_data_from_sidecar_artifacts",
                side_effect=AssertionError("stock replacement should not run"),
            ):
                replaced = try_replace_scheduled_mm_inputs_from_sidecar(
                    runner,
                    scheduler_output,
                )

            self.assertEqual(replaced, 0)
            self.assertTrue(getattr(req_state, "mm_sidecar_vit_dp_prepared", False))

    def test_prepare_scheduled_mm_inputs_before_encoder_runs_once(self) -> None:
        runner = type("Runner", (), {})()
        runner.requests = {}
        scheduler_output = _FakeSchedulerOutput("req-prep-once")
        calls: list[str] = []

        def fake_bind(model_runner, sched_output):
            calls.append("bind")
            return 1

        def fake_preview(model_runner, sched_output):
            calls.append("preview")
            return 1

        def fake_replace(model_runner, sched_output):
            calls.append("replace")
            return 1

        with mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar.bind_scheduled_requests",
            side_effect=fake_bind,
        ), mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar.build_scheduled_source_plan_previews",
            side_effect=fake_preview,
        ), mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar.try_replace_scheduled_mm_inputs_from_sidecar",
            side_effect=fake_replace,
        ):
            self.assertEqual(
                prepare_scheduled_mm_inputs_before_encoder(runner, scheduler_output),
                1,
            )
            self.assertEqual(
                prepare_scheduled_mm_inputs_before_encoder(runner, scheduler_output),
                0,
            )

        self.assertEqual(calls, ["bind", "preview", "replace"])
        self.assertEqual(runner.mm_sidecar_last_prepare_count, 1)

    def test_resolve_vit_dp_local_indices_supports_tp4(self) -> None:
        binding = SimpleNamespace(decoded_plan=SimpleNamespace(planned_items=()))
        req_state = SimpleNamespace(mm_features=[])
        image_features = [
            SimpleNamespace(data={"image_grid_thw": SimpleNamespace(data=[1, 10, 100])}),
            SimpleNamespace(data={"image_grid_thw": SimpleNamespace(data=[1, 10, 10])}),
            SimpleNamespace(data={"image_grid_thw": SimpleNamespace(data=[1, 10, 20])}),
            SimpleNamespace(data={"image_grid_thw": SimpleNamespace(data=[1, 5, 10])}),
            SimpleNamespace(data={"image_grid_thw": SimpleNamespace(data=[1, 8, 8])}),
        ]
        image_feature_ids = [0, 1, 2, 3, 4]

        local_indices, order, counts, loads = _resolve_vit_dp_local_indices(
            binding,
            req_state,
            image_features,
            image_feature_ids,
            role=TpWorkerRole(
                local_rank=2,
                world_size=4,
                coordinator_rank=0,
                is_coordinator=False,
            ),
        )

        self.assertEqual(len(order), 5)
        self.assertEqual(sum(counts), 5)
        self.assertEqual(len(counts), 4)
        self.assertEqual(len(loads), 4)
        self.assertEqual(tuple(order[sum(counts[:2]) : sum(counts[:3])]), local_indices)

    def test_build_vit_dp_execution_plan_resolves_local_indices_before_lookup(self) -> None:
        descriptors = tuple(
            SimpleNamespace(request_media_index=index) for index in (10, 20, 30)
        )
        handles = tuple(
            SimpleNamespace(request_media_index=index) for index in (10, 20, 30)
        )
        binding = SimpleNamespace(
            enabled=True,
            request_id="req-vit-plan",
            decoded_plan=SimpleNamespace(
                fallback_descriptors=descriptors,
                handles=handles,
                planned_items=(),
            ),
        )
        req_state = SimpleNamespace()
        role = TpWorkerRole(
            local_rank=1,
            world_size=2,
            coordinator_rank=0,
            is_coordinator=False,
        )

        with mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar.get_request_mm_sidecar_binding",
            return_value=binding,
        ), mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar._scheduled_image_features_for_request",
            return_value=(
                (
                    SimpleNamespace(data={"image_grid_thw": SimpleNamespace(data=[1, 8, 8])}),
                    SimpleNamespace(data={"image_grid_thw": SimpleNamespace(data=[1, 8, 8])}),
                    SimpleNamespace(data={"image_grid_thw": SimpleNamespace(data=[1, 8, 8])}),
                ),
                (10, 20, 30),
            ),
        ), mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar._resolve_vit_dp_local_indices",
            return_value=((2, 0), (2, 1, 0), (1, 2), (64, 128)),
        ), mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar.get_worker_sidecar_client",
            return_value=None,
        ), mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar.SidecarFallbackCoordinator.preview_source_plan",
            return_value="source-plan",
        ):
            plan = _build_vit_dp_execution_plan_for_request(
                model_runner=SimpleNamespace(),
                req_id="req-vit-plan",
                req_state=req_state,
                image_input_ids=[0, 1, 2],
                role=role,
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.local_indices, (2, 0))
        self.assertEqual(
            tuple(descriptor.request_media_index for descriptor in plan.descriptors),
            (10, 30),
        )
        self.assertEqual(
            tuple(handle.request_media_index for handle in plan.handles),
            (10, 30),
        )

    def test_build_vit_dp_execution_plan_allows_empty_local_shard(self) -> None:
        descriptors = tuple(
            SimpleNamespace(request_media_index=index) for index in (10,)
        )
        handles = tuple(
            SimpleNamespace(request_media_index=index) for index in (10,)
        )
        binding = SimpleNamespace(
            enabled=True,
            request_id="req-vit-empty-local",
            decoded_plan=SimpleNamespace(
                fallback_descriptors=descriptors,
                handles=handles,
                planned_items=(),
            ),
        )

        with mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar.get_request_mm_sidecar_binding",
            return_value=binding,
        ), mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar._scheduled_image_features_for_request",
            return_value=(
                (
                    SimpleNamespace(data={"image_grid_thw": SimpleNamespace(data=[1, 8, 8])}),
                ),
                (10,),
            ),
        ), mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar._resolve_vit_dp_local_indices",
            return_value=((), (0,), (1, 0), (64, 0)),
        ), mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar.get_worker_sidecar_client",
            side_effect=AssertionError("empty shard should not build source plan"),
        ):
            plan = _build_vit_dp_execution_plan_for_request(
                model_runner=SimpleNamespace(),
                req_id="req-vit-empty-local",
                req_state=SimpleNamespace(),
                image_input_ids=[0],
                role=TpWorkerRole(
                    local_rank=1,
                    world_size=2,
                    coordinator_rank=0,
                    is_coordinator=False,
                ),
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.local_indices, ())
        self.assertEqual(plan.descriptors, ())
        self.assertEqual(plan.handles, ())
        self.assertIsNone(plan.source_plan)

    def test_all_tp_ranks_ready_for_direct_encode_requires_every_rank(self) -> None:
        class _FakeTensor:
            def __init__(self, value):
                self._value = value

            def item(self):
                return self._value

        class _FakeTorch:
            int32 = "int32"

            @staticmethod
            def device(name):
                return name

            class cuda:
                @staticmethod
                def is_available():
                    return False

            @staticmethod
            def tensor(values, device=None, dtype=None):
                return _FakeTensor(values[0])

        role = TpWorkerRole(
            local_rank=0,
            world_size=2,
            coordinator_rank=0,
            is_coordinator=True,
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "torch": _FakeTorch,
                "vllm.distributed": SimpleNamespace(
                    tensor_model_parallel_all_reduce=lambda tensor: _FakeTensor(1)
                ),
            },
        ):
            self.assertFalse(
                _all_tp_ranks_ready_for_direct_encode(True, role=role)
            )

        with mock.patch.dict(
            "sys.modules",
            {
                "torch": _FakeTorch,
                "vllm.distributed": SimpleNamespace(
                    tensor_model_parallel_all_reduce=lambda tensor: _FakeTensor(2)
                ),
            },
        ):
            self.assertTrue(
                _all_tp_ranks_ready_for_direct_encode(True, role=role)
            )

    def test_execute_mm_encoder_wrapper_falls_back_only_scheduled_subset(self) -> None:
        class FakeGPUModelRunner:
            def __init__(self) -> None:
                self.seen_scheduled = None

            def _update_states(self, scheduler_output):
                return None

            def _batch_mm_inputs_from_scheduler(self, scheduler_output):
                return None

            def _execute_mm_encoder(self, scheduler_output):
                self.seen_scheduled = dict(scheduler_output.scheduled_encoder_inputs)
                return "fallback-result"

        self.assertTrue(install_gpu_model_runner_patch(FakeGPUModelRunner))
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=(),
            scheduled_encoder_inputs={
                "req-direct": [0],
                "req-fallback": [1],
            },
        )
        result = VitDpDirectEncodeResult(
            handled_request_ids=("req-direct",),
            fallback_scheduled={"req-fallback": [1]},
        )

        with mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar.prepare_scheduled_mm_inputs_before_encoder",
            return_value=0,
        ), mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar._try_execute_vit_dp_sidecar_direct_encode",
            return_value=result,
        ):
            runner = FakeGPUModelRunner()
            self.assertEqual(
                runner._execute_mm_encoder(scheduler_output),
                "fallback-result",
            )

        self.assertEqual(runner.seen_scheduled, {"req-fallback": [1]})
        self.assertEqual(
            scheduler_output.scheduled_encoder_inputs,
            {
                "req-direct": [0],
                "req-fallback": [1],
            },
        )

    def test_execute_mm_encoder_wrapper_returns_empty_when_all_direct(self) -> None:
        class FakeGPUModelRunner:
            def _update_states(self, scheduler_output):
                return None

            def _batch_mm_inputs_from_scheduler(self, scheduler_output):
                return None

            def _execute_mm_encoder(self, scheduler_output):
                raise AssertionError("stock encoder should not run")

        self.assertTrue(install_gpu_model_runner_patch(FakeGPUModelRunner))
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=(),
            scheduled_encoder_inputs={"req-direct": [0]},
        )
        result = VitDpDirectEncodeResult(
            handled_request_ids=("req-direct",),
            fallback_scheduled={},
        )

        with mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar.prepare_scheduled_mm_inputs_before_encoder",
            return_value=0,
        ), mock.patch(
            "mm_sidecar.integrations.vllm_patch.worker_sidecar._try_execute_vit_dp_sidecar_direct_encode",
            return_value=result,
        ):
            self.assertEqual(FakeGPUModelRunner()._execute_mm_encoder(scheduler_output), [])

    def test_manual_encode_and_gather_local_items_reconstructs_original_order(self) -> None:
        class _FakeTensor:
            def __init__(self, values):
                self.values = list(values)
                self.shape = (len(self.values), 3)
                self.device = "cpu"
                self.dtype = "float32"

            def __getitem__(self, key):
                if isinstance(key, slice):
                    return _FakeTensor(self.values[key])
                return self.values[key]

            def contiguous(self):
                return self

        class _FakeTorch:
            float32 = "float32"

            @staticmethod
            def cat(items, dim=0):
                values = []
                for item in items:
                    values.extend(item.values)
                return _FakeTensor(values)

            @staticmethod
            def empty(shape, device=None, dtype=None):
                return _FakeTensor([None] * int(shape[0]))

        image_features = [
            SimpleNamespace(mm_position=SimpleNamespace(get_num_embeds=lambda: 2)),
            SimpleNamespace(mm_position=SimpleNamespace(get_num_embeds=lambda: 1)),
            SimpleNamespace(mm_position=SimpleNamespace(get_num_embeds=lambda: 3)),
        ]
        local_items = ["img2", "img1"]
        local_outputs = [_FakeTensor(["c0", "c1", "c2"]), _FakeTensor(["b0"])]

        model_runner = SimpleNamespace(
            device="cpu",
            pin_memory=False,
            model=SimpleNamespace(
                use_data_parallel=True,
                visual=SimpleNamespace(out_hidden_size=3, dtype="float32"),
                is_multimodal_pruning_enabled=False,
                embed_multimodal=lambda **kwargs: [local_outputs.pop(0)],
            ),
        )

        def fake_group_and_batch_mm_kwargs(local_mm_kwargs, device=None, pin_memory=False):
            for item in local_mm_kwargs:
                yield ("image", 1, {"image": [item]})

        with mock.patch.dict(
            "sys.modules",
            {
                "torch": _FakeTorch,
                "vllm.distributed": SimpleNamespace(
                    tensor_model_parallel_all_gather=lambda tensor, dim=0: _FakeTensor(
                        ["a0", "a1", "pad0", "pad1"] + tensor.values
                    )
                ),
                "vllm.multimodal.utils": SimpleNamespace(
                    group_and_batch_mm_kwargs=fake_group_and_batch_mm_kwargs
                ),
                "vllm.v1.worker.utils": SimpleNamespace(
                    sanity_check_mm_encoder_outputs=lambda outputs, expected_num_items: None
                ),
            },
        ):
            outputs = _manual_encode_and_gather_local_items(
                model_runner,
                image_features=tuple(image_features),
                local_indices=(2, 1),
                local_items=local_items,
                order=(0, 2, 1),
                counts=(1, 2),
            )

        self.assertEqual(outputs[0].values, ["a0", "a1"])
        self.assertEqual(outputs[1].values, ["b0"])
        self.assertEqual(outputs[2].values, ["c0", "c1", "c2"])


if __name__ == "__main__":
    unittest.main()
