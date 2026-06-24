from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
    bind_request_mm_sidecar,
    build_worker_source_plan,
    prepare_scheduled_mm_inputs_before_encoder,
    install_gpu_model_runner_patch,
    reset_worker_sidecar_client_cache,
    try_replace_scheduled_mm_inputs_from_sidecar,
)
from mm_sidecar.sidecar import InlineProcessorWorkerPool, SidecarManager
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
    def __init__(self, req_id: str) -> None:
        self.scheduled_new_reqs = [_FakeNewReqData(req_id)]
        self.scheduled_encoder_inputs = {req_id: [0]}


class _FakeFeature:
    modality = "image"
    data = "native-data"


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

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
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

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
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

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
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

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
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

            manager = SidecarManager(worker_pool=InlineProcessorWorkerPool())
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
            manager.close()

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


if __name__ == "__main__":
    unittest.main()
