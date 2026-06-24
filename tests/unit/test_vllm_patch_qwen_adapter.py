from __future__ import annotations

import unittest
from importlib.util import find_spec

import numpy as np

from mm_sidecar.contracts import (
    ArtifactDescriptor,
    ImageTensorPayload,
    ProcessorConfig,
    ProcessorSignature,
    StorageKind,
)
from mm_sidecar.integrations.vllm_patch.qwen_adapter import (
    is_synthetic_qwen_mm_kwargs_item,
    planned_item_to_synthetic_qwen_mm_kwargs_item,
    replace_feature_data_from_sidecar_artifacts,
    sidecar_artifact_to_qwen_mm_kwargs_item,
)
from mm_sidecar.sidecar.protocol import PreparedArtifact, SidecarHandle


def _make_signature() -> ProcessorSignature:
    return ProcessorSignature.from_config(
        ProcessorConfig(
            model_name="qwen3.5-vl",
            revision="adapter-test",
            processor_name="qwen3-vl",
            patch_size=14,
            merge_size=2,
            temporal_patch_size=1,
            min_pixels=28 * 28,
            max_pixels=1280 * 28 * 28,
        )
    )


def _make_artifact(index: int = 0) -> PreparedArtifact:
    signature = _make_signature()
    grid_thw = (1, 36, 20)
    pixel_values = np.arange(720 * 588, dtype=np.float32).reshape(720, 588)
    descriptor = ArtifactDescriptor(
        artifact_id="artifact-adapter",
        item_identity="image-adapter",
        processor_signature=signature,
        image_grid_thw=grid_thw,
        payload_shape=(720, 588),
        payload_dtype="float32",
        storage_kind=StorageKind.CPU_MEMORY,
        payload_nbytes=int(pixel_values.nbytes),
    )
    payload = ImageTensorPayload(
        pixel_values=pixel_values,
        image_grid_thw=grid_thw,
        payload_shape=(720, 588),
        payload_dtype="float32",
        storage_kind=StorageKind.CPU_MEMORY,
        resized_size_hw=(504, 280),
        orig_size_hw=(512, 288),
    )
    return PreparedArtifact(
        handle=SidecarHandle(
            request_id="req-adapter",
            request_media_index=index,
            cache_key="cache-adapter",
            epoch=0,
        ),
        descriptor=descriptor,
        payload=payload,
        timings_ms={"total": 1.0},
    )


class _FakeFeature:
    def __init__(self, modality: str = "image") -> None:
        self.modality = modality
        self.data = None


class _FakeReqState:
    def __init__(self) -> None:
        self.mm_features = [_FakeFeature()]


class VllmPatchQwenAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        if find_spec("packaging") is None:
            self.skipTest("local vLLM import requires packaging")

    def test_sidecar_artifact_to_qwen_mm_kwargs_item_shapes(self) -> None:
        item = sidecar_artifact_to_qwen_mm_kwargs_item(_make_artifact())

        self.assertIn("pixel_values", item)
        self.assertIn("image_grid_thw", item)
        self.assertEqual(tuple(item["pixel_values"].data.shape), (720, 588))
        self.assertEqual(tuple(item["image_grid_thw"].data.shape), (3,))
        self.assertEqual(str(item["pixel_values"].data.dtype), "torch.float32")
        self.assertEqual(item["image_grid_thw"].data.tolist(), [1, 36, 20])
        self.assertTrue(item["image_grid_thw"].field.keep_on_cpu)
        self.assertFalse(is_synthetic_qwen_mm_kwargs_item(item))

    def test_planned_item_to_synthetic_qwen_mm_kwargs_item(self) -> None:
        signature = _make_signature().value.replace("|temporal=1|", "|temporal=2|")
        item = planned_item_to_synthetic_qwen_mm_kwargs_item(
            {
                "request_media_index": 0,
                "image_grid_thw": [1, 36, 20],
                "processor_signature": signature,
            },
            processor_signature=None,
        )

        self.assertTrue(is_synthetic_qwen_mm_kwargs_item(item))
        self.assertIn("pixel_values", item)
        self.assertIn("image_grid_thw", item)
        self.assertEqual(tuple(item["pixel_values"].data.shape), (0, 1176))
        self.assertEqual(item["image_grid_thw"].data.tolist(), [1, 36, 20])
        self.assertTrue(item["image_grid_thw"].field.keep_on_cpu)

    def test_replace_feature_data_from_sidecar_artifacts(self) -> None:
        req_state = _FakeReqState()
        replaced = replace_feature_data_from_sidecar_artifacts(
            req_state,
            [_make_artifact()],
        )

        self.assertEqual(replaced, 1)
        self.assertIsNotNone(req_state.mm_features[0].data)
        self.assertIn("pixel_values", req_state.mm_features[0].data)


if __name__ == "__main__":
    unittest.main()
