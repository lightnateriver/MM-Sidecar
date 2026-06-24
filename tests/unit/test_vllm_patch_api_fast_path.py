from __future__ import annotations

import unittest
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from mm_sidecar.integrations.vllm_patch.api_fast_path import try_apply_api_fast_path
from mm_sidecar.integrations.vllm_patch.context import (
    RequestCapture,
    reset_current_capture,
    set_current_capture,
)
from mm_sidecar.integrations.vllm_patch.qwen_adapter import (
    get_request_payload_from_qwen_mm_kwargs_item,
    is_synthetic_qwen_mm_kwargs_item,
)


class _FakeMMDataItems:
    def __init__(self, count: int) -> None:
        self.count = count

    def get_all_counts(self) -> dict[str, int]:
        return {"image": self.count}


@dataclass
class _FakeInputs:
    prompt: str
    mm_data_items: _FakeMMDataItems
    hf_processor_mm_kwargs: dict[str, Any]
    tokenization_kwargs: dict[str, Any]


class _FakePlaceholderInfo:
    def __init__(self, offset: int, length: int) -> None:
        self.offset = offset
        self.length = length

    def to_range(self):
        from vllm.multimodal.inputs import PlaceholderRange

        return PlaceholderRange(offset=self.offset, length=self.length)


class Qwen3VLMultiModalProcessor:
    __module__ = "vllm.model_executor.models.qwen3_vl"

    def __init__(self) -> None:
        self.seen_mm_kwargs = None

    def _apply_hf_processor_text_only(self, prompt: str, tokenization_kwargs: dict):
        self.seen_tokenization_kwargs = tokenization_kwargs
        return [10, 20]

    def _apply_hf_processor_tokens_only(self, prompt):
        return list(prompt)

    def _get_mm_prompt_updates(
        self,
        mm_data_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ):
        self.seen_mm_kwargs = out_mm_kwargs
        return {"updates": "fake"}

    def _maybe_apply_prompt_updates(
        self,
        *,
        mm_items,
        prompt_ids,
        mm_kwargs,
        mm_prompt_updates,
        is_update_applied,
    ):
        self.seen_is_update_applied = is_update_applied
        return prompt_ids + [30], {"image": [_FakePlaceholderInfo(2, 180)]}


def _payload(image_count: int = 1) -> dict[str, Any]:
    signature = (
        "model=qwen3.5-vl|rev=test|processor=qwen3-vl|patch=14|merge=2"
        "|temporal=2|min_pixels=784|max_pixels=1003520|do_resize=1"
    )
    planned_items = [
        {
            "request_media_index": index,
            "item_identity": f"image-{index}",
            "source_key": f"local_path:/tmp/image-{index}.jpg",
            "transport": "local_path",
            "orig_size_hw": [512, 288],
            "preprocessed_size_hw": [504, 280],
            "image_grid_thw": [1, 36, 20],
            "placeholder_token_count": 180,
            "processor_signature": signature,
        }
        for index in range(image_count)
    ]
    return {
        "enabled": True,
        "prepared_image_count": image_count,
        "total_placeholder_token_count": 180 * image_count,
        "processor_signature": signature,
        "planned_items": planned_items,
        "handles": [
            {
                "request_id": "req-api-fast",
                "request_media_index": index,
                "cache_key": f"cache-key-{index}",
                "epoch": 0,
            }
            for index in range(image_count)
        ],
    }


class VllmPatchApiFastPathTests(unittest.TestCase):
    def setUp(self) -> None:
        if find_spec("packaging") is None:
            self.skipTest("local vLLM import requires packaging")

    def test_try_apply_api_fast_path_builds_synthetic_mm_input(self) -> None:
        capture = RequestCapture(
            request_id="req-api-fast",
            method="POST",
            path="/v1/chat/completions",
        )
        capture.set_sidecar_prepare(_payload())
        token = set_current_capture(capture)
        try:
            processor = Qwen3VLMultiModalProcessor()
            result = try_apply_api_fast_path(
                processor,
                _FakeInputs(
                    prompt="<|vision_start|><|image_pad|><|vision_end|>",
                    mm_data_items=_FakeMMDataItems(1),
                    hf_processor_mm_kwargs={},
                    tokenization_kwargs={"add_special_tokens": False},
                ),
                timing_ctx=None,
            )
        finally:
            reset_current_capture(token)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["type"], "multimodal")
        self.assertEqual(result["prompt_token_ids"], [10, 20, 30])
        self.assertEqual(result["mm_hashes"], {"image": ["cache-key-0"]})
        self.assertEqual(len(result["mm_placeholders"]["image"]), 1)
        item = result["mm_kwargs"]["image"][0]
        self.assertTrue(is_synthetic_qwen_mm_kwargs_item(item))
        self.assertIs(get_request_payload_from_qwen_mm_kwargs_item(item), capture.sidecar_prepare)
        self.assertEqual(tuple(item["pixel_values"].data.shape), (0, 1176))
        self.assertEqual(item["image_grid_thw"].data.tolist(), [1, 36, 20])
        self.assertIs(processor.seen_mm_kwargs, result["mm_kwargs"])
        self.assertFalse(processor.seen_is_update_applied)
        self.assertTrue(capture.sidecar_prepare["api_fast_path"]["used"])

    def test_try_apply_api_fast_path_rejects_missing_planned_items(self) -> None:
        capture = RequestCapture(
            request_id="req-api-fast-mismatch",
            method="POST",
            path="/v1/chat/completions",
        )
        payload = _payload(image_count=1)
        payload["planned_items"] = []
        capture.set_sidecar_prepare(payload)
        token = set_current_capture(capture)
        try:
            result = try_apply_api_fast_path(
                Qwen3VLMultiModalProcessor(),
                _FakeInputs(
                    prompt="prompt",
                    mm_data_items=_FakeMMDataItems(1),
                    hf_processor_mm_kwargs={},
                    tokenization_kwargs={},
                ),
                timing_ctx=None,
            )
        finally:
            reset_current_capture(token)

        self.assertIsNone(result)
        self.assertFalse(capture.sidecar_prepare["api_fast_path"]["used"])
        self.assertEqual(
            capture.sidecar_prepare["api_fast_path"]["reason"],
            "planned_item_count_mismatch",
        )


if __name__ == "__main__":
    unittest.main()
